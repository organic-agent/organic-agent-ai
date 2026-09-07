from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path


# 대상 선별 SQL 테스트는 이미지·벡터·DB 드라이버를 실행하지 않는다. Lambda 의존성을 전부
# 설치하지 않은 로컬/CI에서도 이 경계를 검증할 수 있도록 import 자리만 최소 대체한다.
EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))

try:
    import numpy  # noqa: F401
except ModuleNotFoundError:
    sys.modules["numpy"] = types.ModuleType("numpy")

try:
    import psycopg  # noqa: F401
except ModuleNotFoundError:
    psycopg_module = types.ModuleType("psycopg")
    psycopg_module.Error = RuntimeError
    sys.modules["psycopg"] = psycopg_module

try:
    from pgvector.psycopg import register_vector  # noqa: F401
except ModuleNotFoundError:
    pgvector_package = types.ModuleType("pgvector")
    pgvector_package.__path__ = []
    pgvector_psycopg = types.ModuleType("pgvector.psycopg")
    pgvector_psycopg.register_vector = lambda connection: None
    sys.modules["pgvector"] = pgvector_package
    sys.modules["pgvector.psycopg"] = pgvector_psycopg

metadata_module = types.ModuleType("embedder.metadata")
metadata_module.PhotoMetadata = type("PhotoMetadata", (), {})
sys.modules.setdefault("embedder.metadata", metadata_module)

from embedder import db
from embedder.admin_event import AdminPhotoEvent


class _Cursor:
    def __init__(self, connection: "_Connection"):
        self.connection = connection

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def execute(self, sql: str, params: tuple) -> None:
        self.connection.executed.append((" ".join(sql.split()), params))
        self.rowcount = self.connection.next_rowcount()

    def executemany(self, sql: str, params) -> None:
        rows = tuple(params)
        self.connection.executed.append((" ".join(sql.split()), rows))
        self.rowcount = self.connection.next_rowcount()

    def fetchall(self):
        return self.connection.rows

    def fetchone(self):
        return self.connection.rows[0] if self.connection.rows else None


class _Connection:
    def __init__(self, rows: list[tuple], rowcounts: list[int] | None = None):
        self.rows = rows
        self.executed: list[tuple[str, tuple]] = []
        self.rowcounts = iter(rowcounts or [])

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def next_rowcount(self) -> int:
        return next(self.rowcounts, len(self.rows))


class FetchTargetsTest(unittest.TestCase):
    """앱의 휴지통(deleted_at)·업로드 상태와 맞물리는 선별 조건이 SQL에 남아 있는지 지킨다.

    V17이 studio_deletion_claims를 DROP했을 때 Lambda가 그 테이블을 계속 읽어 전량 실패한 적이
    있다. 스키마 계약이 바뀌면 이 테스트가 먼저 깨져야 한다.
    """

    def test_skips_pending_and_trashed_rows(self) -> None:
        connection = _Connection(rows=[(1, "galleries/7/a.jpg"), (2, "galleries/7/b.jpg")])

        targets = db.fetch_targets(connection, gallery_id=7, force=False)

        sql, params = connection.executed[0]
        self.assertEqual((7,), params)
        self.assertIn("JOIN galleries g ON g.id = p.gallery_id", sql)
        self.assertIn("p.status <> 'PENDING'", sql)
        self.assertIn("p.deleted_at IS NULL", sql)
        self.assertIn("g.deleted_at IS NULL", sql)
        self.assertNotIn("studio_deletion_claims", sql)
        self.assertEqual(
            [db.PhotoRef(1, "galleries/7/a.jpg"), db.PhotoRef(2, "galleries/7/b.jpg")],
            targets,
        )

    def test_default_run_only_picks_photos_without_embedding(self) -> None:
        # V29부터 벡터는 photos가 아니라 photo_analysis에 있다. photos.embedding을 계속 읽으면
        # 컬럼이 없어 전량 실패한다.
        connection = _Connection(rows=[])

        db.fetch_targets(connection, gallery_id=7, force=False)

        sql, _ = connection.executed[0]
        self.assertNotIn("p.embedding", sql)
        self.assertIn(
            "NOT EXISTS (SELECT 1 FROM photo_analysis a WHERE a.photo_id = p.id AND a.embedding IS NOT NULL)",
            sql,
        )
        self.assertTrue(sql.endswith("ORDER BY p.id"))

    def test_since_skips_only_vectors_stored_in_this_run(self) -> None:
        """force 의 시작 시각(runStartedAt)이 오면 그 이후 적재된 벡터만 건너뛴다 — 재호출이 force 를 잃어도 옛 벡터는 재계산(#56)."""
        connection = _Connection(rows=[])

        db.fetch_targets(connection, gallery_id=7, force=False, since="2026-09-06T00:00:00+00:00")

        sql, params = connection.executed[0]
        self.assertIn(
            "NOT EXISTS (SELECT 1 FROM photo_analysis a WHERE a.photo_id = p.id"
            " AND a.embedding IS NOT NULL AND a.updated_at >= %s)",
            sql,
        )
        self.assertEqual((7, "2026-09-06T00:00:00+00:00"), params)
        self.assertTrue(sql.endswith("ORDER BY p.id"))

    def test_force_recomputes_everything(self) -> None:
        connection = _Connection(rows=[])

        db.fetch_targets(connection, gallery_id=7, force=True)

        sql, _ = connection.executed[0]
        self.assertNotIn("photo_analysis", sql)


class _ManyCursor(_Cursor):
    def executemany(self, sql: str, rows) -> None:
        rows = list(rows)
        self.connection.executed.append((" ".join(sql.split()), rows))
        # 실제 드라이버처럼 배치 크기를 갱신 행 수로 본다. rowcounts를 넘긴 테스트는 그 값을 쓴다.
        self.rowcount = next(self.connection.rowcounts, len(rows))


class _ManyConnection(_Connection):
    def cursor(self) -> _ManyCursor:
        return _ManyCursor(self)


class StoreEmbeddingsTest(unittest.TestCase):
    """벡터는 photo_analysis에, 파생본·EXIF·상태는 photos에 -- 두 문장이 한 배치에서 함께 나가는지 지킨다."""

    def test_writes_vector_to_photo_analysis_and_status_to_photos(self) -> None:
        connection = _ManyConnection(rows=[])
        ref = db.PhotoRef(1, "galleries/7/a.jpg")

        stored = db.store_embeddings(
            connection,
            [(ref, "VECTOR", "previews/galleries/7/a.jpg", None)],
            model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        )

        self.assertEqual(1, stored)
        analysis_sql, analysis_rows = connection.executed[0]
        photos_sql, photo_rows = connection.executed[1]
        self.assertIn("INSERT INTO photo_analysis (photo_id, embedding, embedding_model", analysis_sql)
        self.assertIn("ON CONFLICT (photo_id) DO UPDATE", analysis_sql)
        self.assertEqual([(1, "VECTOR", "facebook/dinov3-vitb16-pretrain-lvd1689m")], analysis_rows)
        self.assertIn("UPDATE photos", photos_sql)
        self.assertNotIn("embedding", photos_sql)
        self.assertIn("status = 'EMBEDDED'", photos_sql)
        self.assertEqual(1, photo_rows[0][-2])
        self.assertEqual("galleries/7/a.jpg", photo_rows[0][-1])

    def test_preview_key_is_overwritten_and_exif_is_coalesced(self) -> None:
        # 벡터는 이번에 올린 미리보기 파일에서 나오므로(#24) preview_key는 이전 값을 지킬 이유가
        # 없다. EXIF만 추출 실패 시 이전 값을 지키는 COALESCE다.
        connection = _ManyConnection(rows=[])
        ref = db.PhotoRef(1, "galleries/7/a.jpg")

        db.store_embeddings(connection, [(ref, "VECTOR", "previews/galleries/7/a.jpg", None)], model_id="m")

        photos_sql, photo_rows = connection.executed[1]
        self.assertIn("SET preview_key = %s,", photos_sql)
        self.assertNotIn("COALESCE(%s, preview_key)", photos_sql)
        self.assertIn("taken_at = COALESCE(%s, taken_at)", photos_sql)
        self.assertEqual("previews/galleries/7/a.jpg", photo_rows[0][0])

    def test_empty_batch_writes_nothing(self) -> None:
        connection = _ManyConnection(rows=[])

        self.assertEqual(0, db.store_embeddings(connection, [], model_id="facebook/dinov3-vitb16-pretrain-lvd1689m"))
        self.assertEqual([], connection.executed)

    def test_store_cas_uses_fetched_storage_key_and_active_resource_boundaries(self) -> None:
        connection = _Connection(rows=[], rowcounts=[1, 1])
        ref = db.PhotoRef(17, "galleries/7/original-before-replacement.jpg")

        stored = db.store_embeddings(connection, [(ref, [0.1, 0.2], None, None)], model_id="test-model")

        self.assertEqual(1, stored)
        sql, rows = connection.executed[1]
        self.assertIn("WHERE id = %s AND storage_key = %s AND deleted_at IS NULL", sql)
        self.assertIn("g.id = photos.gallery_id AND g.deleted_at IS NULL", sql)
        self.assertEqual(17, rows[0][-2])
        self.assertEqual("galleries/7/original-before-replacement.jpg", rows[0][-1])

    def test_store_ignores_stale_photo_replaced_after_fetch(self) -> None:
        connection = _Connection(rows=[], rowcounts=[1, 0])
        old_ref = db.PhotoRef(17, "galleries/7/old.jpg")

        stored = db.store_embeddings(connection, [(old_ref, [0.1, 0.2], "previews/old.jpg", None)], model_id="test-model")

        self.assertEqual(0, stored)


class GalleryLockTest(unittest.TestCase):
    def test_uses_session_advisory_lock_with_module_namespace(self) -> None:
        connection = _Connection(rows=[(True,)])

        self.assertTrue(db.try_lock_gallery(connection, 7))

        sql, params = connection.executed[0]
        self.assertIn("pg_try_advisory_lock(%s, %s)", sql)
        self.assertNotIn("xact", sql)   # 세션 수준이어야 배치 commit을 넘어 유지된다
        # 뒤쪽 키 = 갤러리 × LOCK_STRIDE + 샤드(#56). 샤드 없는 실행은 샤드 0 과 같은 키다.
        self.assertEqual((db.GALLERY_LOCK_NAMESPACE, 7 * db.LOCK_STRIDE), params)

    def test_shard_lock_key_is_gallery_times_stride_plus_index(self) -> None:
        connection = _Connection(rows=[(True,)])

        self.assertTrue(db.try_lock_gallery(connection, 7, shard_index=3))

        _, params = connection.executed[0]
        self.assertEqual((db.GALLERY_LOCK_NAMESPACE, 7 * db.LOCK_STRIDE + 3), params)
        self.assertNotEqual(7 * db.LOCK_STRIDE + 3, 8 * db.LOCK_STRIDE)   # 옆 갤러리와 겹치지 않는다

    def test_held_lock_returns_false(self) -> None:
        connection = _Connection(rows=[(False,)])

        self.assertFalse(db.try_lock_gallery(connection, 7))


class AdminPhotoJobDatabaseContractTest(unittest.TestCase):
    def event(self) -> AdminPhotoEvent:
        return AdminPhotoEvent(11, 2, "QUALITY_ANALYSIS", 31, 41, "galleries/41/photo.jpg", 51)

    def test_verification_cas_includes_job_attempt_revision_and_storage_key(self) -> None:
        connection = _Connection(rows=[(1,)])

        self.assertTrue(db.verify_admin_photo_event(connection, self.event()))

        sql, params = connection.executed[0]
        self.assertIn("j.attempt_count = %s", sql)
        self.assertIn("j.revision_id = %s", sql)
        self.assertIn("p.storage_key = %s", sql)
        self.assertIn("r.storage_key = %s", sql)
        self.assertEqual(11, params[0])
        self.assertEqual(2, params[1])

    def test_photo_result_and_job_success_are_both_required_before_commit(self) -> None:
        connection = _Connection(rows=[], rowcounts=[1, 0])
        result = type("Quality", (), {"score": 82.5, "signals": {"algorithmVersion": "technical-v1"}})()

        with self.assertRaises(db.AdminJobClaimLost):
            db.complete_admin_quality(connection, self.event(), result)

        self.assertEqual(2, len(connection.executed))
        self.assertIn("technical_quality_score", connection.executed[0][0])
        self.assertIn("status = 'SUCCEEDED'", connection.executed[1][0])

    def test_explicit_failure_only_updates_matching_exact_attempt(self) -> None:
        connection = _Connection(rows=[], rowcounts=[1])

        updated = db.fail_admin_photo_job(connection, self.event(), "NO_SUCH_KEY")

        self.assertEqual(1, updated)
        sql, params = connection.executed[0]
        self.assertIn("attempt_count = %s", sql)
        self.assertIn("revision_id = %s", sql)
        self.assertIn("status IN ('DISPATCHING', 'DISPATCHED')", sql)
        self.assertEqual("NO_SUCH_KEY", params[0])


if __name__ == "__main__":
    unittest.main()


class FetchByIdsTest(unittest.TestCase):
    """v2(#73): id 목록 조회는 status 를 보지 않고 휴지통·storage_key 만 거르며, 요청 순서를 지킨다."""

    def test_filters_and_keeps_request_order(self) -> None:
        connection = _Connection(rows=[(1, "galleries/7/a.jpg"), (3, "galleries/7/c.jpg")])
        refs = db.fetch_by_ids(connection, [3, 1, 2])
        sql, params = connection.executed[0]
        self.assertIn("p.id = ANY(%s)", sql)
        self.assertIn("p.storage_key IS NOT NULL", sql)
        self.assertIn("p.deleted_at IS NULL", sql)
        self.assertIn("g.deleted_at IS NULL", sql)
        self.assertNotIn("status", sql)
        self.assertEqual(([3, 1, 2],), params)
        self.assertEqual([(3, "galleries/7/c.jpg"), (1, "galleries/7/a.jpg")], [(r.photo_id, r.storage_key) for r in refs])

    def test_empty_ids_skip_the_query(self) -> None:
        connection = _Connection(rows=[])
        self.assertEqual([], db.fetch_by_ids(connection, []))
        self.assertEqual([], connection.executed)


class StoreEmbeddingsStatusSwitchTest(unittest.TestCase):
    """v2(#73): set_status=None 이면 photos.status 를 건드리지 않는다. 값은 대문자·밑줄만."""

    def _ref(self):
        return db.PhotoRef(photo_id=1, storage_key="galleries/7/a.jpg")

    def test_default_writes_embedded(self) -> None:
        connection = _Connection(rows=[])
        db.store_embeddings(connection, [(self._ref(), "VECTOR", "previews/galleries/7/a.jpg", None)], model_id="m")
        photos_sql, _ = connection.executed[1]
        self.assertIn("status = 'EMBEDDED'", photos_sql)

    def test_none_leaves_status_alone(self) -> None:
        connection = _Connection(rows=[])
        db.store_embeddings(connection, [(self._ref(), "VECTOR", "previews/galleries/7/a.jpg", None)], model_id="m",
                            set_status=None)
        photos_sql, _ = connection.executed[1]
        self.assertNotIn("status", photos_sql)
        self.assertIn("preview_key = %s", photos_sql)

    def test_rejects_odd_status_value(self) -> None:
        connection = _Connection(rows=[])
        with self.assertRaises(ValueError):
            db.store_embeddings(connection, [(self._ref(), "VECTOR", "p.jpg", None)], model_id="m", set_status="x'; --")
