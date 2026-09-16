from __future__ import annotations

import sys
import unittest
from pathlib import Path

EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import db_fakes

db_fakes.install_stubs()
_Connection = db_fakes._Connection
_ManyConnection = db_fakes._ManyConnection

from embedder.domain.photo import EmbeddingResult, PhotoRef
from embedder.repository import photos


class FetchTargetsTest(unittest.TestCase):
    """앱의 휴지통(deleted_at)·업로드 상태와 맞물리는 선별 조건이 SQL에 남아 있는지 지킨다.

    V17이 studio_deletion_claims를 DROP했을 때 Lambda가 그 테이블을 계속 읽어 전량 실패한 적이
    있다. 스키마 계약이 바뀌면 이 테스트가 먼저 깨져야 한다.
    """

    def test_skips_pending_and_trashed_rows(self) -> None:
        connection = _Connection(rows=[(1, "galleries/7/a.jpg"), (2, "galleries/7/b.jpg")])

        targets = photos.fetch_targets(connection, gallery_id=7)

        sql, params = connection.executed[0]
        self.assertEqual((7,), params)
        self.assertIn("JOIN galleries g ON g.id = p.gallery_id", sql)
        self.assertIn("p.status <> 'PENDING'", sql)
        self.assertIn("p.deleted_at IS NULL", sql)
        self.assertIn("g.deleted_at IS NULL", sql)
        self.assertNotIn("studio_deletion_claims", sql)
        self.assertEqual(
            [PhotoRef(1, "galleries/7/a.jpg"), PhotoRef(2, "galleries/7/b.jpg")],
            targets,
        )

    def test_default_run_only_picks_photos_without_embedding(self) -> None:
        # V29부터 벡터는 photos가 아니라 photo_analysis에 있다. photos.embedding을 계속 읽으면
        # 컬럼이 없어 전량 실패한다.
        connection = _Connection(rows=[])

        photos.fetch_targets(connection, gallery_id=7)

        sql, _ = connection.executed[0]
        self.assertNotIn("p.embedding", sql)
        self.assertIn(
            "NOT EXISTS (SELECT 1 FROM photo_analysis a WHERE a.photo_id = p.id AND a.embedding IS NOT NULL)",
            sql,
        )
        self.assertTrue(sql.endswith("ORDER BY p.id"))


class StoreEmbeddingsTest(unittest.TestCase):
    """벡터는 photo_analysis에, 파생본·EXIF·상태는 photos에 -- 두 문장이 한 배치에서 함께 나가는지 지킨다."""

    def test_writes_vector_to_photo_analysis_and_status_to_photos(self) -> None:
        connection = _ManyConnection(rows=[])
        ref = PhotoRef(1, "galleries/7/a.jpg")

        stored = photos.store_embeddings(
            connection,
            [EmbeddingResult(ref, "VECTOR", "previews/galleries/7/a.jpg", None)],
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
        self.assertNotIn("status", photos_sql)                     # V15 계약(#83): 기본은 status 를 안 쓴다
        self.assertEqual(1, photo_rows[0][-2])
        self.assertEqual("galleries/7/a.jpg", photo_rows[0][-1])

    def test_preview_key_is_overwritten_and_exif_is_coalesced(self) -> None:
        # 벡터는 이번에 올린 미리보기 파일에서 나오므로(#24) preview_key는 이전 값을 지킬 이유가
        # 없다. EXIF만 추출 실패 시 이전 값을 지키는 COALESCE다.
        connection = _ManyConnection(rows=[])
        ref = PhotoRef(1, "galleries/7/a.jpg")

        photos.store_embeddings(connection, [EmbeddingResult(ref, "VECTOR", "previews/galleries/7/a.jpg", None)], model_id="m")

        photos_sql, photo_rows = connection.executed[1]
        self.assertIn("SET preview_key = %s,", photos_sql)
        self.assertNotIn("COALESCE(%s, preview_key)", photos_sql)
        self.assertIn("taken_at = COALESCE(%s, taken_at)", photos_sql)
        self.assertEqual("previews/galleries/7/a.jpg", photo_rows[0][0])

    def test_empty_batch_writes_nothing(self) -> None:
        connection = _ManyConnection(rows=[])

        self.assertEqual(0, photos.store_embeddings(connection, [], model_id="facebook/dinov3-vitb16-pretrain-lvd1689m"))
        self.assertEqual([], connection.executed)

    def test_store_cas_uses_fetched_storage_key_and_active_resource_boundaries(self) -> None:
        connection = _Connection(rows=[], rowcounts=[1, 1])
        ref = PhotoRef(17, "galleries/7/original-before-replacement.jpg")

        stored = photos.store_embeddings(connection, [EmbeddingResult(ref, [0.1, 0.2], None, None)], model_id="test-model")

        self.assertEqual(1, stored)
        sql, rows = connection.executed[1]
        self.assertIn("WHERE id = %s AND storage_key = %s AND deleted_at IS NULL", sql)
        self.assertIn("g.id = photos.gallery_id AND g.deleted_at IS NULL", sql)
        self.assertEqual(17, rows[0][-2])
        self.assertEqual("galleries/7/original-before-replacement.jpg", rows[0][-1])

    def test_store_ignores_stale_photo_replaced_after_fetch(self) -> None:
        connection = _Connection(rows=[], rowcounts=[1, 0])
        old_ref = PhotoRef(17, "galleries/7/old.jpg")

        stored = photos.store_embeddings(connection, [EmbeddingResult(old_ref, [0.1, 0.2], "previews/old.jpg", None)], model_id="test-model")

        self.assertEqual(0, stored)


class FetchByIdsTest(unittest.TestCase):
    """v2(#73): id 목록 조회는 status 를 보지 않고 휴지통·storage_key 만 거르며, 요청 순서를 지킨다."""

    def test_filters_and_keeps_request_order(self) -> None:
        connection = _Connection(rows=[(1, "galleries/7/a.jpg"), (3, "galleries/7/c.jpg")])
        refs = photos.fetch_by_ids(connection, [3, 1, 2])
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
        self.assertEqual([], photos.fetch_by_ids(connection, []))
        self.assertEqual([], connection.executed)


class NeverWritesPhotoStatusTest(unittest.TestCase):
    """wes V15(#100): photos.status 는 PENDING·UPLOADED 뿐이고 embedder 역할에 그 컬럼 UPDATE 권한이 없다.
    적재 경로 어디에서도 status 를 쓰지 않는다."""

    def _ref(self):
        return PhotoRef(photo_id=1, storage_key="galleries/7/a.jpg")

    def test_store_embeddings_never_touches_status(self) -> None:
        connection = _Connection(rows=[])
        photos.store_embeddings(connection, [EmbeddingResult(self._ref(), "VECTOR", "previews/galleries/7/a.jpg", None)], model_id="m")
        photos_sql, _ = connection.executed[1]
        self.assertNotIn("status", photos_sql)
        self.assertIn("preview_key = %s", photos_sql)


if __name__ == "__main__":
    unittest.main()
