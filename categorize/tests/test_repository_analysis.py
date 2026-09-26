"""DbStore — 갤러리 한 번 읽기(한 쿼리) · 대표 사진 배치 다운로드 · S3 풀 크기."""

from __future__ import annotations

import re

import numpy as np
import pytest

from categorize.config.settings import MODEL_VERSION, Settings
from categorize.domain.analysis import ConceptAssignment
from categorize.repository.analysis import PREVIEW_DOWNLOAD_WORKERS, DbStore
from categorize.repository.storage import PreviewStorage
from tests.db_fakes import JobConn, RowConn


def _db_store(tmp_path, conn, bucket=None):
    settings = Settings(out_root=tmp_path, dataset_root=tmp_path, work_dir=tmp_path / "work", s3_bucket=bucket)
    return DbStore(settings, connection=conn)


def test_db_store_read_gallery_is_one_query(tmp_path):
    """분석 행 + DINOv3 + CLIP 을 한 SELECT 로. model_version 없는 행은 벡터만 남고 행 목록에서 빠진다."""
    e, c = np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)
    conn = RowConn(rows=[
        (11, "couple", 50.0, 50.0, {"technical_score": 0.5}, -1, 0, -1, MODEL_VERSION, e, "dinov3", c),
        (12, "unknown", 50.0, 50.0, None, -1, 0, -1, None, e, "dinov3", None),          # 임베딩만, 점수 아직
    ])
    data = _db_store(tmp_path, conn).read_gallery("7")

    assert len(conn.executed) == 1
    sql, params = conn.executed[0]
    assert params == (7,)
    assert "a.embedding" in sql and "a.clip_embedding" in sql and "JOIN photos" in sql
    assert [r.photo_id for r in data.rows] == ["11"]
    assert data.rows[0].sub_scores == {"technical_score": 0.5}
    assert set(data.embeddings) == {"11", "12"} and set(data.clip_embeddings) == {"11"}


def test_db_store_read_gallery_rejects_mixed_embedding_models(tmp_path):
    e = np.ones(4, dtype=np.float32)
    conn = RowConn(rows=[
        (11, "couple", 50.0, 50.0, {}, -1, 0, -1, MODEL_VERSION, e, "dinov3", e),
        (12, "couple", 50.0, 50.0, {}, -1, 0, -1, MODEL_VERSION, e, "dinov2", e),
    ])
    with pytest.raises(RuntimeError, match="embedding_model"):
        _db_store(tmp_path, conn).read_gallery("7")


def test_db_store_preview_paths_batches_select_and_downloads_concurrently(tmp_path, monkeypatch):
    """preview_key 는 SELECT 한 번(ANY), 다운로드는 스레드풀 — 사진 수만큼 왕복하지 않는다."""
    import threading
    import time as _time

    conn = RowConn(rows=[(11, "previews/a.jpg"), (12, "previews/b.jpg"), (13, "previews/c.jpg")])
    store = _db_store(tmp_path, conn, bucket="bkt")
    seen_threads: set[int] = set()
    lock = threading.Lock()

    class _FakeStorage:
        def download(self, key, dest):
            with lock:
                seen_threads.add(threading.get_ident())
            _time.sleep(0.05)                      # 겹쳐 돌면 세 건이 0.15초가 아니라 ~0.05초
            if key.endswith("c.jpg"):
                raise OSError("no such key")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"jpeg")
            return dest

    store._storage = _FakeStorage()
    started = _time.monotonic()
    paths = store.preview_paths("7", ["11", "12", "13"])
    elapsed = _time.monotonic() - started

    assert len(conn.executed) == 1
    sql, params = conn.executed[0]
    assert "ANY(%s)" in sql and params == ([11, 12, 13],)
    assert set(paths) == {"11", "12"}                     # 실패한 13 은 빠진다
    assert len(seen_threads) > 1 and elapsed < 0.14

    # 이미 받아 둔 파일은 SELECT 도 다운로드도 없이 돌려준다
    again = store.preview_paths("7", ["11", "12"])
    assert again == paths and len(conn.executed) == 1


def test_preview_storage_pool_matches_download_workers(monkeypatch):
    """풀이 스레드 수보다 작으면 urllib3 가 'Connection pool is full' 을 찍는다(#109) — DbStore 는 PREVIEW_DOWNLOAD_WORKERS 를 그대로 넘긴다."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "ap-northeast-2")     # 클라이언트 생성만 — 자격·네트워크 없음
    pooled = PreviewStorage("bkt", max_concurrency=PREVIEW_DOWNLOAD_WORKERS)
    assert pooled._client._client_config.max_pool_connections == PREVIEW_DOWNLOAD_WORKERS == 16
    assert PreviewStorage("bkt")._client._client_config.max_pool_connections == 10   # 기본값은 boto 기본(10) 아래로 안 내려간다


def test_write_assignments_maps_layers_to_old_columns(tmp_path):
    """필드는 wes 층 이름, 컬럼은 옛 이름 — 1층 이름은 parent_name, 2층 이름은 concept_name 컬럼에 들어간다
    (proposed_concept → proposed_parent, clip_concept → clip_parent).
    컬럼 이름을 바꿀 때(wes Flyway) 이 테스트가 같이 바뀌어야 한다."""
    conn = JobConn()
    row = ConceptAssignment(embed_group_id=4, concept_name="기타", detail_name="해변",
                            confidence=0.9, assigned_by="vlm", proposed_concept="수영장", clip_concept="야외 자연")
    _db_store(tmp_path, conn).write_assignments("7", 3, [row])

    sql, params = conn.executed[0]
    cols = [c.strip() for c in re.search(r"INSERT INTO ai_concept_assignments \((.*?)\)", sql).group(1).split(",")]
    written = dict(zip(cols, params[0]))
    assert written["parent_name"] == "기타"          # 1층
    assert written["concept_name"] == "해변"         # 2층
    assert written["proposed_parent"] == "수영장"    # 1층 제안
    assert written["clip_parent"] == "야외 자연"     # 1층 검증 라벨
    assert (written["job_id"], written["gallery_id"], written["embed_group_id"]) == (3, 7, 4)
    assert conn.commits == 1
