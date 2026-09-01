"""DbStore 왕복 테스트 — wes Flyway(V29)가 적용된 Postgres가 필요하다.

DB_HOST 가 없으면 건너뛴다. 로컬: wes `docker-compose.local.yml` pg 에 wes 를 한 번 bootRun 해
스키마를 올린 뒤
    DB_HOST=localhost DB_NAME=wes DB_USER=wes DB_PASSWORD=wes DB_SSLMODE=disable cd photoselect && pytest tests/test_db_store.py

고정하는 계약:
  · write_analysis 는 분석 컬럼만 쓰고 embedding(임베더 DINOv3)은 건드리지 않는다
  · read_embeddings 는 임베더 벡터를 돌려준다 (CLIP 아님)
  · read_evidence 는 items·pairs·rejected 를 읽고 ratings 는 비어 있다
  · write/read_recommendations 는 selection 단위다
  · jobs.claim 은 PENDING 만 집고, finish 는 round·result 를 남긴다
"""

from __future__ import annotations

import os
import uuid

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("DB_HOST"), reason="DB_HOST 없음 — wes pg 필요")


@pytest.fixture
def conn():
    from photoselect import db
    from photoselect.v1.config import Settings
    c = db.connect(Settings.from_env())
    yield c
    c.rollback()
    c.close()


@pytest.fixture
def seed(conn):
    """users → studios → galleries → photos(3, EMBEDDED, DINOv3 벡터) → photo_selections. 끝나면 지운다."""
    tag = f"dbstore-{uuid.uuid4().hex[:8]}"   # 유니크 컬럼(provider_id, gallery_url)이 앞선 실행과 안 겹치게
    with conn.cursor() as cur:
        cur.execute("INSERT INTO users (provider, provider_id, nickname, role, created_at, updated_at) "
                    "VALUES ('GOOGLE', %s, 'dbstore', 'USER', now(), now()) RETURNING id", (tag,))
        user_id = cur.fetchone()[0]
        cur.execute("INSERT INTO studios (user_id, name, gallery_url, created_at, updated_at) "
                    "VALUES (%s, 'dbstore', %s, now(), now()) RETURNING id", (user_id, tag))
        studio_id = cur.fetchone()[0]
        cur.execute("INSERT INTO galleries (studio_id, title, status, created_at, updated_at) "
                    "VALUES (%s, 'dbstore', 'OPEN', now(), now()) RETURNING id", (studio_id,))
        gallery_id = cur.fetchone()[0]
        photo_ids = []
        for i in range(3):
            cur.execute("INSERT INTO photos (gallery_id, storage_key, original_file_name, display_order, status, "
                        "content_type, preview_key, created_at, updated_at) VALUES (%s, %s, %s, %s, 'EMBEDDED', "
                        "'image/jpeg', %s, now(), now()) RETURNING id",
                        (gallery_id, f"dbstore-test/{gallery_id}/{i}.jpg", f"{i}.jpg", i, f"previews/{i}.jpg"))
            photo_ids.append(cur.fetchone()[0])
        for i, pid in enumerate(photo_ids):
            vec = np.zeros(768, dtype=np.float32); vec[i] = 1.0
            cur.execute("INSERT INTO photo_analysis (photo_id, embedding, embedding_model, created_at, updated_at) "
                        "VALUES (%s, %s, 'facebook/dinov3-vitb16-pretrain-lvd1689m', now(), now())", (pid, vec))
        cur.execute("INSERT INTO photo_selections (gallery_id, status, created_at, updated_at) "
                    "VALUES (%s, 'SELECTING', now(), now()) RETURNING id", (gallery_id,))
        selection_id = cur.fetchone()[0]
    conn.commit()
    yield {"user": user_id, "gallery": str(gallery_id), "photos": [str(p) for p in photo_ids],
           "selection": str(selection_id)}
    conn.rollback()
    with conn.cursor() as cur:   # galleries 이하는 CASCADE. studios·users 는 명시적으로
        cur.execute("DELETE FROM galleries WHERE id = %s", (gallery_id,))
        cur.execute("DELETE FROM studios WHERE id = %s", (studio_id,))
        cur.execute("DELETE FROM users WHERE id = %s", (user_id,))
    conn.commit()


def _rows(photo_ids):
    from photoselect.v1.store import PhotoAnalysis
    return [PhotoAnalysis(photo_id=p, scene="ceremony", framing="close_up", lighting="natural",
                          expression="smile", subjects="couple", caption=f"cap {p}",
                          technical_pct=10.0 * i, aesthetic_pct=90.0 - i,
                          face_boxes={"face_count": 2},
                          sub_scores={"technical_score": 0.5, "eyes_open": float("nan")},   # 얼굴 없음 → NaN
                          cluster_id=i // 2, cluster_rank=i % 2, rank_reason_code="eyes_open",
                          model_version="test-v1")
            for i, p in enumerate(photo_ids)]


def test_write_analysis_keeps_embedder_vectors(conn, seed):
    from photoselect.v1.config import Settings
    from photoselect.v1.store import DbStore
    st = DbStore(Settings.from_env(), connection=conn)
    g, pids = seed["gallery"], seed["photos"]

    assert st.read_analysis(g) == []                      # 임베딩만 있는 행은 '분석됨'이 아니다
    ids, E = st.read_embeddings(g)
    assert ids == pids and E.shape == (3, 768) and E[1, 1] == 1.0

    clip = np.full((3, 768), 0.5, dtype=np.float32)         # 저장되면 안 되는 벡터
    st.write_analysis(g, _rows(pids), (pids, clip))

    rows = st.read_analysis(g)
    assert [r.photo_id for r in rows] == pids
    assert rows[2].technical_pct == 20.0 and rows[2].cluster_id == 1 and rows[2].cluster_rank == 0
    assert rows[0].face_boxes == {"face_count": 2} and rows[0].caption == f"cap {pids[0]}"
    assert rows[0].sub_scores == {"technical_score": 0.5, "eyes_open": None}   # NaN 은 jsonb null 로
    ids2, E2 = st.read_embeddings(g)
    assert np.array_equal(E, E2), "write_analysis 가 임베더 벡터를 덮어썼다"

    # 재실행(UPSERT)은 행을 늘리지 않고 값만 갈아 끼운다
    again = _rows(pids); again[0].scene = "reception"
    st.write_analysis(g, again, (pids, clip))
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), max(version) FROM photo_analysis WHERE photo_id = ANY(%s)",
                    ([int(p) for p in pids],))
        n, ver = cur.fetchone()
    assert n == 3 and ver == 2
    assert st.read_analysis(g)[0].scene == "reception"


def test_evidence_and_recommendations_are_per_selection(conn, seed):
    from photoselect.v1.config import Settings
    from photoselect.v1.store import DbStore, Recommendation
    g, pids, sid = seed["gallery"], seed["photos"], seed["selection"]
    st = DbStore(Settings.from_env(), selection_id=sid, connection=conn)
    st.write_analysis(g, _rows(pids), (pids, np.zeros((3, 768))))

    assert st.gallery_of_selection(sid) == g
    assert st.target_count(g) is None
    with conn.cursor() as cur:
        cur.execute("UPDATE galleries SET max_selectable_photo_count = 40 WHERE id = %s", (int(g),))
    assert st.target_count(g) == 40
    assert st.read_recommendations(g) == []
    st.write_recommendations(g, [
        Recommendation(photo_id=pids[2], round=1, rank=0, score_breakdown={"score": 0.9}, reason="좋다"),
        Recommendation(photo_id=pids[0], round=1, rank=1, score_breakdown={"score": 0.4}, reason=""),
    ])
    recs = st.read_recommendations(g)
    assert [(r.photo_id, r.rank) for r in recs] == [(pids[2], 0), (pids[0], 1)]
    assert recs[0].score_breakdown == {"score": 0.9} and recs[1].reason == ""

    with conn.cursor() as cur:   # 부부의 반응은 wes API 가 쓴다 — 여기서는 흉내
        cur.execute("INSERT INTO photo_selection_items (selection_id, photo_id, created_at, updated_at) "
                    "VALUES (%s, %s, now(), now())", (int(sid), int(pids[2])))
        cur.execute("UPDATE ai_recommendations SET rejected_at = now() WHERE selection_id = %s AND photo_id = %s",
                    (int(sid), int(pids[0])))
        cur.execute("INSERT INTO pair_comparison_events (selection_id, photo_a, photo_b, chosen_photo_id, axis, "
                    "answered_by, created_at, updated_at) VALUES (%s, %s, %s, %s, 'framing', %s, now(), now())",
                    (int(sid), int(pids[0]), int(pids[1]), int(pids[1]), seed["user"]))
    conn.commit()

    ev = st.read_evidence(g)
    assert ev.selected == [pids[2]]
    assert ev.rejected == [pids[0]]
    assert ev.pairs == [(pids[1], pids[0], "framing")]
    assert ev.ratings == {}                                # photo_ratings 는 읽지 않는다
    assert DbStore(Settings.from_env(), connection=conn).read_evidence(g, None).is_empty()


def test_jobs_claim_and_finish(conn, seed):
    from photoselect import jobs
    sid = int(seed["selection"])
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ai_selection_jobs (selection_id, mode, status, created_at, updated_at) "
                    "VALUES (%s, 'draft', 'PENDING', now(), now()) RETURNING id", (sid,))
        job_id = cur.fetchone()[0]
    conn.commit()

    assert jobs.selection_job_info(conn, job_id) == (sid, "draft")
    assert jobs.claim(conn, jobs.SELECTION, job_id) is True
    assert jobs.claim(conn, jobs.SELECTION, job_id) is False      # 두 번은 못 집는다
    jobs.finish(conn, jobs.SELECTION, job_id, {"k": 2}, round_no=1)
    with conn.cursor() as cur:
        cur.execute("SELECT status, round, result, started_at IS NOT NULL, finished_at IS NOT NULL "
                    "FROM ai_selection_jobs WHERE id = %s", (job_id,))
        assert cur.fetchone() == ("DONE", 1, {"k": 2}, True, True)

    with conn.cursor() as cur:
        cur.execute("INSERT INTO ai_selection_jobs (selection_id, mode, status, created_at, updated_at) "
                    "VALUES (%s, 'refine', 'PENDING', now(), now()) RETURNING id", (sid,))
        job2 = cur.fetchone()[0]
    conn.commit()
    assert jobs.claim_next(conn, jobs.SELECTION) == job2
    jobs.fail(conn, jobs.SELECTION, job2, "boom")
    with conn.cursor() as cur:
        cur.execute("SELECT status, error FROM ai_selection_jobs WHERE id = %s", (job2,))
        assert cur.fetchone() == ("FAILED", "boom")


def test_worker_drains_pending_jobs(conn, seed, monkeypatch):
    """워커 한 바퀴: 분석 잡은 S3 없이 FAILED 로 닫히고(워커는 안 죽는다), 추천 잡은 DONE + 추천 행."""
    from photoselect import worker
    from photoselect.v1.config import Settings
    from photoselect.v1.store import DbStore
    g, pids, sid = seed["gallery"], seed["photos"], seed["selection"]
    DbStore(Settings.from_env(), connection=conn).write_analysis(g, _rows(pids), (pids, np.zeros((3, 768))))
    with conn.cursor() as cur:
        cur.execute("INSERT INTO ai_analysis_jobs (gallery_id, status, created_at, updated_at, version) "
                    "VALUES (%s, 'PENDING', now(), now(), 0) RETURNING id", (int(g),))
        a_job = cur.fetchone()[0]
        cur.execute("INSERT INTO ai_selection_jobs (selection_id, mode, status, created_at, updated_at, version) "
                    "VALUES (%s, 'draft', 'PENDING', now(), now(), 0) RETURNING id", (int(sid),))
        s_job = cur.fetchone()[0]
    conn.commit()
    monkeypatch.delenv("S3_BUCKET", raising=False)

    worker.loop(Settings.from_env(), use_vlm=False, once=True)

    with conn.cursor() as cur:
        cur.execute("SELECT status, error FROM ai_analysis_jobs WHERE id = %s", (a_job,))
        a_status, a_err = cur.fetchone()
        cur.execute("SELECT status, round, result FROM ai_selection_jobs WHERE id = %s", (s_job,))
        s_status, s_round, s_result = cur.fetchone()
        cur.execute("SELECT count(*) FROM ai_recommendations WHERE selection_id = %s", (int(sid),))
        n_recs = cur.fetchone()[0]
    assert a_status == "FAILED" and "S3_BUCKET" in a_err
    assert s_status == "DONE" and s_round == 1 and s_result["round"] == 1
    assert n_recs >= 1 and n_recs == s_result["k"]   # 연사 클러스터(0,1)는 대표 한 장만 → 3장 중 2장


def test_read_embeddings_rejects_mixed_models(conn, seed):
    """임베더 모델 교체(DINOv2→v3) 중 일부만 재임베딩된 갤러리는 클러스터를 돌리면 안 된다."""
    with conn.cursor() as cur:
        cur.execute("UPDATE photo_analysis SET embedding_model = 'facebook/dinov2-base' WHERE photo_id = %s",
                    (int(seed["photos"][0]),))
    from photoselect.v1.config import Settings
    from photoselect.v1.store import DbStore
    st = DbStore(Settings.from_env(), connection=conn)
    with pytest.raises(RuntimeError, match="embedding_model"):
        st.read_embeddings(seed["gallery"])
