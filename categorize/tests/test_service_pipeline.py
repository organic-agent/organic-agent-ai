"""pipeline — 순수 로직(concat · 대표 순위)과 그룹 → naming 흐름. 갤러리는 한 번만 읽는다."""

from __future__ import annotations

import numpy as np

from categorize.domain.analysis import PhotoAnalysis
from categorize.domain.photo import PhotoRef
from categorize.service import pipeline
from categorize.service.pipeline import assign_ranks, concat_space
from tests.helpers import CountingStore, FakeLlm, unit, with_knobs, world


# ── 순수 로직 ────────────────────────────────────────────────────────────────
def test_concat_space_is_mean_of_cosines():
    rng = np.random.default_rng(1)
    E = np.stack([unit(rng.normal(size=8)) for _ in range(4)])
    C = np.stack([unit(rng.normal(size=8)) for _ in range(4)])
    X = concat_space(E, C)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
    want = (E @ E.T + C @ C.T) / 2
    np.testing.assert_allclose(X @ X.T, want, atol=1e-5)


def test_assign_ranks_puts_reason_in_sub_scores():
    rows = [PhotoAnalysis(photo_id="a", technical_pct=90, sub_scores={"sharpness": 100.0}, cluster_id=0),
            PhotoAnalysis(photo_id="b", technical_pct=10, sub_scores={"sharpness": 10.0}, cluster_id=0)]
    assign_ranks(rows)
    best = next(r for r in rows if r.cluster_rank == 0)
    assert best.photo_id == "a"
    assert best.sub_scores["rank_reason"] == "technical"


# ── 그룹 → naming ────────────────────────────────────────────────────────────
def test_pipeline_groups_then_names(tmp_path, caplog):
    store, rows, E, C, settings = world(tmp_path)
    refs = [PhotoRef(photo_id=r.photo_id, path=None) for r in rows]   # 파일이 필요 없다
    settings = with_knobs(settings, group_distance=0.4, group_min_groups=2, group_max_share=0.6)

    with caplog.at_level("INFO", logger="categorize.service.pipeline"):
        result = pipeline.run(store, "g", refs, settings, FakeLlm(), job_id=None)

    # 그룹 단계 45s 를 읽기/연사/계층으로 나누는 로그 두 줄(#111) — 운영 CloudWatch 에서 이 줄을 찾는다
    stage_logs = [m for m in caplog.messages if m.startswith("[categorize] 갤러리 g ")]
    assert len(stage_logs) == 2
    assert stage_logs[0].startswith(f"[categorize] 갤러리 g 읽기: {len(rows)}행 · dinov3 {len(rows)} · clip {len(rows)} · ")
    assert stage_logs[1].startswith(f"[categorize] 갤러리 g 그룹화: {len(rows)}장 · 연사 {len(rows)} (")
    assert f"· 그룹 {int(result['groups']['groups'])} (" in stage_logs[1] and "읽기 뒤 누적" in stage_logs[1]

    assert result["mode"] == "categorize" and result["photos"] == len(rows)
    assert result["embeddingsSource"] == "dinov3"
    assert result["clusters"] == len(rows)
    assert result["naming"]["vlmGroups"] == result["groups"]["groups"]
    back = store.read_analysis("g")
    assert all(r.embed_group_id >= 0 and "sharpness_pct" in r.sub_scores for r in back)
    assert all(a.assigned_by == "vlm" for a in store.read_assignments("g"))


def test_pipeline_falls_back_to_clip_when_no_embedder_vectors(tmp_path):
    store, rows, E, C, settings = world(tmp_path)
    ids = [r.photo_id for r in rows]
    store.write_analysis("g", rows, ([], np.zeros((0, 0))), (ids, C))
    refs = [PhotoRef(photo_id=r.photo_id, path=None) for r in rows]

    result = pipeline.run(store, "g", refs, settings, None)

    assert result["embeddingsSource"] == "clip"
    assert result["naming"] == "skipped (no --llm)"


def test_pipeline_reads_gallery_once_and_hands_grouped_to_naming(tmp_path):
    """그룹화가 읽은 행·concat 공간을 naming 이 그대로 받는다 — DB 를 두 번 읽지 않는다."""
    store, rows, *_, settings = world(tmp_path)
    counting = CountingStore(store)
    refs = [PhotoRef(photo_id=r.photo_id, path=None) for r in rows]
    settings = with_knobs(settings, group_distance=0.4, group_min_groups=2, group_max_share=0.6)

    result = pipeline.run(counting, "g", refs, settings, FakeLlm(), job_id=None)

    assert counting.reads == 1
    assert result["naming"]["vlmGroups"] == result["groups"]["groups"]
