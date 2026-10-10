"""pipeline — 순수 로직(concat · 대표 순위)과 구간 → 1층 → 2층 → 이름 흐름. 갤러리는 한 번만 읽는다."""

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
    np.testing.assert_allclose(X @ X.T, (E @ E.T + C @ C.T) / 2, atol=1e-5)


def test_assign_ranks_puts_reason_in_sub_scores():
    rows = [PhotoAnalysis(photo_id="a", technical_pct=90, aesthetic_pct=50, sub_scores={"sharpness": 100.0}, burst_id=0),
            PhotoAnalysis(photo_id="b", technical_pct=10, aesthetic_pct=50, sub_scores={"sharpness": 10.0}, burst_id=0)]
    assign_ranks(rows)
    best = next(r for r in rows if r.burst_rank == 0)
    assert best.photo_id == "a" and best.sub_scores["rank_reason"] == "technical"


# ── 흐름 ─────────────────────────────────────────────────────────────────────
def test_revisited_concept_lands_in_one_folder(tmp_path, caplog):
    """컨셉 0 → 1 → 0(다시 찍음). 시간 구간 셋을 VLM 이 둘로 묶으면 1층 둘, 2층은 1층 안에서."""
    w = world(tmp_path, order=(0, 1, 0), sets=2, per_set=5)
    llm = FakeLlm(merge=lambda u: {0: 0, 1: 1, 2: 0}[u])
    with caplog.at_level("INFO", logger="categorize.service.pipeline"):
        result = pipeline.run(w.store, "g", w.refs, w.settings, llm, concept_count=2)

    stage = [m for m in caplog.messages if m.startswith("[categorize] 갤러리 g ")]
    assert len(stage) == 2 and "구간 3 time" in stage[1]
    assert result["segmentMode"] == "time" and result["segments"] == 3
    assert result["concepts"]["concepts"] == ["컨셉0", "컨셉1"]
    assert llm.kinds() == ["concept", "detail", "detail"] and result["naming"]["llmCalls"] == 3

    back = w.store.read_analysis("g")
    by_id = {r.photo_id: r for r in back}
    assignments = {a.embed_group_id: a for a in w.store.read_assignments("g")}
    for r, c in zip(w.rows, w.concept_of):
        a = assignments[by_id[r.photo_id].embed_group_id]
        assert a.concept_name == f"컨셉{c}"                    # 사진이 제 컨셉 폴더에
    # 같은 세트는 두 번 찍어도 한 2층 그룹
    set_groups = {}
    for r, s in zip(w.rows, w.set_of):
        set_groups.setdefault(s, set()).add(by_id[r.photo_id].embed_group_id)
    assert all(len(g) == 1 for g in set_groups.values())
    assert result["embedGroups"]["embedGroups"] == 4


def test_without_times_the_gallery_is_split_by_image(tmp_path):
    w = world(tmp_path, order=(0, 1, 2), timed=False)
    settings = with_knobs(w.settings, visual_units=3)
    result = pipeline.run(w.store, "g", w.refs, settings, FakeLlm())
    assert result["segmentMode"] == "visual" and len(result["concepts"]["concepts"]) == 3


def test_without_llm_segments_become_concepts_and_groups_are_saved(tmp_path):
    w = world(tmp_path, order=(0, 1))
    result = pipeline.run(w.store, "g", w.refs, w.settings, None)
    assert result["naming"] == "skipped (no --llm)"
    assert all(r.embed_group_id >= 0 and "sharpness_pct" in r.sub_scores for r in w.store.read_analysis("g"))


def test_falls_back_to_clip_when_no_embedder_vectors(tmp_path):
    w = world(tmp_path)
    ids = [r.photo_id for r in w.rows]
    w.store.write_analysis("g", w.rows, ([], np.zeros((0, 0))), (ids, w.C))
    result = pipeline.run(w.store, "g", w.refs, w.settings, None)
    assert result["embeddingsSource"] == "clip"


def test_reads_gallery_once(tmp_path):
    w = world(tmp_path, sets=2, per_set=5)
    counting = CountingStore(w.store)
    pipeline.run(counting, "g", w.refs, w.settings, FakeLlm())
    assert counting.reads == 1


def test_photos_without_scores_are_skipped(tmp_path):
    w = world(tmp_path)
    refs = w.refs + [PhotoRef(photo_id="없는.jpg", path=None)]
    result = pipeline.run(w.store, "g", refs, w.settings, FakeLlm())
    assert result["photos"] == len(w.rows)


# ── 화질 점수가 덜 찼을 때(wes #274 2물결) ──────────────────────────────────────
def _set_quality(store, gallery, scored):
    rows = store.read_analysis(gallery)
    for r in rows:
        r.quality_scored = scored(r)
    store._write_rows(gallery, rows)


def test_unfinished_quality_scores_leave_ranks_empty_and_folders_unchanged(tmp_path):
    """한 장이라도 화질 점수가 덜 찼으면 백분위·순위는 비우고, 폴더용(연사 묶음·그룹)은 다 찼을 때와 같다."""
    w = world(tmp_path, order=(0, 1), sets=2, per_set=5)
    pipeline.run(w.store, "g", w.refs, w.settings, None)
    groups_when_scored = {r.photo_id: (r.embed_group_id, r.burst_id) for r in w.store.read_analysis("g")}

    first = w.rows[0].photo_id
    _set_quality(w.store, "g", lambda r: r.photo_id != first)
    result = pipeline.run(w.store, "g", w.refs, w.settings, None)

    rows = w.store.read_analysis("g")
    assert result["ranked"] is False
    assert {r.photo_id: (r.embed_group_id, r.burst_id) for r in rows} == groups_when_scored
    assert all(r.technical_pct is None and r.aesthetic_pct is None and r.burst_rank is None for r in rows)
    assert not any("sharpness_pct" in r.sub_scores or "rank_reason" in r.sub_scores for r in rows)


def test_rank_mode_fills_ranks_once_quality_scores_are_complete(tmp_path):
    w = world(tmp_path, order=(0, 1), sets=2, per_set=5)
    _set_quality(w.store, "g", lambda r: False)
    pipeline.run(w.store, "g", w.refs, w.settings, None)
    groups = {r.photo_id: (r.embed_group_id, r.burst_id) for r in w.store.read_analysis("g")}

    _set_quality(w.store, "g", lambda r: True)
    result = pipeline.rank(w.store, "g")

    rows = w.store.read_analysis("g")
    assert result["ranked"] == len(w.rows) and result["pending"] == 0
    assert all(r.technical_pct is not None and r.burst_rank is not None for r in rows)
    assert {r.photo_id: (r.embed_group_id, r.burst_id) for r in rows} == groups
    assert sum(1 for r in rows if r.burst_rank == 0) == len({r.burst_id for r in rows})


def test_rank_mode_waits_while_quality_scores_are_pending(tmp_path):
    w = world(tmp_path, order=(0,), per_set=4)
    _set_quality(w.store, "g", lambda r: False)
    pipeline.run(w.store, "g", w.refs, w.settings, None)

    result = pipeline.rank(w.store, "g")

    assert result == {"gallery": "g", "mode": "rank", "ranked": 0, "pending": len(w.rows)}
    assert all(r.technical_pct is None for r in w.store.read_analysis("g"))
