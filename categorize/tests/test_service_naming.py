"""naming — 가짜 LLM 으로 ① 구간 → 컨셉 묶기 ② 1층별 2층 이름 · 최근접 · 검증."""

from __future__ import annotations

import pytest

from categorize.service import grouping, naming, pipeline
from categorize.service.naming import DETAIL_ALL, name_concepts, name_details
from tests.helpers import CountingStore, FakeLlm, with_knobs, world


def _grouped(w, settings=None):
    grouped, _ = pipeline.group(w.store, "g", w.refs, settings or w.settings)
    return grouped


def _details(w, grouped, unit_concept, concepts, llm, settings=None, store=None):
    """1층이 정해진 뒤의 흐름(pipeline.run 과 같다) — 2층 그룹을 매기고 이름."""
    settings = settings or w.settings
    concept_of_row = [0] * len(grouped.rows)
    for u, members in enumerate(grouped.segmentation.units):
        for i in members:
            concept_of_row[i] = unit_concept[u]
    ids = grouping.detail_groups(grouped.X, concept_of_row, settings.knobs.group_distance)
    for r, g in zip(grouped.rows, ids):
        r.embed_group_id = int(g)
    return name_details(store or w.store, "g", grouped, concepts, concept_of_row, settings, llm), concept_of_row


# ── ① 1층 ────────────────────────────────────────────────────────────────────
def test_time_mode_forces_the_requested_concept_count(tmp_path):
    w = world(tmp_path, order=(0, 1, 0))
    llm = FakeLlm(merge=lambda u: {0: 0, 1: 1, 2: 0}[u])
    unit_concept, concepts, summary = name_concepts(w.store, "g", _grouped(w), w.settings, llm, concept_count=2)
    assert "정확히 2개" in llm.calls[0][1]
    assert unit_concept == [0, 1, 0] and [c.name for c in concepts] == ["컨셉0", "컨셉1"]
    assert summary["segmentMode"] == "time" and summary["conceptCountForced"]


def test_visual_mode_does_not_force_the_count(tmp_path):
    """시간이 없으면 K를 강제하지 않는다 — 강제하면 흰 스튜디오 둘을 합쳤다(갤러리 18)."""
    w = world(tmp_path, timed=False)
    settings = with_knobs(w.settings, visual_units=3)
    llm = FakeLlm()
    _, concepts, summary = name_concepts(w.store, "g", _grouped(w, settings), settings, llm, concept_count=2)
    assert "정확히" not in llm.calls[0][1] and "2개라고 기억" in llm.calls[0][1]
    assert summary["segmentMode"] == "visual" and not summary["conceptCountForced"]
    assert len(concepts) == 3


def test_concept_names_are_made_unique(tmp_path):
    """wes 는 1층을 이름으로 묶는다 — 같은 이름이 둘이면 폴더가 합쳐진다."""
    w = world(tmp_path)
    _, concepts, _ = name_concepts(w.store, "g", _grouped(w), w.settings, FakeLlm(names=lambda c: "화이트 스튜디오"))
    assert [c.name for c in concepts] == ["화이트 스튜디오", "화이트 스튜디오 2", "화이트 스튜디오 3"]


def test_missing_segment_takes_the_neighbour_concept_in_time_mode(tmp_path):
    w = world(tmp_path, order=(0, 1, 2))
    unit_concept, concepts, summary = name_concepts(w.store, "g", _grouped(w), w.settings, FakeLlm(drop_segments={1}))
    assert unit_concept == [0, 0, 1] and len(concepts) == 2
    assert summary["missingSegments"] == 1


def test_missing_segment_takes_the_most_similar_concept_in_visual_mode(tmp_path):
    w = world(tmp_path, order=(0, 1, 2), timed=False)
    settings = with_knobs(w.settings, visual_units=3)
    grouped = _grouped(w, settings)
    unit_concept, _, _ = name_concepts(w.store, "g", grouped, settings, FakeLlm(drop_segments={2}))
    assert len(set(unit_concept)) == 2 and unit_concept[2] in unit_concept[:2]


def test_concept_call_sends_one_tile_per_segment_in_one_batch(tmp_path):
    w = world(tmp_path)
    counting = CountingStore(w.store)
    name_concepts(counting, "g", _grouped(w), w.settings, FakeLlm())
    assert len(counting.preview_calls) == 1


# ── ② 2층 ────────────────────────────────────────────────────────────────────
def test_single_group_concept_is_named_all_without_a_call(tmp_path):
    w = world(tmp_path, sets=1)
    llm = FakeLlm()
    grouped = _grouped(w)
    unit_concept, concepts, _ = name_concepts(w.store, "g", grouped, w.settings, llm)
    result, _ = _details(w, grouped, unit_concept, concepts, llm)
    back = w.store.read_assignments("g")
    assert {a.detail_name for a in back} == {DETAIL_ALL}
    assert llm.kinds() == ["concept"] and result["llmCalls"] == 0


def test_sets_inside_a_concept_get_their_own_names(tmp_path):
    w = world(tmp_path, order=(0, 1), sets=3, per_set=6)
    llm = FakeLlm()
    grouped = _grouped(w)
    unit_concept, concepts, _ = name_concepts(w.store, "g", grouped, w.settings, llm)
    result, concept_of_row = _details(w, grouped, unit_concept, concepts, llm)
    back = w.store.read_assignments("g")
    assert llm.kinds().count("detail") == 2 and result["llmCalls"] == 2
    assert result["embedGroups"] == 6 and result["vlmGroups"] == 6
    assert all(a.detail_name.startswith("세트") for a in back)
    # 2층이 1층을 넘지 않는다
    by_group = {}
    for r, c in zip(grouped.rows, concept_of_row):
        by_group.setdefault(r.embed_group_id, set()).add(c)
    assert all(len(cs) == 1 for cs in by_group.values())


def test_coverage_target_sends_the_rest_to_the_nearest_group(tmp_path):
    w = world(tmp_path, order=(0,), sets=4, per_set=6)
    settings = with_knobs(w.settings, naming_coverage=0.4)
    llm = FakeLlm()
    grouped = _grouped(w, settings)
    unit_concept, concepts, _ = name_concepts(w.store, "g", grouped, settings, llm)
    result, _ = _details(w, grouped, unit_concept, concepts, llm, settings)
    assert result["vlmGroups"] == 2 and result["nearestGroups"] == 2
    assert all(a.detail_name.startswith("세트") for a in w.store.read_assignments("g"))


def test_detail_calls_run_concurrently_and_failures_propagate(tmp_path):
    import threading
    import time as _time

    class SlowLlm(FakeLlm):
        def __init__(self, fail=False):
            super().__init__()
            self.threads, self.lock, self.fail = set(), threading.Lock(), fail

        def complete_json(self, system, user, schema, max_tokens):
            out = super().complete_json(system, user, schema, max_tokens)
            if self.calls[-1][0] == "detail":
                with self.lock:
                    self.threads.add(threading.get_ident())
                _time.sleep(0.05)
                if self.fail:
                    raise RuntimeError("bedrock 429")
            return out

    w = world(tmp_path, order=(0, 1, 2), sets=2, per_set=5)
    llm = SlowLlm()
    grouped = _grouped(w)
    unit_concept, concepts, _ = name_concepts(w.store, "g", grouped, w.settings, llm)
    started = _time.monotonic()
    _details(w, grouped, unit_concept, concepts, llm)
    assert len(llm.threads) > 1 and _time.monotonic() - started < 0.14

    llm = SlowLlm(fail=True)
    grouped = _grouped(w)
    unit_concept, concepts, _ = name_concepts(w.store, "g", grouped, w.settings, llm)
    with pytest.raises(RuntimeError, match="bedrock 429"):
        _details(w, grouped, unit_concept, concepts, llm)


def test_detail_representatives_are_fetched_in_one_batch(tmp_path):
    w = world(tmp_path, order=(0, 1), sets=3, per_set=5)
    llm = FakeLlm()
    grouped = _grouped(w)
    unit_concept, concepts, _ = name_concepts(w.store, "g", grouped, w.settings, llm)
    counting = CountingStore(w.store)
    _details(w, grouped, unit_concept, concepts, llm, store=counting)
    assert len(counting.preview_calls) == 1
    assert len(set(counting.preview_calls[0])) == len(counting.preview_calls[0])


def test_naming_module_has_no_fixed_concept_list():
    assert not hasattr(naming, "CONCEPTS") and not hasattr(naming, "CLIP_CONCEPT_KEY")
