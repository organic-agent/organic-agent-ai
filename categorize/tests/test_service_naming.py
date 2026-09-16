"""naming — 가짜 LLM 으로 ② 이름 ③ 최근접 배정 ④ 검증(clip_parent 다수결 · 배경 밝기)과 고립 그룹 보강(#118)."""

from __future__ import annotations

import pytest

from categorize.config.settings import Settings
from categorize.repository.local import LocalStore
from categorize.service import naming
from categorize.service.naming import majority
from tests.helpers import CountingStore, FakeLlm, with_knobs, world


def test_naming_all_groups_named_by_vlm(tmp_path):
    store, rows, *_, settings = world(tmp_path)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    back = store.read_assignments("g")
    n_groups = len({r.embed_group_id for r in rows})
    assert len(back) == n_groups == result["groups"]
    assert all(a.assigned_by == "vlm" for a in back)
    assert all(not a.needs_review for a in back)
    assert result["vlmGroups"] == n_groups and result["nearestGroups"] == 0


def test_naming_nearest_inherits_and_flags_far_groups(tmp_path):
    store, rows, *_, settings = world(tmp_path, n_groups=4, per_group=8, clip_parent=None)
    settings = with_knobs(settings, naming_max_groups=2, nearest_tau=0.05)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    back = {a.embed_group_id: a for a in store.read_assignments("g")}
    assert result["vlmGroups"] == 2
    nearest = [a for a in back.values() if a.assigned_by == "nearest"]
    assert nearest
    for a in nearest:
        if a.parent_name == "기타":
            assert a.needs_review
        else:
            assert a.concept_name.startswith("세트")


def test_naming_low_confidence_and_clip_mismatch_need_review(tmp_path):
    store, rows, *_, settings = world(tmp_path, clip_parent="야외 자연")
    gids = sorted({r.embed_group_id for r in rows})
    naming.run(store, "g", settings, FakeLlm(low_conf_gid=gids[0]), job_id=None)
    back = store.read_assignments("g")
    assert all(a.needs_review for a in back)
    assert all(a.clip_parent == "야외 자연" for a in back)


def test_naming_merge_call_unifies_names_across_chunks(tmp_path):
    store, rows, *_, settings = world(tmp_path, n_groups=5, per_group=6)
    settings = with_knobs(settings, naming_chunk=2, naming_spread_extra=9.0)
    llm = FakeLlm()
    result = naming.run(store, "g", settings, llm, job_id=None)
    kinds = [kind for kind, _ in llm.calls]
    assert kinds.count("vision") == 3 and kinds.count("merge") == 1
    assert result["llmCalls"] == 4
    assert all(a.concept_name.endswith("(통일)") for a in store.read_assignments("g"))


def test_naming_chunk_calls_run_concurrently(tmp_path):
    """청크 vision 호출은 서로 독립이라 동시에 나간다(#115) — 세 청크가 0.05s 씩이면 직렬 0.15s 가 아니라 ≈0.05s."""
    import threading
    import time as _time

    class SlowLlm(FakeLlm):
        def __init__(self):
            super().__init__()
            self.threads: set[int] = set()
            self.lock = threading.Lock()

        def complete_json(self, system, user, schema, max_tokens):
            with self.lock:
                self.threads.add(threading.get_ident())
            if not isinstance(user, str):
                _time.sleep(0.05)
            return super().complete_json(system, user, schema, max_tokens)

    store, rows, *_, settings = world(tmp_path, n_groups=5, per_group=6)
    settings = with_knobs(settings, naming_chunk=2, naming_spread_extra=9.0)
    llm = SlowLlm()
    started = _time.monotonic()
    result = naming.run(store, "g", settings, llm, job_id=None)
    elapsed = _time.monotonic() - started

    assert result["llmCalls"] == 4 and [k for k, _ in llm.calls].count("vision") == 3
    assert len(llm.threads) > 1 and elapsed < 0.14
    assert all(a.concept_name.endswith("(통일)") for a in store.read_assignments("g"))   # 결과 계약 불변

    # naming_parallel=1 이면 직렬로 돌아간다 — 스로틀 때의 손잡이
    settings_serial = with_knobs(settings, naming_chunk=2, naming_spread_extra=9.0, naming_parallel=1)
    started = _time.monotonic()
    naming.run(store, "g", settings_serial, SlowLlm(), job_id=None)
    assert _time.monotonic() - started >= 0.15


def test_naming_chunk_failure_still_fails_the_run(tmp_path):
    """청크 하나가 실패하면 naming 전체가 실패한다 — 병렬화가 예외를 삼키지 않는다(통합 호출만 실패를 삼킨다)."""
    class FlakyLlm(FakeLlm):
        def complete_json(self, system, user, schema, max_tokens):
            out = super().complete_json(system, user, schema, max_tokens)
            if [k for k, _ in self.calls].count("vision") == 2:
                raise RuntimeError("bedrock 429")
            return out

    store, rows, *_, settings = world(tmp_path, n_groups=5, per_group=6)
    settings = with_knobs(settings, naming_chunk=2, naming_spread_extra=9.0)
    with pytest.raises(RuntimeError, match="bedrock 429"):
        naming.run(store, "g", settings, FlakyLlm(), job_id=None)


def test_naming_coverage_target_limits_vlm_groups(tmp_path):
    """이웃이 가까울 때(group_distance 를 넘게 잡아 고립 보강을 끈 상태) 커버리지가 상한이다."""
    store, rows, *_, settings = world(tmp_path, n_groups=4, per_group=8, clip_parent=None)
    settings = with_knobs(settings, naming_coverage=0.5, nearest_tau=1.0, group_distance=2.0)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["vlmGroups"] < result["groups"]
    assert result["nearestGroups"] == result["groups"] - result["vlmGroups"]
    assert 0.5 <= result["coverage"] < 1.0


# ── 고립 그룹 보강 · 배경 검증 (#118) ────────────────────────────────────────
def test_isolated_group_is_named_even_outside_the_coverage_target(tmp_path):
    """합성 세계의 그룹 중심은 서로 직교에 가깝다 — 전부 고립이라 커버리지를 넘겨 모두 이름을 받는다."""
    store, rows, *_, settings = world(tmp_path, n_groups=4, per_group=8, clip_parent=None)
    settings = with_knobs(settings, naming_coverage=0.5, nearest_tau=1.0, group_distance=0.2)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["vlmGroups"] == result["groups"]
    assert result["isolatedGroups"] == result["groups"] - 2      # 커버리지로 2개, 나머지는 고립 보강
    assert result["nearestGroups"] == 0


def test_isolation_is_keyed_to_group_distance_not_nearest_tau(tmp_path):
    """τ 에 걸면 오배정이 그대로 남는다(운영 갤러리 25: 거리 0.239 가 τ=0.25 를 통과) — 기준은 group_distance 다.

    τ=0(모든 그룹이 τ 밖)인데 group_distance=2(고립 없음)로 두면, 기준을 잘못 잡은 구현만 전부 이름을 짓는다."""
    store, rows, *_, settings = world(tmp_path, n_groups=4, per_group=8, clip_parent=None)
    settings = with_knobs(settings, naming_coverage=0.5, nearest_tau=0.0, group_distance=2.0)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["isolatedGroups"] == 0
    assert result["vlmGroups"] < result["groups"]


def test_borrowed_name_with_a_different_background_needs_review(tmp_path):
    """이름을 빌려온 그룹과 배경 밝기가 다르면 확인 대상 — 부모(실내 스튜디오)로는 검은·흰 스튜디오가 안 갈린다."""
    store, rows, *_, settings = world(tmp_path, n_groups=3, per_group=6, clip_parent=None,
                                      bg=lambda g, j: 5.0 if g == 2 else 200.0)
    settings = with_knobs(settings, naming_coverage=0.4, nearest_tau=1.0, group_distance=2.0)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["nearestGroups"] >= 1
    assert result["bgMismatchGroups"] == 1
    borrowed = [a for a in store.read_assignments("g") if a.assigned_by == "nearest"]
    assert any(a.needs_review for a in borrowed)


def test_background_outlier_inside_a_named_group_needs_review(tmp_path):
    """대표 사진을 보고 지은 이름인데 멤버 배경이 다르면, 그 그룹 자체가 확인 대상이다."""
    store, rows, *_, settings = world(tmp_path, n_groups=2, per_group=8, clip_parent=None,
                                      bg=lambda g, j: 200.0 if j % 2 == 0 else 5.0)
    settings = with_knobs(settings, group_distance=2.0)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["vlmGroups"] == result["groups"]
    assert result["bgMismatchGroups"] == result["groups"]
    assert all(a.needs_review for a in store.read_assignments("g"))


def test_missing_bg_luma_never_triggers_review(tmp_path):
    """재점수 전 갤러리는 bg_luma 가 없다 — 없는 신호가 리뷰를 켜면 안 된다."""
    store, rows, *_, settings = world(tmp_path, n_groups=3, per_group=6, clip_parent=None)
    settings = with_knobs(settings, naming_coverage=0.4, nearest_tau=1.0, group_distance=2.0)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["bgMismatchGroups"] == 0
    assert result["needsReview"] == 0


def test_naming_spread_adds_second_rep(tmp_path):
    store, rows, *_, settings = world(tmp_path)
    settings = with_knobs(settings, naming_spread_extra=0.0)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["extraReps"] == result["groups"]
    assert result["vlmGroups"] == result["groups"]


def test_naming_requires_llm(tmp_path):
    store, *_, settings = world(tmp_path)
    with pytest.raises(RuntimeError, match="Bedrock"):
        naming.run(store, "g", settings, None)


def test_naming_requires_full_analysis(tmp_path):
    store = LocalStore(tmp_path / "out", dataset_root=tmp_path)
    settings = Settings(out_root=tmp_path / "out", dataset_root=tmp_path)
    with pytest.raises(RuntimeError, match="FULL"):
        naming.run(store, "empty", settings, FakeLlm())


def test_majority_ignores_none_and_breaks_ties_by_first_seen():
    assert majority(["야외 자연", None, "야외 자연", "실내 스튜디오"]) == "야외 자연"
    assert majority([None, None]) is None
    assert majority(["a", "b"]) == "a"


def test_naming_fetches_all_representatives_in_one_batch(tmp_path):
    """대표 사진 경로는 preview_paths 한 번으로 — 사진마다 SELECT + GET 을 직렬로 하지 않는다."""
    store, rows, *_, settings = world(tmp_path)
    counting = CountingStore(store)

    result = naming.run(counting, "g", settings, FakeLlm(), job_id=None)

    assert len(counting.preview_calls) == 1
    assert len(counting.preview_calls[0]) >= result["vlmGroups"]
    assert len(set(counting.preview_calls[0])) == len(counting.preview_calls[0])
