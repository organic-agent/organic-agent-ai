"""v3 단위 테스트 — 합성 데이터로 임베딩 그룹·naming(가짜 LLM)·LocalStore 왕복을 돈다. 모델 없음."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from photoselect.v3.analyze import assign_ranks, concat_space, percentile
from photoselect.v3.config import PARENTS, Settings, V3Knobs, V3_MODEL_VERSION
from photoselect.v3 import cluster, compare, concept, draft, naming
from photoselect.v3.store import ConceptAssignment, FolderSetMissing, LocalStore, PhotoAnalysis


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _world(tmp_path, n_groups=3, per_group=10, seed=0):
    """그룹마다 중심 벡터 + 노이즈. E(임베더)·C(CLIP)는 같은 그룹 구조를 공유한다."""
    rng = np.random.default_rng(seed)
    dim = 32
    centers = [_unit(rng.normal(size=dim)) for _ in range(n_groups)]
    rows, E, C = [], [], []
    for g, c in enumerate(centers):
        for j in range(per_group):
            E.append(_unit(c + 0.08 * rng.normal(size=dim)))
            C.append(_unit(c + 0.08 * rng.normal(size=dim)))
            pid = f"g{g}-{j:02d}"
            rows.append(PhotoAnalysis(
                photo_id=pid, subjects="couple",
                sub_scores={"technical_score": rng.uniform(0.3, 0.8),
                            "aesthetic_score": rng.uniform(5, 6.5),
                            "sharpness": rng.uniform(50, 500)},
                model_version=V3_MODEL_VERSION))
    E, C = np.stack(E), np.stack(C)
    for r, t in zip(rows, percentile([r.sub_scores["technical_score"] for r in rows])):
        r.technical_pct = t
    X = concat_space(E, C)
    gids, _ = concept.concept_groups(X, 0.4, min_groups=2, max_share=0.6)
    for i, (r, g) in enumerate(zip(rows, gids)):
        r.embed_group_id = int(g)
        r.cluster_id = i          # 연사 없음 — 전부 단독 클러스터
    assign_ranks(rows)

    # 대표 사진용 실제 JPEG (naming 이 jpeg_bytes 를 부른다)
    img_root = tmp_path / "dataset"
    img_root.mkdir(exist_ok=True)
    for r in rows:
        Image.new("RGB", (32, 32), (200, 180, 160)).save(img_root / f"{r.photo_id}.jpg")
        r.photo_id = r.photo_id + ".jpg"    # LocalStore.preview_path 는 dataset_root/photo_id
    ids = [r.photo_id for r in rows]

    store = LocalStore(tmp_path / "out", dataset_root=img_root)
    store.write_analysis("g", rows, (ids, E), (ids, C))
    settings = Settings(out_root=tmp_path / "out", dataset_root=img_root)
    return store, rows, E, C, settings


# ── 순수 로직 ────────────────────────────────────────────────────────────────
def test_concat_space_is_mean_of_cosines():
    rng = np.random.default_rng(1)
    E = np.stack([_unit(rng.normal(size=8)) for _ in range(4)])
    C = np.stack([_unit(rng.normal(size=8)) for _ in range(4)])
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


# ── 연사 클러스터 — 카메라 파티션 (review-v3-design.md (1)) ──────────────────
def test_partition_order_splits_cameras_and_sorts_by_taken_at():
    cams = ["A", "B", "A", "B", None]
    times = [3, 1, 1, 2, None]
    parts = cluster.partition_order(cams, times)
    assert parts == [[2, 0], [1, 3], [4]]     # 파티션은 첫 등장 순, 안은 taken_at 순


def test_partition_order_keeps_input_order_when_taken_at_missing():
    # 부분 정렬 금지 — 하나라도 시각이 없으면 그 파티션은 입력 순서 그대로
    assert cluster.partition_order(["A", "A", "A"], [5, None, 1]) == [[0, 1, 2]]


def test_partitioned_bursts_never_cross_cameras():
    # 두 카메라가 같은 순간을 찍으면 임베딩이 거의 같다(코사인≈1). 파티션 없이는 교차
    # 병합되지만, 파티션을 나누면 카메라 경계를 절대 넘지 않아야 한다.
    rng = np.random.default_rng(2)
    base = _unit(rng.normal(size=16))
    emb = np.stack([_unit(base + 0.01 * rng.normal(size=16)) for _ in range(6)])
    cams = ["A", "B", "A", "B", "A", "B"]
    assert len(set(cluster.cluster_bursts(emb, 0.9, 8))) == 1   # 순서만으론 전부 한 덩어리
    parts = cluster.partition_order(cams, list(range(6)))
    cids = cluster.cluster_bursts_partitioned(emb, parts, 0.9, 8)
    assert (cids >= 0).all()
    a_ids = {cids[i] for i in range(6) if cams[i] == "A"}
    b_ids = {cids[i] for i in range(6) if cams[i] == "B"}
    assert a_ids.isdisjoint(b_ids)
    assert len(a_ids) == 1 and len(b_ids) == 1   # 카메라 안에서는 여전히 연사 하나


# ── 임베딩 그룹 — 대칭 적응 규칙 (review-v3-design.md (10)) ──────────────────
def _two_cluster_emb(n_per=10, noise=0.05, seed=3):
    rng = np.random.default_rng(seed)
    centers = [_unit(rng.normal(size=16)) for _ in range(2)]
    return np.stack([_unit(c + noise * rng.normal(size=16))
                     for c in centers for _ in range(n_per)])


def test_concept_groups_raises_threshold_on_fragmentation():
    emb = _two_cluster_emb()
    # 임계 0.01은 20장을 조각낸다 → 그룹 수 > n×frag_share(7) → 0.05씩 상승해 2그룹까지 병합
    labels, used = concept.concept_groups(emb, 0.01, min_groups=2)
    assert used > 0.01
    assert len(set(labels.tolist())) == 2


def test_concept_groups_raise_steps_back_before_overmerge():
    emb = _two_cluster_emb()
    # min_groups=4 면 2그룹으로의 병합은 과병합 — 상승은 그 직전에 멈추고 되돌아가야 한다
    labels, used = concept.concept_groups(emb, 0.01, min_groups=4)
    g = len(set(labels.tolist()))
    assert g >= 4                                   # 과병합 조건을 깨는 상승은 채택 안 됨
    assert np.bincount(labels).max() <= len(emb) * 0.5


# ── LocalStore 왕복 ──────────────────────────────────────────────────────────
def test_local_store_roundtrip(tmp_path):
    store, rows, E, C, _ = _world(tmp_path)
    got = store.read_analysis("g")
    assert [r.photo_id for r in got] == [r.photo_id for r in rows]
    assert all(r.embed_group_id >= 0 for r in got)
    ids, E2 = store.read_embeddings("g")
    cids, C2 = store.read_clip_embeddings("g")
    assert ids == cids == [r.photo_id for r in rows]
    np.testing.assert_allclose(E2, E, atol=1e-6)
    np.testing.assert_allclose(C2, C, atol=1e-6)

    a = [ConceptAssignment(embed_group_id=0, parent_name="야외 자연", concept_name="해변",
                           confidence=0.9, assigned_by="vlm")]
    store.write_assignments("g", None, a)
    back = store.read_assignments("g")
    assert back == a


# ── naming (가짜 LLM·검증기) ─────────────────────────────────────────────────
class FakeLlm:
    """청크 vision 호출은 그룹마다 이름을, 통합 호출은 concept 표기를 바꿔 돌려준다."""

    def __init__(self, parent="실내 스튜디오", low_conf_gid=None):
        self.calls: list[tuple[str, bool]] = []   # (종류, 이미지 포함 여부)
        self.parent = parent
        self.low_conf_gid = low_conf_gid

    def complete_json(self, system, user, schema, max_tokens):
        has_image = not isinstance(user, str) and any(k == "image" for k, _ in user)
        kind = "vision" if has_image else "merge"
        self.calls.append((kind, has_image))
        if kind == "vision":
            gids = [int(t.split("]")[0].split()[1]) for k, t in user
                    if k == "text" and t.startswith("[그룹")]
            return {"groups": [
                {"group_id": g, "parent": self.parent, "proposed_parent": None,
                 "concept": f"세트{g}",
                 "confidence": 0.3 if g == self.low_conf_gid else 0.95}
                for g in gids]}
        # merge: concept 표기 통일 시늉
        lines = user.splitlines()
        out = []
        for ln in lines:
            gid = int(ln.split(":")[0].split()[1])
            out.append({"group_id": gid, "parent": self.parent, "proposed_parent": None,
                        "concept": f"세트{gid}(통일)"})
        return {"groups": out}


class FakeVerifier:
    def __init__(self, answer):
        self.answer = answer

    def majority(self, clip_embs):
        return self.answer


def test_naming_all_groups_named_by_vlm(tmp_path):
    store, rows, *_ , settings = _world(tmp_path)
    llm = FakeLlm()
    result = naming.run(store, "g", settings, llm, job_id=None,
                        verifier=FakeVerifier("실내 스튜디오"))
    back = store.read_assignments("g")
    n_groups = len({r.embed_group_id for r in rows})
    assert len(back) == n_groups == result["groups"]
    assert all(a.assigned_by == "vlm" for a in back)
    assert all(not a.needs_review for a in back)          # conf 높고 CLIP 일치
    assert result["vlmGroups"] == n_groups and result["nearestGroups"] == 0


def test_naming_nearest_inherits_and_flags_far_groups(tmp_path):
    store, rows, *_, settings = _world(tmp_path, n_groups=4, per_group=8)
    k = V3Knobs(naming_max_groups=2, nearest_tau=0.05)    # 상위 2그룹만 VLM, τ 빡빡하게
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        v3=k, llm=settings.llm)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None,
                        verifier=FakeVerifier(None))
    back = {a.embed_group_id: a for a in store.read_assignments("g")}
    assert result["vlmGroups"] == 2
    nearest = [a for a in back.values() if a.assigned_by == "nearest"]
    assert nearest, "K 밖 그룹이 nearest 로 배정돼야 한다"
    for a in nearest:                                      # τ=0.05 면 다른 세트는 전부 '기타'
        if a.parent_name == "기타":
            assert a.needs_review
        else:
            assert a.concept_name.startswith("세트")       # 상속받은 이름


def test_naming_low_confidence_and_clip_mismatch_need_review(tmp_path):
    store, rows, *_, settings = _world(tmp_path)
    gids = sorted({r.embed_group_id for r in rows})
    llm = FakeLlm(low_conf_gid=gids[0])
    naming.run(store, "g", settings, llm, job_id=None,
               verifier=FakeVerifier("야외 자연"))          # VLM(실내 스튜디오)과 불일치
    back = store.read_assignments("g")
    assert all(a.needs_review for a in back)               # 불일치라 전부 review
    assert all(a.clip_parent == "야외 자연" for a in back)


def test_naming_merge_call_unifies_names_across_chunks(tmp_path):
    store, rows, *_, settings = _world(tmp_path, n_groups=5, per_group=6)
    # spread 대표 추가를 꺼서(임계 9) 그룹당 1장 → 그룹 5개 = vision 3청크 + merge 1
    k = V3Knobs(naming_chunk=2, naming_spread_extra=9.0)
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        v3=k, llm=settings.llm)
    llm = FakeLlm()
    result = naming.run(store, "g", settings, llm, job_id=None,
                        verifier=FakeVerifier("실내 스튜디오"))
    kinds = [kind for kind, _ in llm.calls]
    assert kinds.count("vision") == 3 and kinds.count("merge") == 1
    assert result["llmCalls"] == 4
    assert all(a.concept_name.endswith("(통일)") for a in store.read_assignments("g"))


def test_naming_coverage_target_limits_vlm_groups(tmp_path):
    store, rows, *_, settings = _world(tmp_path, n_groups=4, per_group=8)
    # 커버리지 50% → 크기순 일부 그룹만 VLM, 나머지는 nearest 상속 (review-v3-design.md (3))
    k = V3Knobs(naming_coverage=0.5, nearest_tau=1.0)
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        v3=k, llm=settings.llm)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None,
                        verifier=FakeVerifier(None))
    assert result["vlmGroups"] < result["groups"]
    assert result["nearestGroups"] == result["groups"] - result["vlmGroups"]
    assert 0.5 <= result["coverage"] < 1.0          # 목표 도달 지점에서 멈춘 실측치


def test_naming_spread_adds_second_rep(tmp_path):
    store, rows, *_, settings = _world(tmp_path)
    # 임계 0 → 모든 그룹이 이질 판정 → 중심 최근접 + 최원점 2장 (review-v3-design.md (2))
    k = V3Knobs(naming_spread_extra=0.0)
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        v3=k, llm=settings.llm)
    llm = FakeLlm()
    result = naming.run(store, "g", settings, llm, job_id=None,
                        verifier=FakeVerifier("실내 스튜디오"))
    assert result["extraReps"] == result["groups"]
    assert result["vlmGroups"] == result["groups"]   # 대표가 늘어도 전 그룹이 이름을 받는다


def test_naming_requires_llm(tmp_path):
    store, *_, settings = _world(tmp_path)
    with pytest.raises(RuntimeError, match="Bedrock"):
        naming.run(store, "g", settings, None)


def test_naming_requires_full_analysis(tmp_path):
    store = LocalStore(tmp_path / "out", dataset_root=tmp_path)
    settings = Settings(out_root=tmp_path / "out", dataset_root=tmp_path)
    with pytest.raises(RuntimeError, match="FULL"):
        naming.run(store, "empty", settings, FakeLlm(), verifier=FakeVerifier(None))


def test_parents_fixed_lists():
    assert PARENTS["OTHER"] == PARENTS["REHEARSAL"]
    for lst in PARENTS.values():
        assert "기타" in lst


# ── 폴더별 추천 (draft, plan-v3-folder-compare.md §2) ────────────────────────
def _named_world(tmp_path, **kw):
    """_world + 그룹마다 (부모, 컨셉) 배정 → 로컬 폴더 세트가 생긴다. 그룹 g → '세트g' 폴더."""
    store, rows, E, C, settings = _world(tmp_path, **kw)
    gids = sorted({r.embed_group_id for r in rows})
    store.write_assignments("g", None, [
        ConceptAssignment(embed_group_id=g, parent_name="실내 스튜디오", concept_name=f"세트{g}",
                          confidence=0.9, assigned_by="vlm")
        for g in gids])
    return store, rows, E, C, settings


def _evidence(store, selected=(), rejected=()):
    (store._dir("g") / "evidence.json").write_text(
        json.dumps({"selected": list(selected), "rejected": list(rejected)}), encoding="utf-8")


class FakeReasonLlm:
    """reasons.generate 용 — 요청된 photo_id 마다 고정 문장을 돌려준다."""

    def __init__(self):
        self.calls = 0

    def complete_json(self, system, user, schema, max_tokens):
        self.calls += 1
        pids = [t.split("### photo_id: ")[1].splitlines()[0]
                for k, t in user if k == "text" and t.startswith("### photo_id:")]
        return {"reasons": [{"photo_id": p, "reason": f"LLM이 본 {p}"} for p in pids]}


def test_draft_requires_folder_set(tmp_path):
    store, *_ , settings = _world(tmp_path)          # 배정 없음 — 세트 없음
    with pytest.raises(FolderSetMissing):
        draft.run(store, "g", settings, target=6)


def test_draft_fills_each_folder_by_quota(tmp_path):
    store, rows, *_, settings = _named_world(tmp_path)      # 3그룹 × 10장
    result = draft.run(store, "g", settings, target=6)
    assert result["folders"] == 3 and result["k"] == 6      # 폴더당 6·10/30 = 2장
    assert set(result["perFolder"].values()) == {2}
    back = store.read_recommendations("g")
    assert all(r.folder_id is not None for r in back)
    assert all(r.reason for r in back)                      # llm 없음 → 2단계에서 템플릿으로 채움
    assert sorted(r.rank for r in back if r.folder_id == back[0].folder_id) == [1, 2]
    assert any("폴더:" in f for r in back for f in r.score_breakdown["facts"])


def test_draft_cap_half_of_folder(tmp_path):
    store, rows, *_, settings = _named_world(tmp_path, n_groups=2, per_group=4)   # 폴더 4장씩
    result = draft.run(store, "g", settings, target=30)
    assert all(v <= 2 for v in result["perFolder"].values())    # n_f ≤ ceil(4·0.5)


def test_draft_unfiled_virtual_folder(tmp_path):
    store, rows, *_, settings = _named_world(tmp_path)
    gids = sorted({r.embed_group_id for r in rows})
    store.write_assignments("g", None, [                        # 마지막 그룹 배정 제거
        ConceptAssignment(embed_group_id=g, parent_name="실내 스튜디오", concept_name=f"세트{g}",
                          confidence=0.9, assigned_by="vlm") for g in gids[:-1]])
    result = draft.run(store, "g", settings, target=6)
    assert result["unfiled"] == 10                              # 빠진 그룹 10장이 미분류로
    assert "미분류›미분류" in result["perFolder"]
    unfiled = [r for r in store.read_recommendations("g") if r.folder_id is None]
    assert unfiled and all(r.score_breakdown["folder"] == "미분류›미분류" for r in unfiled)


def test_draft_refine_keeps_one_per_folder_when_target_met(tmp_path):
    store, rows, *_, settings = _named_world(tmp_path)
    selected = [r.photo_id for r in rows[:12]]                  # 목표 10 < 담은 12
    _evidence(store, selected=selected)
    result = draft.run(store, "g", settings, target=10)
    assert result["done"] is False and result["remaining"] < 0
    # 후보가 남은 폴더는 대표 1장, 전부 담긴 폴더(첫 그룹 10장)만 0장
    assert sorted(result["perFolder"].values()) == [0, 1, 1]
    back = store.read_recommendations("g")
    assert not (set(selected) & {r.photo_id for r in back})     # 담은 사진은 다시 안 나온다


def test_draft_rejected_excluded_but_shown_reexposed(tmp_path):
    store, rows, *_, settings = _named_world(tmp_path)
    r1 = draft.run(store, "g", settings, target=6)
    first = {r.photo_id for r in store.read_recommendations("g") if r.round == 1}
    rejected = sorted(first)[:2]
    _evidence(store, rejected=rejected)
    r2 = draft.run(store, "g", settings, target=6)
    second = {r.photo_id for r in store.read_recommendations("g") if r.round == 2}
    assert not (set(rejected) & second)                         # 거절은 제외
    assert first - set(rejected) <= second | first              # 이전 노출은 다시 나올 수 있다(§2.4)
    assert r2["round"] == 2


def test_draft_two_stage_reasons_with_llm(tmp_path):
    store, rows, *_, settings = _named_world(tmp_path)

    stages = []
    class SpyStore(type(store)):
        def write_recommendations(self, gallery, rows_):
            stages.append(("insert", [r.reason for r in rows_]))
            super().write_recommendations(gallery, rows_)
        def update_reasons(self, gallery, round_no, reasons_):
            stages.append(("update", list(reasons_.values())))
            super().update_reasons(gallery, round_no, reasons_)
    spy = SpyStore(store.root, dataset_root=store.dataset_root)

    llm = FakeReasonLlm()
    draft.run(spy, "g", settings, target=6, llm=llm)
    assert stages[0][0] == "insert" and all(t is None for t in stages[0][1])   # 1단계: reason NULL
    assert stages[1][0] == "update" and all(t.startswith("LLM이 본") for t in stages[1][1])
    assert llm.calls >= 1
    assert all(r.reason.startswith("LLM이 본") for r in spy.read_recommendations("g"))


# ── 비교샷 (compare, plan-v3-folder-compare.md §3) ───────────────────────────
def _pa(pid, tech=50.0, aes=50.0, sharp=100.0, cluster=-1, cluster_rank=0, subjects="couple"):
    return PhotoAnalysis(photo_id=pid, subjects=subjects, technical_pct=tech, aesthetic_pct=aes,
                         sub_scores={"sharpness": sharp}, cluster_id=cluster,
                         cluster_rank=cluster_rank, model_version=V3_MODEL_VERSION)


def test_compare_template_priority_sharpness_then_pcts():
    a, b = _pa("a", sharp=200.0), _pa("b", sharp=100.0)
    assert compare.template_verdict(a, b)[0] == "a"             # 초점 1.25배 우선
    a, b = _pa("a", tech=40.0), _pa("b", tech=60.0)
    assert compare.template_verdict(a, b)[0] == "b"             # 다음은 기술 백분위
    a, b = _pa("a", aes=70.0), _pa("b", aes=60.0)
    assert compare.template_verdict(a, b)[0] == "a"             # 다음은 미학
    a, b = _pa("a"), _pa("b")
    assert compare.template_verdict(a, b)[0] == "a"             # 차이 없으면 a


def test_compare_facts_burst_folder_selected():
    a = _pa("a", tech=70.0, cluster=3, cluster_rank=0)
    b = _pa("b", tech=50.0, cluster=3, cluster_rank=1, subjects="bride")
    facts, d = compare.collect_facts(a, b, "실내 › 세트0", "야외 › 해변", {"b"})
    text = "\n".join(facts)
    assert "연사" in text and d["same_burst"]["best"] == "a"
    assert "기술" in text and "20포인트" in text
    assert "실내 › 세트0" in text and "야외 › 해변" in text
    assert "유형" in text and "이미 담은" in text


class FakeCompareLlm:
    def __init__(self, out):
        self.out, self.calls = out, 0

    def complete_json(self, system, user, schema, max_tokens):
        self.calls += 1
        if isinstance(self.out, Exception):
            raise self.out
        return self.out


def _compare_world(tmp_path):
    store, rows, *_ , settings = _named_world(tmp_path)
    a, b = rows[0].photo_id, rows[1].photo_id
    return store, settings, a, b


def test_compare_llm_verdict_saved_and_cached_order_free(tmp_path):
    store, settings, a, b = _compare_world(tmp_path)
    llm = FakeCompareLlm({"chosen": "b", "confidence": "clear", "reason": "시선이 살아 있어요"})
    r1 = compare.run(store, "g", settings, a, b, llm=llm)
    assert r1["chosenPhotoId"] == b and r1["source"] == "llm" and not r1["cached"]
    r2 = compare.run(store, "g", settings, b, a, llm=llm)     # 순서를 뒤집어도 캐시
    assert r2["cached"] and r2["chosenPhotoId"] == b and llm.calls == 1


def test_compare_falls_back_to_template_on_bad_llm(tmp_path):
    store, settings, a, b = _compare_world(tmp_path)
    for bad in (FakeCompareLlm({"chosen": "c", "confidence": "clear", "reason": "x"}),
                FakeCompareLlm(RuntimeError("timeout"))):
        (store._dir("g") / "verdicts.jsonl").unlink(missing_ok=True)
        r = compare.run(store, "g", settings, a, b, llm=bad)
        assert r["source"] == "template" and r["chosenPhotoId"] in (a, b) and r["reason"]


def test_compare_without_llm_uses_template_and_requires_analysis(tmp_path):
    store, settings, a, b = _compare_world(tmp_path)
    r = compare.run(store, "g", settings, a, b, llm=None)
    assert r["source"] == "template" and r["confidence"] == "slight"
    with pytest.raises(SystemExit, match="분석되지 않은"):
        compare.run(store, "g", settings, a, "ghost.jpg", llm=None)


def test_quality_floor_gates_reason_material():
    k = V3Knobs()
    row = PhotoAnalysis(photo_id="x", technical_pct=95, aesthetic_pct=95,
                        sub_scores={"technical_score": 0.2, "aesthetic_score": 6.0})
    assert draft._quality_material(row, k) is None              # 원점수 하한 미만 → 품질 표현 제외
    row.sub_scores["technical_score"] = 0.7
    assert draft._quality_material(row, k) is not None
