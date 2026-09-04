"""photoselect 단위 테스트 — 합성 데이터로 SCORE 재개·CATEGORIZE(임베딩 그룹)·naming(가짜 LLM)·
LocalStore 왕복을 돈다. 모델 없음(torch 없이 돈다). 추천·비교샷 테스트는 wes로 갔다(#25)."""

from __future__ import annotations

import json
import subprocess
import sys
import types

import numpy as np
import pytest
from PIL import Image

from photoselect.categorize import assign_ranks, concat_space, percentile
from photoselect.config import PARENTS, Settings, Knobs, MODEL_VERSION
from photoselect import categorize, cluster, concept, naming, score
from photoselect.gallery import PhotoRef
from photoselect.store import ConceptAssignment, LocalStore, PhotoAnalysis
from photoselect.subjects import majority


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _world(tmp_path, n_groups=3, per_group=10, seed=0, clip_parent="실내 스튜디오"):
    """그룹마다 중심 벡터 + 노이즈. E(임베더)·C(CLIP)는 같은 그룹 구조를 공유한다.
    clip_parent 는 SCORE 가 사진마다 저장하는 부모 검증 라벨(sub_scores.clip_parent)이다."""
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
                            "sharpness": rng.uniform(50, 500),
                            "clip_parent": clip_parent},
                model_version=MODEL_VERSION))
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


def test_naming_all_groups_named_by_vlm(tmp_path):
    store, rows, *_ , settings = _world(tmp_path)
    llm = FakeLlm()
    result = naming.run(store, "g", settings, llm, job_id=None)
    back = store.read_assignments("g")
    n_groups = len({r.embed_group_id for r in rows})
    assert len(back) == n_groups == result["groups"]
    assert all(a.assigned_by == "vlm" for a in back)
    assert all(not a.needs_review for a in back)          # conf 높고 CLIP 일치
    assert result["vlmGroups"] == n_groups and result["nearestGroups"] == 0


def test_naming_nearest_inherits_and_flags_far_groups(tmp_path):
    store, rows, *_, settings = _world(tmp_path, n_groups=4, per_group=8, clip_parent=None)
    k = Knobs(naming_max_groups=2, nearest_tau=0.05)    # 상위 2그룹만 VLM, τ 빡빡하게
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        knobs=k, llm=settings.llm)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
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
    store, rows, *_, settings = _world(tmp_path, clip_parent="야외 자연")   # VLM(실내 스튜디오)과 불일치
    gids = sorted({r.embed_group_id for r in rows})
    llm = FakeLlm(low_conf_gid=gids[0])
    naming.run(store, "g", settings, llm, job_id=None)
    back = store.read_assignments("g")
    assert all(a.needs_review for a in back)               # 불일치라 전부 review
    assert all(a.clip_parent == "야외 자연" for a in back)


def test_naming_merge_call_unifies_names_across_chunks(tmp_path):
    store, rows, *_, settings = _world(tmp_path, n_groups=5, per_group=6)
    # spread 대표 추가를 꺼서(임계 9) 그룹당 1장 → 그룹 5개 = vision 3청크 + merge 1
    k = Knobs(naming_chunk=2, naming_spread_extra=9.0)
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        knobs=k, llm=settings.llm)
    llm = FakeLlm()
    result = naming.run(store, "g", settings, llm, job_id=None)
    kinds = [kind for kind, _ in llm.calls]
    assert kinds.count("vision") == 3 and kinds.count("merge") == 1
    assert result["llmCalls"] == 4
    assert all(a.concept_name.endswith("(통일)") for a in store.read_assignments("g"))


def test_naming_coverage_target_limits_vlm_groups(tmp_path):
    store, rows, *_, settings = _world(tmp_path, n_groups=4, per_group=8, clip_parent=None)
    # 커버리지 50% → 크기순 일부 그룹만 VLM, 나머지는 nearest 상속 (review-v3-design.md (3))
    k = Knobs(naming_coverage=0.5, nearest_tau=1.0)
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        knobs=k, llm=settings.llm)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["vlmGroups"] < result["groups"]
    assert result["nearestGroups"] == result["groups"] - result["vlmGroups"]
    assert 0.5 <= result["coverage"] < 1.0          # 목표 도달 지점에서 멈춘 실측치


def test_naming_spread_adds_second_rep(tmp_path):
    store, rows, *_, settings = _world(tmp_path)
    # 임계 0 → 모든 그룹이 이질 판정 → 중심 최근접 + 최원점 2장 (review-v3-design.md (2))
    k = Knobs(naming_spread_extra=0.0)
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        knobs=k, llm=settings.llm)
    llm = FakeLlm()
    result = naming.run(store, "g", settings, llm, job_id=None)
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
        naming.run(store, "empty", settings, FakeLlm())


def test_parents_fixed_list():
    assert "기타" in PARENTS
    assert len(PARENTS) == len(set(PARENTS))


# ── SCORE / CATEGORIZE 분리 (#26) ─────────────────────────────────────────────
class _FakeLaion:
    """CLIP 흉내 — 이미지·텍스트 모두 32차원 단위 벡터. torch 없음."""

    def __init__(self, dim=32, seed=7):
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.embed_calls = 0

    def embed(self, path):
        self.embed_calls += 1
        return _unit(self.rng.normal(size=self.dim))

    def embed_texts(self, prompts):
        return np.stack([_unit(self.rng.normal(size=self.dim)) for _ in prompts])

    def score_from_embedding(self, emb):
        return 5.5


class _FakeArniqa:
    def score(self, path):
        return {"technical_score": 0.6}


@pytest.fixture
def fake_runners(monkeypatch):
    """score.run 이 함수 안에서 import 하는 torch 러너·classical 을 가짜로 바꾼다."""
    laion = _FakeLaion()
    runners = types.ModuleType("photoselect.runners")
    runners.LaionRunner = lambda: laion
    runners.ArniqaRunner = lambda: _FakeArniqa()
    classical = types.ModuleType("photoselect.classical")
    classical.measure = lambda path: {"sharpness": 100.0, "highlight_clip": 0.0, "shadow_clip": 0.0, "mean_luma": 120.0}
    monkeypatch.setitem(sys.modules, "photoselect.runners", runners)
    monkeypatch.setitem(sys.modules, "photoselect.classical", classical)
    return laion


def _refs(store, gallery, rows):
    return [PhotoRef(photo_id=r.photo_id, path=store.preview_path(gallery, r.photo_id)) for r in rows]


def test_score_skips_scored_photos_and_stores_clip_parent(tmp_path, fake_runners):
    store, rows, *_, settings = _world(tmp_path)
    refs = _refs(store, "g", rows)

    # 이미 같은 MODEL_VERSION + CLIP 벡터가 있다 → 전부 건너뛴다. 임베더 벡터 유무는 보지 않는다.
    result = score.run(store, "g", refs, settings)
    assert result["skipped"] == len(rows) and result["processed"] == 0
    assert fake_runners.embed_calls == 0

    # 새 사진 한 장만 추가되면 그것만 점수를 낸다. clip_parent·subjects 가 sub_scores 에 실린다.
    new_id = "new-photo.jpg"
    Image.new("RGB", (32, 32)).save(store.dataset_root / new_id)
    result = score.run(store, "g", refs + [PhotoRef(photo_id=new_id, path=str(store.dataset_root / new_id))], settings)
    assert result["processed"] == 1 and result["skipped"] == len(rows)
    back = {r.photo_id: r for r in store.read_analysis("g")}
    assert back[new_id].sub_scores["clip_parent"] in PARENTS
    assert "technical_score" in back[new_id].sub_scores and back[new_id].model_version == MODEL_VERSION
    ids, C = store.read_clip_embeddings("g")
    assert new_id in ids and C.shape[0] == len(rows) + 1

    # force 면 전부 다시.
    result = score.run(store, "g", refs, settings, force=True)
    assert result["processed"] == len(rows)


def test_write_scores_keeps_group_columns_and_write_groups_keeps_score_columns(tmp_path):
    store, rows, *_ = _world(tmp_path)
    before = {r.photo_id: (r.technical_pct, r.cluster_id, r.embed_group_id, r.subjects) for r in rows}

    # SCORE 의 쓰기: pct·cluster·group 기본값을 가진 행을 넘겨도 그 컬럼은 그대로다
    fresh = [PhotoAnalysis(photo_id=r.photo_id, subjects="bride", sub_scores=dict(r.sub_scores),
                           model_version=MODEL_VERSION) for r in rows]
    store.write_scores("g", fresh, ([], np.zeros((0, 0))))
    after = {r.photo_id: r for r in store.read_analysis("g")}
    for pid, (pct, cid, gid, _) in before.items():
        assert (after[pid].technical_pct, after[pid].cluster_id, after[pid].embed_group_id) == (pct, cid, gid)
        assert after[pid].subjects == "bride"

    # CATEGORIZE 의 쓰기: subjects 는 건드리지 않는다
    grouped = [PhotoAnalysis(photo_id=r.photo_id, subjects="unknown", technical_pct=1.0,
                             cluster_id=99, embed_group_id=7) for r in rows]
    store.write_groups("g", grouped)
    after = {r.photo_id: r for r in store.read_analysis("g")}
    assert all(after[pid].subjects == "bride" for pid in before)
    assert all(after[pid].embed_group_id == 7 and after[pid].cluster_id == 99 for pid in before)


def test_categorize_groups_then_names_without_torch(tmp_path):
    store, rows, E, C, settings = _world(tmp_path)
    refs = [PhotoRef(photo_id=r.photo_id, path=None) for r in rows]   # CATEGORIZE 는 파일이 필요 없다
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        knobs=Knobs(group_distance=0.4, group_min_groups=2, group_max_share=0.6),
                        llm=settings.llm)
    llm = FakeLlm()

    result = categorize.run(store, "g", refs, settings, llm, job_id=None)

    assert result["mode"] == "categorize" and result["photos"] == len(rows)
    assert result["embeddingsSource"] == "dinov3"          # 로컬 world 는 E 를 저장해 두었다
    assert result["clusters"] == len(rows)                 # 순서·시각 없음 → 연사 없음
    assert result["naming"]["vlmGroups"] == result["groups"]["groups"]
    back = store.read_analysis("g")
    assert all(r.embed_group_id >= 0 and "sharpness_pct" in r.sub_scores for r in back)
    assert all(a.assigned_by == "vlm" for a in store.read_assignments("g"))


def test_categorize_falls_back_to_clip_when_no_embedder_vectors(tmp_path):
    store, rows, E, C, settings = _world(tmp_path)
    ids = [r.photo_id for r in rows]
    store.write_analysis("g", rows, ([], np.zeros((0, 0))), (ids, C))   # 임베더 벡터 없음(로컬)
    refs = [PhotoRef(photo_id=r.photo_id, path=None) for r in rows]

    result = categorize.run(store, "g", refs, settings, None)

    assert result["embeddingsSource"] == "clip"
    assert result["naming"] == "skipped (no --llm)"


def test_categorize_and_naming_do_not_import_torch():
    code = ("import sys; import photoselect.categorize, photoselect.naming; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True)


def test_worker_maps_wes_modes_to_steps():
    from photoselect import worker
    assert worker.MODE_STEPS == {"FULL": ("score", "categorize"), "NAMING": ("categorize",)}


def test_majority_ignores_none_and_breaks_ties_by_first_seen():
    assert majority(["야외 자연", None, "야외 자연", "실내 스튜디오"]) == "야외 자연"
    assert majority([None, None]) is None
    assert majority(["a", "b"]) == "a"
