"""합성 갤러리로 학습·평가·게이트·직렬화·golden 로더를 검증한다. DB 없음."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from preference.config.settings import Knobs, Settings
from preference.domain.features import FEATURE_SPEC, N_EMB, N_SCALAR
from preference.domain.gallery import GalleryData, LabeledGallery
from preference.domain.model import PreferenceModel, lam, prior, z
from preference.repository.golden import load_golden
from preference.repository.local_store import LocalStore
from preference.service import job
from preference.service.evaluate import auc, evaluate_gallery, gate, logo, recall_at_k, recalls, sign_test
from preference.service.features import build
from preference.service.train import make_sample, train

GOLDEN = Path(__file__).resolve().parent.parent / "golden" / "파일명_정리.xlsx"
GOLDEN_JSON = Path(__file__).resolve().parent.parent / "golden" / "dataset1-golden.json"


# ── 합성 갤러리 ─────────────────────────────────────────────────────────────
def synth(gallery_id: str, n: int = 600, n_pos: int = 20, seed: int = 0, taste: np.ndarray | None = None,
          burst: int = 3) -> LabeledGallery:
    """n 장, burst 장씩 연사 클러스터. 숨은 취향 방향 taste 에 가까운 사진을 고른다 — prior 와는 무관하게."""
    rng = np.random.default_rng(seed)
    dino = rng.normal(size=(n, 768)); clip = rng.normal(size=(n, 768))
    # 연사: 같은 클러스터는 거의 같은 벡터
    cluster = np.arange(n) // burst
    for c in np.unique(cluster):
        m = cluster == c
        dino[m] = dino[m][0] + 0.02 * rng.normal(size=(m.sum(), 768))
        clip[m] = clip[m][0] + 0.02 * rng.normal(size=(m.sum(), 768))
    rank = np.arange(n) % burst
    taste = taste if taste is not None else np.concatenate([rng.normal(size=768), np.zeros(768)])
    emb = np.concatenate([dino / np.linalg.norm(dino, axis=1, keepdims=True), clip / np.linalg.norm(clip, axis=1, keepdims=True)], axis=1)
    affinity = emb @ taste
    # 클러스터당 하나(대표)만 후보로 두고 취향 상위 n_pos 를 선택
    reps = np.where(rank == 0)[0]
    chosen = reps[np.argsort(-affinity[reps])[:n_pos]]
    gd = GalleryData(
        gallery_id=gallery_id, photo_ids=[f"{gallery_id}-{i}" for i in range(n)],
        file_names=[f"IMG{i:05d}.JPG" for i in range(n)],
        technical_pct=rng.uniform(0, 100, n), aesthetic_pct=rng.uniform(0, 100, n),
        sharpness_pct=rng.uniform(0, 100, n),
        subjects=list(rng.choice(["bride", "groom", "couple", "group", "unknown"], n)),
        cluster_id=cluster, cluster_rank=rank, embed_group_id=rng.integers(0, 12, n),
        display_order=np.arange(n), embedding=dino.astype(np.float32), clip_embedding=clip.astype(np.float32),
        embedding_model="dinov3-test", model_version="test-0.1", shoot_type="REHEARSAL",
    )
    return LabeledGallery(data=gd, selected_ids=[gd.photo_ids[i] for i in chosen])


KNOBS = Knobs(n0=5.0)


def test_features_shape_and_centering():
    lg = synth("g")
    f = build(lg.data)
    assert f.scalar.shape == (600, N_SCALAR) and f.emb.shape == (600, N_EMB)
    assert np.allclose(f.emb.mean(axis=0), 0, atol=1e-9)
    assert set(np.unique(f.scalar[:, 3:7])) <= {0.0, 1.0}


def test_sample_excludes_siblings_and_uses_cluster_reps():
    lg = synth("g", burst=3)
    s = make_sample(lg)
    pos = lg.positive_idx
    cid = lg.data.cluster_id
    pos_clusters = set(cid[pos].tolist())
    neg = s.idx[s.y == 0]
    assert not (set(cid[neg].tolist()) & pos_clusters)             # 양성 클러스터의 형제는 음성에 없다
    assert len(neg) == len(set(cid[neg].tolist()))                 # 클러스터당 음성 1장
    assert abs(s.w[s.y == 1].sum() - 0.5) < 1e-9 and abs(s.w[s.y == 0].sum() - 0.5) < 1e-9


def test_train_learns_taste_beyond_prior():
    lg = synth("g", seed=1)
    model = train([lg], KNOBS)
    ev = evaluate_gallery(lg, model, KNOBS, lam_override=1.0)
    assert ev.metrics["pref"]["auc"] > 0.95                        # in-sample 은 외운다
    assert ev.metrics["pref"]["recall_cluster"] == 1.0                # 클러스터 기준 — 형제 중 어느 장인지는 못 가른다
    assert ev.metrics["prior"]["auc"] < 0.7                         # prior 는 취향과 무관
    assert model.lam == pytest.approx(lam(1, 5.0)) == pytest.approx(1 / 6)


def test_logo_generalizes_when_taste_is_shared():
    rng = np.random.default_rng(7)
    taste = np.concatenate([rng.normal(size=768), np.zeros(768)])
    gs = [synth(f"g{i}", seed=10 + i, taste=taste) for i in range(4)]
    rows = logo(gs, KNOBS)
    assert len(rows) == 4
    for r in rows:
        assert r.metrics["pref"]["auc"] > 0.6                       # 다른 갤러리에서도 방향을 찾는다 (양성 20 × 3 갤러리)
        assert r.metrics["fused"]["recall_cluster"] >= r.metrics["fused"]["recall_strict"]
    assert np.mean([r.metrics["fused"]["recall_cluster"] for r in rows]) > np.mean([r.metrics["prior"]["recall_cluster"] for r in rows])


def test_logo_needs_two_galleries_and_gate_reports_it():
    verdict = gate([synth("g")], KNOBS)
    assert verdict["passed"] is False and "측정 불가" in verdict["reasons"][0]


def test_gate_passes_on_shared_taste_and_fails_on_random():
    rng = np.random.default_rng(3)
    taste = np.concatenate([rng.normal(size=768), np.zeros(768)])
    shared = [synth(f"s{i}", seed=100 + i, taste=taste, n=300, n_pos=15) for i in range(6)]
    knobs = Knobs(n0=5.0, gate_p=0.05)
    v = gate(shared, knobs)
    assert v["mean_diff"] > 0 and v["p_value"] < 0.05
    assert v["passed"] is True, v["reasons"]
    assert len(v["curve"]) == knobs.gate_stability_window + 1
    # 갤러리마다 다른 취향 → 유의한 개선 없음. 갤러리 6개면 부호 검정이 5/6 우연(11%)에도 열리므로 10개로 본다 —
    # 게이트가 소규모 n 에서 약하다는 사실 자체가 plan §5 의 "갤러리 수를 늘려가며" 의 근거다.
    random = [synth(f"r{i}", seed=300 + i, n=300, n_pos=15) for i in range(10)]
    v2 = gate(random, knobs)
    assert v2["passed"] is False and v2["p_value"] > 0.05


def test_metrics_helpers():
    scores = np.array([0.1, 0.9, 0.5, 0.7])
    assert recall_at_k(scores, np.array([1, 0]), 2) == 0.5
    assert recall_at_k(scores, np.array([0]), 1, cluster_id=np.array([5, 5, 6, 7])) == 1.0   # 형제(1번)가 대표로 올라갔다
    assert recalls(scores, np.array([0]), 1, np.array([5, 5, 6, 7])) == (0.0, 1.0)
    assert recalls(scores, np.array([1, 3]), 2, np.array([5, 5, 6, 7])) == (1.0, 1.0)         # dedup 뒤 상위 2 = 1번·3번
    assert auc(scores, np.array([1]), np.array([0, 2])) == 1.0
    assert sign_test(np.array([1, 1, 1, 1, 1, 1.0])) < 0.05 and sign_test(np.array([1, -1, 1, -1.0])) > 0.3
    assert z(np.array([1.0, 1.0])).tolist() == [0.0, 0.0]
    assert prior(np.array([100.0]), np.array([0.0])).tolist() == [50.0]


def test_model_roundtrip_and_fuse_matches_formula():
    lg = synth("g", seed=2)
    model = train([lg], KNOBS)
    row = model.to_row()
    assert row["feature_spec"] == FEATURE_SPEC and len(row["w_emb"]) == N_EMB and len(row["w_scalar"]) == N_SCALAR
    back = PreferenceModel.from_row(json.loads(json.dumps(row)))
    f = build(lg.data)
    pr = prior(lg.data.technical_pct, lg.data.aesthetic_pct)
    expected = z(pr) + back.lam * z(f.scalar @ back.w_scalar + f.emb @ back.w_emb + back.bias)
    assert np.allclose(back.fuse(pr, f), expected)
    assert np.allclose(back.raw(f), model.raw(f))


def test_local_store_roundtrip_and_train_job(tmp_path: Path):
    st = LocalStore(tmp_path)
    for i in range(2):
        lg = synth(f"g{i}", seed=30 + i)
        st.write_gallery(lg.data)
        st.write_selected(lg.data.gallery_id, lg.selected_ids)
    back = st.read_gallery("g0")
    assert back.n == 600 and back.embedding_model == "dinov3-test" and back.shoot_type == "REHEARSAL"
    assert st.list_closed_galleries() == ["g0", "g1"]
    settings = Settings(out_root=tmp_path, knobs=KNOBS)
    result = job.run_train(st, settings, fallback=st)
    assert result["status"] == "ok" and result["stored"] == "local" and result["active"] is False
    assert result["gate"]["n_galleries"] == 2 and result["n_galleries"] == 2
    saved = json.loads((tmp_path / "preference" / "models" / "0001.json").read_text())
    assert saved["holdout"]["n_galleries"] == 2 and len(saved["holdout"]["rows"]) == 2


def test_sanity_job(tmp_path: Path):
    st = LocalStore(tmp_path)
    lg = synth("g", seed=4)
    st.write_gallery(lg.data)
    result = job.run_sanity(st, Settings(out_root=tmp_path, knobs=KNOBS), "g", lg.selected_ids)
    assert result["passed"] is True and result["n_positives_matched"] == 20
    assert result["n_excluded_siblings"] == 40          # 20 클러스터 × 형제 2장
    assert result["k"] == 60 and "prior" in result["metrics_lambda_1"]


def test_golden_loader_reads_30():
    items = load_golden(GOLDEN)
    assert len(items) == 30
    assert sum(1 for it in items if it.cut == "A") == 20 and sum(1 for it in items if it.cut == "B") == 10
    assert items[0].file_name == "LWH00032.JPG" and all(it.file_name.endswith(".JPG") for it in items)


def test_golden_loader_reads_json_20():
    items = load_golden(GOLDEN_JSON)
    assert len(items) == 20
    assert all(it.cut == "A" for it in items)            # 컷 구분이 없는 데이터셋은 전부 A컷
    assert items[0].file_name == "1BE00033.JPG"


def test_golden_loader_json_accepts_bare_list_and_cuts(tmp_path):
    p = tmp_path / "g.json"
    p.write_text('["a.JPG", {"cut": "b", "file": "b.JPG"}]', encoding="utf-8")
    items = load_golden(p)
    assert [(it.cut, it.file_name) for it in items] == [("A", "a.JPG"), ("B", "b.JPG")]
