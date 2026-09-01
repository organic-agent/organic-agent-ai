"""v2 단위 테스트 — 합성 데이터로 컨셉 그룹·재랭킹·점수·근거·초안 한 라운드를 돈다. 모델 없음."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from photoselect.v2.config import Settings, V2Knobs, V2_MODEL_VERSION
from photoselect.v2.store import LocalStore, PhotoAnalysis
from photoselect.v2 import classical, concept, draft, reasons, rerank, scoring
from photoselect.v2.analyze import assign_ranks, percentile


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _world(tmp_path, n_groups=3, per_group=12, seed=0):
    """그룹마다 중심 벡터 + 작은 노이즈. 그룹 안에 연사(거의 같은 벡터) 쌍을 섞는다."""
    rng = np.random.default_rng(seed)
    dim = 32
    centers = [_unit(rng.normal(size=dim)) for _ in range(n_groups)]
    rows, E = [], []
    for g, c in enumerate(centers):
        for j in range(per_group):
            base = c + 0.1 * rng.normal(size=dim)
            if j % 4 == 1:                       # 앞 사진의 연사
                base = E[-1] + 0.01 * rng.normal(size=dim)
            E.append(_unit(base))
            pid = f"g{g}-{j:02d}"
            rows.append(PhotoAnalysis(
                photo_id=pid, scene="unknown", framing="unknown", lighting="unknown", expression="unknown",
                subjects=["bride", "groom", "couple"][(g + j) % 3],
                sub_scores={"technical_score": rng.uniform(0.3, 0.8), "aesthetic_score": rng.uniform(5, 6.5),
                            "sharpness": rng.uniform(50, 500), "highlight_clip": 0.0, "shadow_clip": 0.0},
                model_version=V2_MODEL_VERSION))
    E = np.stack(E)
    for r, t in zip(rows, percentile([r.sub_scores["technical_score"] for r in rows])):
        r.technical_pct = t
    for r, a in zip(rows, percentile([r.sub_scores["aesthetic_score"] for r in rows])):
        r.aesthetic_pct = a
    for r, s in zip(rows, percentile([r.sub_scores["sharpness"] for r in rows])):
        r.sub_scores["sharpness_pct"] = s
    from photoselect.v2 import cluster
    cids = cluster.cluster_bursts(E, 0.99, 8)
    gids, _ = concept.concept_groups(E, 0.4, min_groups=2, max_share=0.6)
    for r, c, g in zip(rows, cids, gids):
        r.cluster_id, r.concept_id = int(c), int(g)
    assign_ranks(rows)
    store = LocalStore(tmp_path)
    store.write_analysis("g", rows, ([r.photo_id for r in rows], E))
    settings = Settings(out_root=tmp_path, dataset_root=tmp_path)
    return store, rows, E, settings


def test_concept_groups_recover_planted_groups(tmp_path):
    store, rows, E, _ = _world(tmp_path)
    gids = np.array([r.concept_id for r in rows])
    # 심은 그룹(사진 id 접두사)과 찾은 그룹이 1:1 로 맞아야 한다
    planted = np.array([int(r.photo_id[1]) for r in rows])
    for g in set(planted):
        assert len(set(gids[planted == g])) == 1
    assert len(set(gids)) == 3


def test_concept_groups_auto_tighten_when_one_blob():
    rng = np.random.default_rng(1)
    E = np.stack([_unit(rng.normal(size=16)) for _ in range(40)])
    gids, used = concept.concept_groups(E, distance=1.0, min_groups=4, max_share=0.5)
    assert used < 1.0 and len(set(gids.tolist())) >= 4


def test_assign_ranks_prefers_technical_then_sharpness():
    a = PhotoAnalysis("a", technical_pct=80, sub_scores={"sharpness": 100}, cluster_id=0)
    b = PhotoAnalysis("b", technical_pct=70, sub_scores={"sharpness": 900}, cluster_id=0)
    c = PhotoAnalysis("c", technical_pct=80, sub_scores={"sharpness": 300}, cluster_id=0)
    assign_ranks([a, b, c])
    assert c.cluster_rank == 0 and c.rank_reason_code in ("technical", "sharpness", "tie")
    assert a.rank_reason_code == "" and b.cluster_rank == 2


def test_coverage_quota_cap_and_floor():
    groups = [0] * 30 + [1] * 6 + [2] * 2 + [3]
    q = rerank.coverage_quota(groups, 10, 1, 0.4)
    assert q[2] >= 1 and q[1] >= 1 and q[0] <= 4 and sum(q.values()) == 10 and 3 not in q
    # 그룹이 k 보다 많으면 큰 그룹부터 k 개만 1장씩
    many = [g for g in range(40) for _ in range(2)]
    q2 = rerank.coverage_quota(many, 10, 1, 0.4)
    assert sum(q2.values()) == 10 and sum(1 for v in q2.values() if v > 0) == 10


def test_explain_selection_one_per_cluster_and_exclude(tmp_path):
    store, rows, E, settings = _world(tmp_path)
    score = np.array([r.technical_pct for r in rows], dtype=float)
    ex = {0, 1}
    picks = rerank.explain_selection(score, E, [r.concept_id for r in rows], [r.cluster_id for r in rows],
                                     10, 0.7, 1, 0.4, exclude=ex)
    idx = [p.index for p in picks]
    assert len(idx) == 10 and not (set(idx) & ex)
    assert len({rows[i].cluster_id for i in idx}) == 10          # 연사 클러스터당 1장
    assert len({rows[i].concept_id for i in idx}) == 3           # 세 그룹 모두 등장


def test_type_stats_and_balance():
    types = ["bride"] * 10 + ["groom"] * 10 + ["couple"] * 20
    mask = np.zeros(40, dtype=bool)
    mask[10:16] = True          # groom 6장만 담음
    st = scoring.type_stats(types, mask)
    assert st["bride"]["deficit"] == pytest.approx(0.25) and st["groom"]["deficit"] == 0
    assert st["groom"]["select_lift"] > 0 > st["bride"]["select_lift"]
    prior = np.arange(40, dtype=float)
    s, bd = scoring.combine(prior, types, st, V2Knobs())
    assert bd["balance_z"][0] > bd["balance_z"][10]             # 부족한 유형이 올라간다


def test_reasons_template_and_facts():
    material = {"prior_z": 1.5, "sibling": {"n": 4, "why_counts": {"덜 선명함": 2, "거의 같은 컷": 1}, "sharpest": True},
                "quality": {"aesthetic_top": 5, "descriptors": ["초점이 또렷함"]}}
    primary, facts = reasons.facts_of(material)
    assert primary == "quality" and any(f.startswith("형제") for f in facts) and any(f.startswith("품질") for f in facts)
    text = reasons.template(primary, material)
    assert "상위 5%" in text and "연사 4장" in text and len(text) <= 60
    assert "덜 선명함 2" in reasons.template("sibling", material)
    assert reasons.facts_of({"prior_z": 0.1}) == ("diversity", ["다양성: 앞서 고른 사진들과 배경·구도가 겹치지 않아 뽑힘 (점수는 평범)"])
    assert reasons.template("concept", {"concept": {"size": 37}}).startswith("이 배경으로 찍은 37장")


def test_reasons_generate_guardrails():
    class Fake:
        def complete_json(self, system, user, schema, max_tokens):
            return {"reasons": [{"photo_id": "a", "reason": "짧고 좋은 문장이에요"},
                                {"photo_id": "zzz", "reason": "요청에 없는 사진"},
                                {"photo_id": "b", "reason": "x" * 61}]}
    items = [reasons.ReasonInput("a", "quality", ["품질: 미학 상위 5%"], "템플릿 a"),
             reasons.ReasonInput("b", "quality", [], "템플릿 b")]
    out = reasons.generate(Fake(), items)
    assert out == {"a": "짧고 좋은 문장이에요", "b": "템플릿 b"}


def test_draft_round_trip(tmp_path):
    store, rows, E, settings = _world(tmp_path)
    r1 = draft.run(store, "g", settings, top_k=10, target=20)
    assert r1["k"] == 10 and r1["pipeline"] == "v2" and len(r1["conceptDistribution"]) == 3
    recs = store.read_recommendations("g")
    assert len(recs) == 10 and all(r.reason for r in recs)
    assert all(r.score_breakdown["primary_reason"] in reasons.REASON_PRIORITY for r in recs)
    # 2라운드: 1라운드에 보여준 사진은 다시 안 나온다. 담은 사진도 제외.
    (store._dir("g") / "evidence.json").write_text(json.dumps({"selected": [recs[0].photo_id, recs[1].photo_id]}))
    r2 = draft.run(store, "g", settings, top_k=10, target=20)
    assert r2["round"] == 2 and r2["shownBefore"] == 10 and r2["selected"] == 2
    ids2 = {r.photo_id for r in store.read_recommendations("g") if r.round == 2}
    assert not ids2 & {r.photo_id for r in recs}
    # 목표 도달 → done
    r3 = draft.run(store, "g", settings, top_k=10, target=2)
    assert r3["done"] is True and r3["k"] == 0


def test_draft_balance_when_subjects_trusted(tmp_path):
    from dataclasses import replace
    store, rows, E, settings = _world(tmp_path)
    settings = replace(settings, v2=replace(settings.v2, subjects_trusted=True, pref_min_selected=3))
    grooms = [r.photo_id for r in rows if r.subjects == "groom"][:5]
    (store._dir("g") / "evidence.json").write_text(json.dumps({"selected": grooms}))
    out = draft.run(store, "g", settings, top_k=8, target=30)
    assert out["preferenceOn"] and out["typeStats"]["groom"]["deficit"] == 0
    recs = store.read_recommendations("g")
    assert any(r.score_breakdown["primary_reason"] == "balance" for r in recs)
    assert all(r.score_breakdown["subjects"] != "groom" or r.score_breakdown["primary_reason"] != "balance" for r in recs)


def test_classical_measures_sharpness_ordering():
    rng = np.random.default_rng(0)
    noise = (rng.uniform(0, 255, size=(300, 400))).astype("uint8")
    sharp = Image.fromarray(noise).convert("RGB")
    from PIL import ImageFilter
    blurred = sharp.filter(ImageFilter.GaussianBlur(4))
    ms, mb = classical.measure_image(sharp), classical.measure_image(blurred)
    assert ms["sharpness"] > 5 * mb["sharpness"]
    assert 0 <= ms["highlight_clip"] <= 1 and 0 <= ms["shadow_clip"] <= 1
