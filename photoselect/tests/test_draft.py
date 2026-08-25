"""B(초안) 단위 테스트 — torch 없이 돈다. 합성 분석 결과로 store→draft→recommendations 한 바퀴.

두 단계의 계약을 고정한다:
  1단계  evidence 없음 → λ=λ_min, pref=0, 사진학 + 커버리지 + MMR
  2단계  evidence 있음 → 같은 코드가 개인화되고, 담은 사진·이미 제시한 사진은 후보에서 빠진다
  종료   담긴 수가 목표 장수에 닿으면 done, 그 전엔 남은 자리만큼만 제시
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from photoselect.axes import AXES
from photoselect.config import ScoreKnobs, Settings
from photoselect.draft import job, rerank, scoring
from photoselect.store import Evidence, LocalStore, PhotoAnalysis


def _synthetic(n: int = 120, seed: int = 0) -> tuple[list[PhotoAnalysis], list[str], np.ndarray]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    rows, ids = [], []
    cid = 0
    i = 0
    while i < n:
        size = rng.randint(1, 4)
        scene = rng.choice(AXES["scene"][:-1])
        base = np_rng.normal(size=64)
        for _ in range(min(size, n - i)):
            pid = f"p{i:04d}.jpg"
            rows.append(PhotoAnalysis(
                photo_id=pid, scene=scene,
                framing=rng.choice(AXES["framing"]), lighting=rng.choice(AXES["lighting"]),
                expression=rng.choice(AXES["expression"][:4]), subjects=rng.choice(AXES["subjects"][:5]),
                technical_pct=rng.uniform(0, 100), aesthetic_pct=rng.uniform(0, 100),
                face_boxes={"face_count": 1.0}, sub_scores={"eyes_open": rng.random()},
                cluster_id=cid, model_version="test",
            ))
            ids.append(pid)
            i += 1
        cid += 1
    emb = np.stack([np_rng.normal(size=64) for _ in ids])
    # 같은 클러스터는 닮게
    for r_idx, r in enumerate(rows):
        emb[r_idx] = emb[[k for k, x in enumerate(rows) if x.cluster_id == r.cluster_id][0]] + 0.1 * np_rng.normal(size=64)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return rows, ids, emb


@pytest.fixture
def world(tmp_path):
    rows, ids, emb = _synthetic()
    st = LocalStore(tmp_path)
    st.write_analysis("g", rows, (ids, emb))
    settings = Settings(out_root=tmp_path, dataset_root=tmp_path)
    return st, settings, rows


def test_phase1_prior_only(world):
    st, settings, rows = world
    out = job.run(st, "g", settings, round_no=1)
    assert out["k"] == settings.score.top_k
    assert out["lambda"] == pytest.approx(settings.score.lambda_min)
    assert out["evidence"] == {"pair": 0, "rating": 0, "selection": 0}
    recs = st.read_recommendations("g")
    assert len(recs) == out["k"]
    # 클러스터당 1장
    cl = [next(r for r in rows if r.photo_id == x.photo_id).cluster_id for x in recs]
    assert len(cl) == len(set(cl))
    # 장면 상한
    cap = int(round(settings.score.scene_cap_ratio * settings.score.top_k))
    assert max(out["sceneDistribution"].values()) <= cap
    # 순위 = 점수 내림차순
    scores = [r.score_breakdown["score"] for r in recs]
    assert scores == sorted(scores, reverse=True)


def test_phase2_personalized_and_excludes_selected(world, tmp_path):
    st, settings, rows = world
    job.run(st, "g", settings, round_no=1)          # 1단계 먼저 — 2단계가 그 위에 쌓인다
    # 온보딩 쌍 12개: closeup을 일관되게 선호하는 고객
    close = [r for r in rows if r.framing == "closeup"]
    other = [r for r in rows if r.framing != "closeup"]
    pairs = [(c.photo_id, o.photo_id, "framing") for c, o in zip(close[:12], other[:12])]
    selected = [r.photo_id for r in rows[:5]]
    (tmp_path / "g" / "evidence.json").write_text(
        __import__("json").dumps({"selected": selected, "ratings": {rows[6].photo_id: 5, rows[7].photo_id: 1},
                                  "pairs": pairs}), encoding="utf-8")
    out = job.run(st, "g", settings, round_no=2)
    assert out["evidence"]["pair"] == 12
    assert out["lambda"] > settings.score.lambda_min
    shown1 = {r.photo_id for r in st.read_recommendations("g") if r.round == 1}
    assert out["excluded"] == len(set(selected) | shown1)      # 담은 것 + 1라운드에 제시한 것
    recs = st.read_recommendations("g")
    r2 = [r for r in recs if r.round == 2]
    assert not {r.photo_id for r in r2} & set(selected), "담은 사진이 다시 추천되면 안 된다"
    # 라운드 1은 보존
    assert any(r.round == 1 for r in recs)
    # framing conf가 올라가고, closeup이 선호로 잡힌다
    assert out["axisConfidence"]["framing"] > 0.3
    assert ("framing", "closeup") in [tuple(p) for p in out["preferences"]]


def test_lambda_curve():
    k = ScoreKnobs()
    assert scoring.lambda_of(0, k) == pytest.approx(k.lambda_min)
    assert scoring.lambda_of(1e9, k) == pytest.approx(k.lambda_max, abs=1e-3)
    assert scoring.lambda_of(k.lambda_k, k) == pytest.approx((k.lambda_min + k.lambda_max) / 2)


def test_coverage_cap_redistributes():
    scenes = ["snap"] * 68 + ["walk"] * 12 + ["prep"] * 8 + ["kiss"] * 6 + ["detail"] * 6
    q = rerank.coverage_quota(scenes, 30, 1, 0.4)
    assert sum(q.values()) == 30
    assert q["snap"] <= 12
    assert all(v >= 1 for v in q.values())


def test_mmr_dedups_twins():
    score = np.array([1.0, 0.99, 0.5, 0.4])
    emb = np.array([[1, 0], [1, 0.01], [0, 1], [0.7, 0.7]], dtype=float)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    assert rerank.mmr_select(score, emb, 2, 0.5)[:2] == [0, 2]


def test_shown_photos_not_repeated_and_target_stops(world, tmp_path):
    st, settings, rows = world
    r1 = job.run(st, "g", settings, round_no=1, target=40)
    shown1 = {r.photo_id for r in st.read_recommendations("g")}
    assert r1["remaining"] == 40 and r1["k"] == 30
    # 담기 없이 다음 라운드 → 제시했던 30장은 안 나온다, 남은 자리 40장만큼
    r2 = job.run(st, "g", settings, round_no=2, target=40)
    shown2 = {r.photo_id for r in st.read_recommendations("g") if r.round == 2}
    assert not shown1 & shown2
    assert r2["shownBefore"] == 30
    # 40장 담으면 완료
    import json
    (tmp_path / "g" / "evidence.json").write_text(json.dumps({"selected": [r.photo_id for r in rows[:40]]}), encoding="utf-8")
    r3 = job.run(st, "g", settings, round_no=3, target=40)
    assert r3["done"] is True and r3["k"] == 0
    # 35장 담으면 5장만 제시
    (tmp_path / "g" / "evidence.json").write_text(json.dumps({"selected": [r.photo_id for r in rows[:35]]}), encoding="utf-8")
    r4 = job.run(st, "g", settings, round_no=4, target=40)
    assert r4["k"] == 5 and r4["remaining"] == 5
