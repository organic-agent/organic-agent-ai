"""초안 한 라운드. 진입점(`__main__ draft` / Lambda handler)이 이 함수를 부른다.

    store.read_analysis + read_embeddings        ← A의 산출물
    store.read_evidence                          ← 선택·별점·쌍 (없으면 사진학만)
    evidence → BT 쌍 → w_hat, conf(axis)         (draft/preference, draft/evidence)
    prior, pref, λ → score                       (draft/scoring)
    후보 = 전체 − 이미 담은 사진 − 이미 제시한 사진   (plan.md §3-B 후보 집합)
    제시 장수 = min(K, 목표 − 담긴 수). 담긴 수 ≥ 목표면 완료.
    커버리지 + 클러스터당 1장 + MMR → K장           (draft/rerank)
    템플릿 이유 문장 → store.write_recommendations

1단계(개인화 전): evidence가 비어 있으면 λ=λ_min, pref=0 → 사실상 사진학 + 커버리지 + MMR.
2단계: evidence.json(또는 DB)이 채워지면 같은 코드가 개인화된다. 분기가 없다.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from photoselect.axes import LABELS_KO, onehot
from photoselect.config import Settings
from photoselect.draft import evidence as ev_mod
from photoselect.draft import preference, rerank, scoring
from photoselect.store import Evidence, Recommendation, Store

log = logging.getLogger(__name__)


def _reason(row, breakdown: dict, prefs: list[tuple[str, str]]) -> str:
    """템플릿 이유 문장. LLM은 여기 안 들어온다 — 그건 C(llm/)의 일이고 이 문장이 폴백이다.

    정밀도가 낮은 태그 값(scene의 walk/prep 등)을 근거로 쓰면 거짓말이 된다(study/02 step8 [B]).
    그래서 scene은 캡션이 있으면 캡션으로 대신하고, 취향 근거는 conf가 충분한 축만 쓴다.
    """
    parts = []
    if row.caption:
        parts.append(row.caption.rstrip("."))
    else:
        s = LABELS_KO["scene"].get(row.scene, "")
        if s:
            parts.append(f"{s} 장면")
    if row.rank_reason_code == "eyes_open":
        parts.append("이 순간의 컷 중 눈을 뜬 컷")
    elif row.rank_reason_code in ("technical", "aesthetic"):
        parts.append("이 순간의 대표 컷")
    matched = [LABELS_KO[ax].get(val, "") for ax, val in prefs if getattr(row, ax) == val]
    matched = [m for m in matched if m]
    if matched:
        parts.append(f"선호하신 {'·'.join(matched)} 스타일")
    if breakdown.get("prior_z", 0) > 1.0:
        parts.append("갤러리 상위 품질")
    return " · ".join(parts) if parts else "추천"


def run(store: Store, gallery: str, settings: Settings, selection_id: str | None = None,
        round_no: int | None = None, top_k: int | None = None, target: int | None = None) -> dict:
    started = time.monotonic()
    knobs = settings.score
    target = target or knobs.target_count

    rows = store.read_analysis(gallery)
    if not rows:
        raise SystemExit(f"분석 결과가 없다: {gallery} — 먼저 analyze를 돌릴 것")
    ids, E = store.read_embeddings(gallery)
    emb_map = dict(zip(ids, E))
    rows = [r for r in rows if r.photo_id in emb_map]
    E = np.stack([emb_map[r.photo_id] for r in rows])
    X = np.array([onehot(r.tags()) for r in rows])
    id_index = {r.photo_id: i for i, r in enumerate(rows)}

    ev: Evidence = store.read_evidence(gallery, selection_id)
    prev = store.read_recommendations(gallery)
    if round_no is None:
        round_no = (max((r.round for r in prev), default=0) + 1)

    # ── 목표 장수: 담긴 수가 목표에 닿으면 끝. 아니면 남은 자리만큼만 제시 ──
    n_selected = sum(1 for p in ev.selected if p in id_index)
    remaining = target - n_selected
    if remaining <= 0:
        return {"gallery": gallery, "round": round_no, "done": True, "selected": n_selected,
                "target": target, "k": 0, "message": "셀렉 완료 — 담긴 사진이 목표 장수에 닿았다"}
    k = min(top_k or knobs.top_k, remaining)

    # ── 취향 ────────────────────────────────────────────────────────────
    pairs, n_ev, counts = ev_mod.to_pairs(ev, rows, knobs)
    pref_raw = None
    conf: dict[str, float] = {}
    prefs: list[tuple[str, str]] = []
    lam = scoring.lambda_of(n_ev, knobs)
    if pairs:
        c = np.array([id_index[a] for a, _ in pairs])
        r = np.array([id_index[b] for _, b in pairs])
        w_hat = preference.fit_bt(X[c] - X[r], knobs.bt_reg)
        conf = preference.axis_confidence(X, c, r, knobs.bt_reg, knobs.conf_prior_a)
        pref_raw = preference.preference_scores(X, w_hat, conf)
        prefs = preference.top_preferences(w_hat, conf)
    log.info("증거 %s → 가중합 %.1f → λ=%.2f · conf=%s", counts, n_ev, lam,
             {a: round(v, 2) for a, v in conf.items()})

    # ── 점수 ────────────────────────────────────────────────────────────
    prior_raw = scoring.prior(np.array([r.technical_pct for r in rows]),
                              np.array([r.aesthetic_pct for r in rows]), knobs)
    score, bd = scoring.combine(prior_raw, pref_raw, lam)

    # ── 후보 집합: 이미 담은 사진 + 이전 라운드에 제시한 사진 제외 ──────────
    # 실측(2026-08-25 6라운드)에서 "별점만 매긴 사진은 남긴다"가 매 라운드 같은 사진을
    # 다시 보여 주는 결과를 냈다. 한 번 본 사진은 담기 여부와 무관하게 다시 제시하지 않는다.
    shown = {id_index[r.photo_id] for r in prev if r.round < round_no and r.photo_id in id_index}
    exclude = {id_index[p] for p in ev.selected if p in id_index} | shown

    # ── 재랭킹 ──────────────────────────────────────────────────────────
    picked = rerank.select_with_coverage(
        score, E, [r.scene for r in rows], [r.cluster_id for r in rows],
        k, knobs.lambda_mmr, knobs.min_per_scene, knobs.scene_cap_ratio, exclude=exclude,
    )
    picked.sort(key=lambda i: -score[i])

    recs = []
    for rank, i in enumerate(picked, 1):
        row = rows[i]
        breakdown = {
            "score": round(float(score[i]), 4),
            "prior_z": round(float(bd["prior_z"][i]), 3),
            "pref_z": round(float(bd["pref_z"][i]), 3),
            "lambda": round(float(bd["lambda"]), 3),
            "technical_pct": round(row.technical_pct, 1),
            "aesthetic_pct": round(row.aesthetic_pct, 1),
            "scene": row.scene, "cluster_id": row.cluster_id,
            "rank_reason_code": row.rank_reason_code,
        }
        recs.append(Recommendation(photo_id=row.photo_id, round=round_no, rank=rank,
                                   score_breakdown=breakdown, reason=_reason(row, breakdown, prefs)))
    store.write_recommendations(gallery, recs)

    scenes = {}
    for i in picked:
        scenes[rows[i].scene] = scenes.get(rows[i].scene, 0) + 1
    return {
        "gallery": gallery, "round": round_no, "done": False, "k": len(recs),
        "selected": n_selected, "target": target, "remaining": remaining,
        "shownBefore": len(shown),
        "evidence": counts, "evidenceWeighted": round(n_ev, 1), "lambda": round(lam, 3),
        "axisConfidence": {a: round(v, 2) for a, v in conf.items()},
        "preferences": prefs, "sceneDistribution": scenes,
        "candidates": len(rows) - len(exclude), "excluded": len(exclude),
        "elapsedSeconds": round(time.monotonic() - started, 2),
    }
