"""v2 배치 B — 초안 한 라운드.

    read_analysis + read_embeddings + read_evidence + read_recommendations
    남은 자리 = target − 담은 수 (0 이면 완료)
    score = z(0.5·tech_pct + 0.5·aes_pct)  [+ 균형·선호, 4단계]
    후보 = 전체 − 담은 사진 − 이전 라운드에 보여준 사진 − 거절
    컨셉 그룹 쿼터 → 연사 클러스터당 1장 → MMR → k장
    근거: balance › quality › sibling › concept › score › diversity  → 템플릿 → (LLM 다듬기)
    → write_recommendations

품질로 사진을 **제거하는 단계는 없다.** 점수가 낮아도 컨셉 쿼터·MMR 로 뽑힐 수 있고, 안 뽑힌 사진은
다음 라운드 후보로 남는다.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from photoselect.v2.config import Settings
from photoselect.v2.store import Evidence, Recommendation, Store
from photoselect.v2 import reasons, rerank, scoring
from photoselect.v2.subjects import LABELS_KO as SUBJECT_KO

log = logging.getLogger(__name__)


def _why_not(pick, other) -> str:
    ps, os_ = pick.sub_scores.get("sharpness") or 0.0, other.sub_scores.get("sharpness") or 0.0
    if ps > 1.25 * os_ and ps > 0:
        return "덜 선명함"
    if pick.technical_pct - other.technical_pct > 5:
        return "화질이 떨어짐"
    if pick.aesthetic_pct - other.aesthetic_pct > 5:
        return "인상이 약함"
    return "거의 같은 컷"


def _quality_material(row, k) -> dict | None:
    top = 100 - k.reason_quality_top_pct
    m: dict = {}
    if row.aesthetic_pct >= top:
        m["aesthetic_top"] = reasons.top_pct(row.aesthetic_pct)
    if row.technical_pct >= top:
        m["technical_top"] = reasons.top_pct(row.technical_pct)
    if not m:
        return None
    desc = []
    if row.sub_scores.get("sharpness_pct", 0) >= top:
        desc.append("초점이 또렷함")
    if row.sub_scores.get("highlight_clip", 1.0) < 0.01 and row.sub_scores.get("shadow_clip", 1.0) < 0.02:
        desc.append("노출이 안정적")
    m["descriptors"] = desc
    return m


def run(store: Store, gallery: str, settings: Settings, selection_id: str | None = None,
        round_no: int | None = None, top_k: int | None = None, target: int | None = None,
        llm=None) -> dict:
    started = time.monotonic()
    k = settings.v2
    target = target or k.target_count

    rows = store.read_analysis(gallery)
    if not rows:
        raise SystemExit(f"분석 결과가 없다: {gallery} — 먼저 analyze 를 돌릴 것")
    ids, E = store.read_embeddings(gallery)
    emb_map = dict(zip(ids, E))
    rows = [r for r in rows if r.photo_id in emb_map]
    E = np.stack([emb_map[r.photo_id] for r in rows])
    id_index = {r.photo_id: i for i, r in enumerate(rows)}
    n = len(rows)

    ev: Evidence = store.read_evidence(gallery, selection_id)
    prev = store.read_recommendations(gallery)
    if round_no is None:
        round_no = max((r.round for r in prev), default=0) + 1

    sel_idx = [id_index[p] for p in ev.selected if p in id_index]
    n_selected = len(sel_idx)
    remaining = target - n_selected
    if remaining <= 0:
        return {"gallery": gallery, "pipeline": "v2", "round": round_no, "done": True,
                "selected": n_selected, "target": target, "k": 0, "message": "셀렉 완료 — 담긴 사진이 목표 장수에 닿았다"}
    kk = min(top_k or k.top_k, remaining)

    # ── 점수 ──
    prior_raw = scoring.prior(np.array([r.technical_pct for r in rows]),
                              np.array([r.aesthetic_pct for r in rows]), k)
    types = [r.subjects for r in rows]
    stats = None
    pref_on = k.subjects_trusted and n_selected >= k.pref_min_selected and any(t != "unknown" for t in types)
    if pref_on:
        mask = np.zeros(n, dtype=bool)
        mask[sel_idx] = True
        stats = scoring.type_stats(types, mask)
    score, bd = scoring.combine(prior_raw, types if pref_on else None, stats, k)

    # ── 후보 ──
    shown = {id_index[r.photo_id] for r in prev if r.round < round_no and r.photo_id in id_index}
    rejected = {id_index[p] for p in ev.rejected if p in id_index}
    exclude = set(sel_idx) | shown | rejected

    # ── 재랭킹 ──
    groups = [r.concept_id for r in rows]
    picks = rerank.explain_selection(score, E, groups, [r.cluster_id for r in rows], kk,
                                     k.lambda_mmr, k.min_per_group, k.group_cap_ratio, exclude=exclude,
                                     min_group_size=k.reason_concept_min_size)
    picks.sort(key=lambda p: -score[p.index])

    # ── 관계 재료 ──
    by_cluster: dict[int, list] = {}
    by_group: dict[int, list[int]] = {}
    for j, r in enumerate(rows):
        by_group.setdefault(r.concept_id, []).append(j)
        if j not in exclude:
            by_cluster.setdefault(r.cluster_id, []).append(r)
    group_best = {g: max(members, key=lambda j: score[j]) for g, members in by_group.items()}

    recs: list[Recommendation] = []
    inputs: list[reasons.ReasonInput] = []
    for rank, pk in enumerate(picks, 1):
        i, row = pk.index, rows[pk.index]
        material: dict = {"prior_z": float(bd["prior_z"][i])}
        siblings = [o for o in by_cluster.get(row.cluster_id, []) if o.photo_id != row.photo_id]
        alternatives = [{"photo_id": o.photo_id, "why_not": _why_not(row, o)}
                        for o in sorted(siblings, key=lambda r: r.cluster_rank)]
        if alternatives:
            why_counts: dict[str, int] = {}
            for a in alternatives:
                why_counts[a["why_not"]] = why_counts.get(a["why_not"], 0) + 1
            sharpest = all(a["why_not"] == "덜 선명함" for a in alternatives) or (
                row.sub_scores.get("sharpness", 0) >= max(o.sub_scores.get("sharpness", 0) for o in siblings))
            material["sibling"] = {"n": len(alternatives) + 1, "why_counts": why_counts, "sharpest": sharpest}
        q = _quality_material(row, k)
        if q:
            material["quality"] = q
        gsize = len(by_group.get(row.concept_id, []))
        if gsize >= k.reason_concept_min_size and group_best.get(row.concept_id) == i:
            material["concept"] = {"size": gsize}
        if pref_on and stats and row.subjects in stats and stats[row.subjects]["deficit"] > 0:
            s = stats[row.subjects]
            material["balance"] = {"label": SUBJECT_KO.get(row.subjects, row.subjects),
                                   "sel": int(s["sel_count"]), "total_sel": n_selected,
                                   "expected": int(round(s["pool_share"] * n_selected))}
        primary, facts = reasons.facts_of(material)
        text = reasons.template(primary, material)

        breakdown = {
            "pipeline": "v2", "score": round(float(score[i]), 4),
            "prior_z": round(float(bd["prior_z"][i]), 3),
            "balance_z": round(float(bd["balance_z"][i]), 3), "affinity_z": round(float(bd["affinity_z"][i]), 3),
            "technical_pct": round(row.technical_pct, 1), "aesthetic_pct": round(row.aesthetic_pct, 1),
            "sharpness_pct": round(float(row.sub_scores.get("sharpness_pct", 50.0)), 1),
            "cluster_id": row.cluster_id, "rank_reason_code": row.rank_reason_code,
            "concept_id": row.concept_id, "concept_size": gsize,
            "slot": pk.slot, "group_rank": pk.group_rank, "group_quota": pk.group_quota,
            "alternatives": alternatives, "primary_reason": primary, "facts": facts,
        }
        if pref_on:
            breakdown["subjects"] = row.subjects
        recs.append(Recommendation(photo_id=row.photo_id, round=round_no, rank=rank,
                                   score_breakdown=breakdown, reason=text))
        inputs.append(reasons.ReasonInput(photo_id=row.photo_id, primary=primary, facts=facts, fallback=text))

    if llm is not None and recs:
        texts = reasons.generate(llm, inputs, settings.llm.reasons_batch, settings.llm.reasons_max_tokens)
        for r in recs:
            r.reason = texts.get(r.photo_id, r.reason)
    store.write_recommendations(gallery, recs)

    group_dist: dict[str, int] = {}
    reason_dist: dict[str, int] = {}
    for r in recs:
        g = str(r.score_breakdown["concept_id"])
        group_dist[g] = group_dist.get(g, 0) + 1
        pr = r.score_breakdown["primary_reason"]
        reason_dist[pr] = reason_dist.get(pr, 0) + 1
    return {
        "gallery": gallery, "pipeline": "v2", "round": round_no, "done": False, "k": len(recs),
        "selected": n_selected, "target": target, "remaining": remaining, "shownBefore": len(shown),
        "preferenceOn": pref_on, "typeStats": stats or {},
        "conceptDistribution": group_dist, "reasonDistribution": reason_dist, "llm": llm is not None,
        "candidates": n - len(exclude), "excluded": len(exclude),
        "elapsedSeconds": round(time.monotonic() - started, 2),
    }
