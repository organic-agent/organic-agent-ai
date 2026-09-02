"""v3 배치 B — 폴더별 추천 (docs/plan-v3-folder-compare.md §2).

    read_analysis + read_embeddings + read_folder_set + read_evidence
    폴더마다 f:  후보 = f.photos − 담은 사진 − 거절     (이전 라운드 노출은 제외하지 않는다 §2.4)
                 n_f = max(1, round(remaining·|f|/Σ|f'|)),  n_f ≤ ceil(|f|·0.5)
                 연사 클러스터당 1장 → MMR → n_f장, 폴더 안 점수 순위가 rank
    미분류(세트에 없는 사진)는 가상 폴더로 같은 규칙. AI 세트가 없으면 FolderSetMissing(→ 409).
    2단계 이유(§2.3 안 C): reason NULL 로 INSERT → 큰 폴더부터 reasons.generate → UPDATE.
    remaining ≤ 0 이어도 done 으로 끝내지 않는다 — 폴더마다 대표 1장은 남긴다(§2.2).

품질로 사진을 **제거하는 단계는 없다.** 점수 하한(quality_floor_*)은 근거 문장에서 품질 표현을
빼는 게이트일 뿐, 후보에서 사진을 거르지 않는다 (review-v3-design.md (6)).
"""

from __future__ import annotations

import logging
import time

import numpy as np

from photoselect_v1.config import Settings
from photoselect_v1.store import Evidence, Folder, Recommendation, Store
from photoselect_v1.recommend import reasons, rerank, scoring
from photoselect_v1.subjects import LABELS_KO as SUBJECT_KO

log = logging.getLogger(__name__)

UNFILED = "미분류"


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
    """백분위 상위면 품질 재료. 단 원점수가 절대 하한 미만이면 None — 백분위가 절대 품질을
    소거하는 문제(review-v3-design.md (6)). 소거는 근거 문장에서만, 후보에서는 아니다."""
    ts = row.sub_scores.get("technical_score")
    aes = row.sub_scores.get("aesthetic_score")
    if (ts is not None and ts < k.quality_floor_technical) or (
            aes is not None and aes < k.quality_floor_aesthetic):
        return None
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
        llm=None, analysis_job_id: int | None = None) -> dict:
    started = time.monotonic()
    k = settings.knobs

    rows = store.read_analysis(gallery)
    if not rows:
        raise SystemExit(f"분석 결과가 없다: {gallery} — 먼저 analyze 를 돌릴 것")
    ids, E = store.read_embeddings(gallery)
    emb_map = dict(zip(ids, E))
    rows = [r for r in rows if r.photo_id in emb_map]
    E = np.stack([emb_map[r.photo_id] for r in rows])
    id_index = {r.photo_id: i for i, r in enumerate(rows)}
    n = len(rows)

    folders = store.read_folder_set(gallery, analysis_job_id)   # 없으면 FolderSetMissing → 409
    ev: Evidence = store.read_evidence(gallery, selection_id)
    prev = store.read_recommendations(gallery)
    if round_no is None:
        round_no = max((r.round for r in prev), default=0) + 1
    target = target or store.target_count(gallery) or k.target_count

    # 미분류 가상 폴더 — 세트에 없는 사진(폴더 생성 뒤 업로드분)도 같은 규칙으로 (§2.1)
    in_set = {p for f in folders for p in f.photo_ids}
    unfiled = [r.photo_id for r in rows if r.photo_id not in in_set]
    if unfiled:
        folders = folders + [Folder(folder_id=None, parent_name=UNFILED, name=UNFILED,
                                    photo_ids=unfiled)]

    sel_idx = [id_index[p] for p in ev.selected if p in id_index]
    n_selected = len(sel_idx)
    remaining = target - n_selected

    # ── 점수 (갤러리 내 백분위 그대로, §2.1) ──
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

    # ── 폴더별 선택 ──
    exclude = set(sel_idx) | {id_index[p] for p in ev.rejected if p in id_index}
    sizes = {f.folder_id: len(f.photo_ids) for f in folders if f.photo_ids}
    quota = rerank.folder_quota(sizes, remaining, k.folder_cap_ratio)
    cluster_ids = [r.cluster_id for r in rows]

    by_cluster: dict[int, list] = {}
    for j, r in enumerate(rows):
        if j not in exclude:
            by_cluster.setdefault(r.cluster_id, []).append(r)

    recs: list[Recommendation] = []
    inputs: list[reasons.ReasonInput] = []
    per_folder: dict[str, int] = {}
    for f in sorted(folders, key=lambda f: -len(f.photo_ids)):     # 큰 폴더부터 — 이유 생성 순서(§2.3)
        if not f.photo_ids:
            continue
        members = [id_index[p] for p in f.photo_ids if p in id_index and id_index[p] not in exclude]
        picks = rerank.select_in_folder(score, E, members, cluster_ids, quota[f.folder_id], k.lambda_mmr)
        per_folder[f"{f.parent_name}›{f.name}"] = len(picks)
        for pk in picks:
            i, row = pk.index, rows[pk.index]
            material: dict = {"prior_z": float(bd["prior_z"][i]),
                              "folder": {"parent": f.parent_name, "name": f.name,
                                         "size": len(f.photo_ids), "rank": pk.folder_rank,
                                         "quota": pk.quota}}
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
            if pref_on and stats and row.subjects in stats and stats[row.subjects]["deficit"] > 0:
                s = stats[row.subjects]
                material["balance"] = {"label": SUBJECT_KO.get(row.subjects, row.subjects),
                                       "sel": int(s["sel_count"]), "total_sel": n_selected,
                                       "expected": int(round(s["pool_share"] * n_selected))}
            primary, facts = reasons.facts_of(material)

            breakdown = {
                "pipeline": "v3", "score": round(float(score[i]), 4),
                "prior_z": round(float(bd["prior_z"][i]), 3),
                "balance_z": round(float(bd["balance_z"][i]), 3), "affinity_z": round(float(bd["affinity_z"][i]), 3),
                "technical_pct": round(row.technical_pct, 1), "aesthetic_pct": round(row.aesthetic_pct, 1),
                "cluster_id": row.cluster_id,
                "folder": f"{f.parent_name}›{f.name}", "folder_size": len(f.photo_ids),
                "folder_rank": pk.folder_rank, "folder_quota": pk.quota,
                "alternatives": alternatives, "primary_reason": primary, "facts": facts,
            }
            if pref_on:
                breakdown["subjects"] = row.subjects
            recs.append(Recommendation(photo_id=row.photo_id, round=round_no, rank=pk.folder_rank,
                                       score_breakdown=breakdown, reason=None, folder_id=f.folder_id))
            image_path, sib_images = None, []
            if llm is not None and settings.llm.reasons_vision:
                image_path = store.preview_path(gallery, row.photo_id)
                if image_path:
                    for a in alternatives[:settings.llm.reasons_sibling_images]:
                        sp = store.preview_path(gallery, a["photo_id"])
                        if sp:
                            sib_images.append(reasons.SiblingImage(a["photo_id"], a["why_not"], sp))
            inputs.append(reasons.ReasonInput(photo_id=row.photo_id, primary=primary, facts=facts,
                                              fallback=reasons.template(primary, material),
                                              image_path=image_path, siblings=sib_images))

    # ── 1단계: reason NULL 로 즉시 적재 — 프론트는 배지를 먼저 그린다 (§2.3 안 C) ──
    store.write_recommendations(gallery, recs)

    # ── 2단계: 이유 채우기 — inputs 는 이미 큰 폴더 순. LLM 실패·부재는 템플릿 ──
    if inputs:
        if llm is not None:
            lk = settings.llm
            texts = reasons.generate(llm, inputs, lk.reasons_batch, lk.reasons_max_tokens,
                                     vision=lk.reasons_vision, image_long_edge=lk.reasons_image_long_edge)
        else:
            texts = {it.photo_id: it.fallback for it in inputs}
        store.update_reasons(gallery, round_no, texts)
        for r in recs:
            r.reason = texts.get(r.photo_id, r.reason)

    reason_dist: dict[str, int] = {}
    for r in recs:
        pr = r.score_breakdown["primary_reason"]
        reason_dist[pr] = reason_dist.get(pr, 0) + 1
    return {
        "gallery": gallery, "pipeline": "v3", "round": round_no, "done": False, "k": len(recs),
        "selected": n_selected, "target": target, "remaining": remaining,
        "folders": len([f for f in folders if f.photo_ids]), "unfiled": len(unfiled),
        "perFolder": per_folder, "reasonDistribution": reason_dist,
        "preferenceOn": pref_on, "llm": llm is not None, "reasonReady": len(inputs),
        "elapsedSeconds": round(time.monotonic() - started, 2),
    }
