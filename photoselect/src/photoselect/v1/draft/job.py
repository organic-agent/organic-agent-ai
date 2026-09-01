"""초안 한 라운드. 진입점(`__main__ draft` / Lambda handler)이 이 함수를 부른다.

    store.read_analysis + read_embeddings        ← A의 산출물
    store.read_evidence                          ← 선택·별점·쌍 (없으면 사진학만)
    evidence → BT 쌍 → w_hat, conf(axis)         (draft/preference, draft/evidence)
    prior, pref, λ → score                       (draft/scoring)
    후보 = 전체 − 이미 담은 사진 − 이미 제시한 사진   (plan.md §3-B 후보 집합)
    제시 장수 = min(K, 목표 − 담긴 수). 담긴 수 ≥ 목표면 완료.
    커버리지 + 클러스터당 1장 + MMR → K장           (draft/rerank)
    선정 근거(slot·취향·품질·연사·유사) → 주 사유 하나 → 템플릿 문장 (LLM은 다듬기만)
    → store.write_recommendations

1단계(개인화 전): evidence가 비어 있으면 λ=λ_min, pref=0 → 사실상 사진학 + 커버리지 + MMR.
2단계: evidence.json(또는 DB)이 채워지면 같은 코드가 개인화된다. 분기가 없다.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from photoselect.v1.axes import FEATURE_INDEX, LABELS_KO, onehot
from photoselect.v1.config import AXIS_PRECISION, Settings
from photoselect.v1.draft import evidence as ev_mod
from photoselect.v1.draft import preference, rerank, scoring
from photoselect.v1.store import Evidence, Recommendation, Store

log = logging.getLogger(__name__)


#: 주 사유 우선순위 — 사진마다 가장 강한 것 하나가 문장의 주어가 된다. 셀렉터가 설득하는 순서다:
#: 본인 행동 되비추기(취향·유사) > "비슷한 N장 중 이거"(형제 대비) > 희소성 > 품질 > 앨범 자리 > 다양성 > 점수.
#: 전부 **셀 수 있는 사실**이다 — 감정 설득은 재료에 없으니 LLM 도 못 한다.
REASON_PRIORITY = ("preference", "similar", "sibling", "rarity", "quality", "burst", "coverage", "diversity", "score")

#: 희소성: 갤러리에서 이 비율 이하로 드문 (피사체·표정) 조합만 "M장뿐"이라고 말한다.
RARITY_MAX_SHARE = 0.05


def _why_not(pick, other) -> str:
    """형제 컷이 밀린 이유 — represent._reason 과 같은 문턱. 셀 수 있는 것만."""
    if other.face_boxes.get("face_count", 0) > 0 and pick.face_boxes.get("face_count", 0) > 0:
        # DB 행은 얼굴 없음이 NaN→null 로 와서 None 일 수 있다
        e_pick, e_other = pick.sub_scores.get("eyes_open") or 0.0, other.sub_scores.get("eyes_open") or 0.0
        if e_pick - e_other > 0.15:
            return "눈 감김"
    if pick.technical_pct - other.technical_pct > 5:
        return "덜 선명함"
    if pick.aesthetic_pct - other.aesthetic_pct > 5:
        return "인상이 약함"
    return "거의 같은 컷"


def _alternatives(pick, members) -> list[dict]:
    """같은 순간(클러스터)의 나머지 컷 + 각각의 탈락 사유. 프론트가 접어서 보여줄 것."""
    return [{"photo_id": o.photo_id, "why_not": _why_not(pick, o)}
            for o in sorted(members, key=lambda r: r.cluster_rank) if o.photo_id != pick.photo_id]


def _rarity(row, combo_count: dict[tuple[str, str], int], n_total: int) -> tuple[str, int] | None:
    """(조합 라벨, 장수). (피사체, 표정)이 갤러리의 RARITY_MAX_SHARE 이하일 때만."""
    subj, expr = LABELS_KO["subjects"].get(row.subjects, ""), LABELS_KO["expression"].get(row.expression, "")
    if not subj or not expr:
        return None
    m = combo_count.get((row.subjects, row.expression), 0)
    if m == 0 or m > max(1, int(RARITY_MAX_SHARE * n_total)):
        return None
    return f"{subj}의 {expr}", m


def _top_pct(pct: float) -> int:
    """백분위 → "상위 N%". 최고점(100)은 0%가 아니라 1%다."""
    return max(1, round(100 - pct))


def _facts(row, pick, breakdown: dict, prefs: list[tuple[str, str]], knobs,
           similar: tuple[str, float] | None, alternatives: list[dict], rarity: tuple[str, int] | None,
           n_total: int, album_pos: int) -> tuple[str, list[str]]:
    """(주 사유 코드, 사실 목록). 사실은 **강한 신호만** — 여기 없는 건 LLM 도 못 쓴다.

    정밀도가 낮은 태그 값(scene의 walk/prep 등)을 근거로 쓰면 거짓말이 된다(study/02 step8 [B]).
    취향 근거는 conf가 충분하고 태그 정밀도가 `reason_min_precision` 이상인 축만 쓴다(`AXIS_PRECISION`).
    품질 백분위는 상위 `reason_quality_top_pct` 안일 때만 말한다 — "상위 68%"는 근거가 아니라 약점이다.
    """
    facts: dict[str, str] = {}
    matched = [LABELS_KO[ax].get(val, "") for ax, val in prefs if getattr(row, ax) == val]
    matched = [m for m in matched if m]
    if matched:
        facts["preference"] = f"취향: 고객이 담기·비교에서 {'·'.join(matched)} 스타일을 일관되게 선호"
    if similar is not None:
        facts["similar"] = f"유사: 고객이 담은 사진 {similar[0]}와 분위기가 가장 가까움 (유사도 {similar[1]:.2f})"
    if alternatives:
        counts: dict[str, int] = {}
        for a in alternatives:
            counts[a["why_not"]] = counts.get(a["why_not"], 0) + 1
        why = " · ".join(f"{k} {v}" for k, v in counts.items())
        facts["sibling"] = f"형제: 같은 순간 {len(alternatives) + 1}장 중 이 컷. 나머지는 {why}"
    if rarity is not None:
        facts["rarity"] = f"희소: {rarity[0]} 컷은 전체 {n_total}장 중 {rarity[1]}장뿐"
    top = 100 - knobs.reason_quality_top_pct
    q = []
    if row.aesthetic_pct >= top:
        q.append(f"미학 상위 {_top_pct(row.aesthetic_pct)}%")
    if row.technical_pct >= top:
        q.append(f"기술 상위 {_top_pct(row.technical_pct)}%")
    if q:
        facts["quality"] = "품질: " + " · ".join(q) + " (갤러리 안에서의 순위)"
    if not alternatives and row.rank_reason_code == "eyes_open":      # 형제가 후보에 없을 때만 (담김·제외)
        facts["burst"] = "연사: 같은 순간을 여러 장 찍은 것 중 눈을 뜬 컷"
    scene_ko = LABELS_KO["scene"].get(pick.scene, "")
    if pick.slot in ("quota", "diversity") and scene_ko:
        facts["coverage"] = (f"앨범 자리: {scene_ko} 장면은 앨범에 {pick.scene_quota}장이면 충분함 — "
                             f"그중 {album_pos}번째")
    if pick.slot == "diversity":
        facts["diversity"] = "다양성: 앞서 고른 사진들과 구도·분위기가 겹치지 않아 끌어올림"
    if breakdown.get("prior_z", 0) > 1.0 and "quality" not in facts:
        facts["score"] = "점수: 기술·미학 종합이 갤러리 평균보다 뚜렷이 높음"
    primary = next((c for c in REASON_PRIORITY if c in facts), "score")
    return primary, [facts[c] for c in REASON_PRIORITY if c in facts]


def _template(row, primary: str, facts: list[str], prefs: list[tuple[str, str]], pick,
              alternatives: list[dict], rarity: tuple[str, int] | None, n_total: int) -> str:
    """LLM 없이도 납득되는 한 문장 — 옆에서 같이 고르는 사람 말투. 주 사유가 주어, 캡션은 사진을 가리킬 뿐."""
    scene_ko = LABELS_KO["scene"].get(row.scene, "")
    what = (row.caption.rstrip(".") if row.caption else (f"{scene_ko} 장면" if scene_ko else "이 컷"))
    if primary == "preference":
        matched = [LABELS_KO[ax].get(v, "") for ax, v in prefs if getattr(row, ax) == v]
        return f"담으신 사진들처럼 {'·'.join(m for m in matched if m)} 컷이에요 — {what}"
    if primary == "similar":
        return f"담으신 사진과 분위기가 가장 가까운 컷이에요 — {what}"
    if primary == "sibling":
        why = facts[[f.split(":")[0] for f in facts].index("형제")].split("나머지는 ", 1)[1]
        return f"비슷한 {len(alternatives) + 1}장 중 이 컷이에요, 나머지는 {why} — {what}"
    if primary == "rarity":
        return f"{rarity[0]} 컷은 전체 {n_total}장 중 {rarity[1]}장뿐이에요 — {what}"
    if primary == "quality":
        q = facts[[f.split(":")[0] for f in facts].index("품질")].split(": ", 1)[1].split(" (")[0]
        return f"{q}의 컷 — {what}"
    if primary == "burst":
        return f"같은 순간 연사 중 눈을 뜬 컷이에요 — {what}"
    if primary == "coverage":
        return f"{scene_ko} 장면은 앨범에 {pick.scene_quota}장이면 충분해요, 그중 하나 — {what}"
    if primary == "diversity":
        return f"앞서 고른 사진과 겹치지 않는 {scene_ko} 컷이에요 — {what}".replace("  ", " ")
    return what


def run(store: Store, gallery: str, settings: Settings, selection_id: str | None = None,
        round_no: int | None = None, top_k: int | None = None, target: int | None = None,
        llm=None) -> dict:
    """llm: `photoselect.llm.client.LlmClient` 또는 None. None이면 템플릿 이유 + 피드백 무시."""
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
        reliable = {ax for ax, p in AXIS_PRECISION.items() if p >= knobs.reason_min_precision}
        prefs = preference.top_preferences(w_hat, conf, allowed_axes=reliable)
    log.info("증거 %s → 가중합 %.1f → λ=%.2f · conf=%s", counts, n_ev, lam,
             {a: round(v, 2) for a, v in conf.items()})

    # ── 자연어 피드백 → 축 가중치. LLM이 없으면 무시(👍/👎·별점만 반영) ────
    nl_changes: list[tuple[str, str, float]] = []
    if llm is not None and ev.feedback:
        from photoselect.v1.llm import feedback as fb_mod
        for text in ev.feedback:
            nl_changes.extend(fb_mod.translate(llm, text, settings.llm.feedback_max_tokens))
    if nl_changes:
        nl_vec = np.zeros(X.shape[1])
        for ax, tag, delta in nl_changes:
            nl_vec[FEATURE_INDEX[f"{ax}={tag}"]] += delta
        nl_raw = X @ nl_vec
        pref_raw = nl_raw if pref_raw is None else pref_raw + nl_raw
        if lam == knobs.lambda_min and n_ev == 0:
            lam = scoring.lambda_of(len(nl_changes) * knobs.w_selection, knobs)   # 약한 증거로 센다

    # ── 점수 ────────────────────────────────────────────────────────────
    prior_raw = scoring.prior(np.array([r.technical_pct for r in rows]),
                              np.array([r.aesthetic_pct for r in rows]), knobs)
    score, bd = scoring.combine(prior_raw, pref_raw, lam)

    # ── 후보 집합: 이미 담은 사진 + 이전 라운드에 제시한 사진 제외 ──────────
    # 실측(2026-08-25 6라운드)에서 "별점만 매긴 사진은 남긴다"가 매 라운드 같은 사진을
    # 다시 보여 주는 결과를 냈다. 한 번 본 사진은 담기 여부와 무관하게 다시 제시하지 않는다.
    shown = {id_index[r.photo_id] for r in prev if r.round < round_no and r.photo_id in id_index}
    rejected = {id_index[p] for p in ev.rejected if p in id_index}   # DB 모드: rejected_at 이 찍힌 것
    exclude = {id_index[p] for p in ev.selected if p in id_index} | shown | rejected

    # ── 재랭킹 ──────────────────────────────────────────────────────────
    picks = rerank.explain_selection(
        score, E, [r.scene for r in rows], [r.cluster_id for r in rows],
        k, knobs.lambda_mmr, knobs.min_per_scene, knobs.scene_cap_ratio, exclude=exclude,
    )
    picks.sort(key=lambda p: -score[p.index])
    picked = [p.index for p in picks]

    # ── "담으신 사진과 비슷한 컷": 담은 사진 임베딩과의 최대 코사인 (2단계부터 생긴다) ──
    sel_idx = [id_index[p] for p in ev.selected if p in id_index]
    similar_of: dict[int, tuple[str, float]] = {}
    if sel_idx and picked:
        sims = E[picked] @ E[sel_idx].T
        for row_i, i in enumerate(picked):
            j = int(np.argmax(sims[row_i]))
            if sims[row_i, j] >= knobs.reason_similar_min_cos:
                similar_of[i] = (rows[sel_idx[j]].photo_id, float(sims[row_i, j]))

    # ── 관계 재료: 같은 순간의 형제(후보에 남은 것만), 희소한 조합, 장면 안에서의 순서 ──
    by_cluster: dict[int, list] = {}
    for j, r in enumerate(rows):
        if j not in exclude:
            by_cluster.setdefault(r.cluster_id, []).append(r)
    combo_count: dict[tuple[str, str], int] = {}
    for r in rows:
        combo_count[(r.subjects, r.expression)] = combo_count.get((r.subjects, r.expression), 0) + 1
    scene_seen: dict[str, int] = {}

    recs = []
    facts_of: dict[str, tuple[str, list[str]]] = {}
    for rank, pk in enumerate(picks, 1):
        i, row = pk.index, rows[pk.index]
        scene_seen[pk.scene] = scene_seen.get(pk.scene, 0) + 1
        alternatives = _alternatives(row, by_cluster.get(row.cluster_id, []))
        rarity = _rarity(row, combo_count, len(rows))
        breakdown = {
            "score": round(float(score[i]), 4),
            "prior_z": round(float(bd["prior_z"][i]), 3),
            "pref_z": round(float(bd["pref_z"][i]), 3),
            "lambda": round(float(bd["lambda"]), 3),
            "technical_pct": round(row.technical_pct, 1),
            "aesthetic_pct": round(row.aesthetic_pct, 1),
            "scene": row.scene, "cluster_id": row.cluster_id,
            "rank_reason_code": row.rank_reason_code,
            "slot": pk.slot, "scene_rank": pk.scene_rank, "scene_quota": pk.scene_quota,
        }
        similar = similar_of.get(i)
        if similar:
            breakdown["similar_to"] = similar[0]
        # 순간 단위 응답 재료 — 프론트는 카드 하나 = 순간 하나, 펼치면 형제와 탈락 사유
        breakdown["moment"] = row.caption
        breakdown["album_role"] = {"scene": pk.scene, "slot": pk.slot,
                                   # fill 은 쿼터 밖에서 채운 자리라 "N/쿼터" 가 안 맞는다 — 자리 수를 실제 장수로
                                   "quota": max(pk.scene_quota, scene_seen[pk.scene]), "position": scene_seen[pk.scene]}
        breakdown["alternatives"] = alternatives
        if rarity:
            breakdown["rarity"] = {"label": rarity[0], "count": rarity[1], "total": len(rows)}
        primary, facts = _facts(row, pk, breakdown, prefs, knobs, similar, alternatives, rarity,
                                len(rows), scene_seen[pk.scene])
        breakdown["primary_reason"] = primary
        facts_of[row.photo_id] = (primary, facts)
        recs.append(Recommendation(photo_id=row.photo_id, round=round_no, rank=rank,
                                   score_breakdown=breakdown,
                                   reason=_template(row, primary, facts, prefs, pk, alternatives, rarity, len(rows))))

    # ── 이유 문장 LLM 일괄 (선택). 템플릿은 폴백으로 항상 남는다 ──────────
    if llm is not None and recs:
        from photoselect.v1.llm import reasons as rs_mod
        by_id = {rows[i].photo_id: rows[i] for i in picked}
        items = [rs_mod.ReasonInput(
            photo_id=r.photo_id, caption=by_id[r.photo_id].caption, tags=by_id[r.photo_id].tags(),
            primary=facts_of[r.photo_id][0], facts=facts_of[r.photo_id][1], fallback=r.reason,
        ) for r in recs]
        texts = rs_mod.generate(llm, items, settings.llm.reasons_batch, settings.llm.reasons_max_tokens)
        for r in recs:
            r.reason = texts.get(r.photo_id, r.reason)
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
        "feedbackChanges": nl_changes, "llm": llm is not None,
        "candidates": len(rows) - len(exclude), "excluded": len(exclude),
        "elapsedSeconds": round(time.monotonic() - started, 2),
    }
