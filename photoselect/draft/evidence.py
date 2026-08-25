"""취향 신호 → Bradley-Terry 쌍. 세 종류의 신호를 한 종류(쌍)로 바꾼다.

    pairs       온보딩 쌍 비교. 그대로 쓴다.                                   가중치 w_pair
    ratings     별점. 별점이 다른 두 사진 → (높은 쪽, 낮은 쪽). 같은 장면 우선.   가중치 w_rating
    selected    담은 사진 vs 안 담은 사진 → (담은, 안 담은). 같은 장면·근접 품질.  가중치 w_selection

파생 쌍(별점·선택)은 온보딩 쌍과 달리 "한 축만 다름·품질 근접" 조건이 없어 랜덤 쌍에
가깝다(study/01 step6 §3). 그래서 ① 같은 장면끼리만 짝짓고 ② 사람당 상한을 두고
③ λ의 증거 수에 더 낮은 가중치로 센다.
"""

from __future__ import annotations

import random

from photoselect.config import ScoreKnobs
from photoselect.store import Evidence, PhotoAnalysis


def to_pairs(ev: Evidence, rows: list[PhotoAnalysis], knobs: ScoreKnobs, seed: int = 0
             ) -> tuple[list[tuple[str, str]], float, dict[str, int]]:
    """(chosen, rejected) 목록, λ용 증거 가중합, 종류별 개수."""
    by_id = {r.photo_id: r for r in rows}
    rng = random.Random(seed)
    pairs: list[tuple[str, str]] = []
    counts = {"pair": 0, "rating": 0, "selection": 0}

    # ① 온보딩 쌍
    for chosen, rejected, _axis in ev.pairs:
        if chosen in by_id and rejected in by_id:
            pairs.append((chosen, rejected))
            counts["pair"] += 1

    # ② 별점 → 쌍. 같은 장면 안에서 별점이 2 이상 차이 나는 조합만.
    rated = [(pid, s) for pid, s in ev.ratings.items() if pid in by_id]
    cand = []
    for i, (a, sa) in enumerate(rated):
        for b, sb in rated[i + 1:]:
            if abs(sa - sb) < 2 or by_id[a].scene != by_id[b].scene:
                continue
            cand.append((a, b) if sa > sb else (b, a))
    rng.shuffle(cand)
    for c in cand[: knobs.max_derived_pairs]:
        pairs.append(c)
        counts["rating"] += 1

    # ③ 선택 → 쌍. 담은 사진마다 같은 장면·품질 근접(백분위 차 ≤ 20)인 미선택 사진 하나.
    sel = [pid for pid in ev.selected if pid in by_id]
    sel_set = set(sel)
    unsel = [r for r in rows if r.photo_id not in sel_set]
    cand = []
    for pid in sel:
        a = by_id[pid]
        pool = [r for r in unsel if r.scene == a.scene
                and abs(r.technical_pct - a.technical_pct) <= 20
                and abs(r.aesthetic_pct - a.aesthetic_pct) <= 20]
        if pool:
            cand.append((pid, rng.choice(pool).photo_id))
    rng.shuffle(cand)
    for c in cand[: knobs.max_derived_pairs]:
        pairs.append(c)
        counts["selection"] += 1

    n_weighted = (counts["pair"] * knobs.w_pair + counts["rating"] * knobs.w_rating
                  + counts["selection"] * knobs.w_selection)
    return pairs, n_weighted, counts
