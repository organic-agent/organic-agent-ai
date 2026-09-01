"""재랭킹 — 점수는 한 장의 성질, 중복·누락은 목록의 성질.

v1 `draft/rerank.py`와 같은 알고리즘이다. 차이는 쿼터 단위가 VLM 장면이 아니라 **컨셉 그룹**이라는 것.

    coverage_quota     그룹마다 최소 장수 + 비례 배분 + 상한(cap)
    mmr_select         점수 높고 서로 안 닮은 q장 (그리디)
    explain_selection  쿼터 → 연사 클러스터당 1장 → 그룹 안에서 MMR → 남은 자리 점수순, slot 과 함께

slot: quota(그룹 쿼터 안, 점수순으로도 뽑혔을 컷) · diversity(MMR 이 끌어올린 컷) · fill(남은 자리)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def mmr_select(score: np.ndarray, emb: np.ndarray, k: int, lam: float,
               candidates: np.ndarray | None = None) -> list[int]:
    pool = np.arange(len(score)) if candidates is None else np.asarray(candidates, dtype=int)
    if len(pool) == 0 or k <= 0:
        return []
    selected: list[int] = []
    max_sim = np.full(len(pool), -np.inf)
    alive = np.ones(len(pool), dtype=bool)
    for _ in range(min(k, len(pool))):
        penalty = np.where(np.isfinite(max_sim), max_sim, 0.0)
        mmr = lam * score[pool] - (1 - lam) * penalty
        mmr[~alive] = -np.inf
        pick = int(np.argmax(mmr))
        selected.append(int(pool[pick]))
        alive[pick] = False
        max_sim = np.maximum(max_sim, emb[pool] @ emb[pool[pick]])
    return selected


def coverage_quota(groups: list[int], k: int, min_per_group: int, cap_ratio: float | None,
                   min_group_size: int = 2) -> dict[int, int]:
    """그룹별 쿼터. 크기 < min_group_size 인 그룹(외톨이 사진)은 쿼터를 안 받는다 — fill 로만 뽑힌다.
    그룹이 k 보다 많으면 큰 그룹부터 k 개만 최소 1장을 받는다(합이 k 를 넘으면 뒤 그룹이 잘려 편향된다)."""
    counts_all = {g: groups.count(g) for g in set(groups)}
    present = sorted((g for g in counts_all if counts_all[g] >= min_group_size), key=lambda g: (-counts_all[g], g))
    if not present:
        return {}
    counts = {g: counts_all[g] for g in present}
    eligible = present[:max(1, k // max(1, min_per_group))]
    quota = {g: (min(min_per_group, counts[g]) if g in eligible else 0) for g in present}
    remaining = k - sum(quota.values())
    if remaining > 0:
        total = sum(counts.values())
        share = {g: counts[g] / total * remaining for g in present}
        base = {g: int(np.floor(share[g])) for g in present}
        for g in present:
            quota[g] += base[g]
        left = remaining - sum(base.values())
        for g in sorted(present, key=lambda g: -(share[g] - base[g]))[:max(0, left)]:
            quota[g] += 1
    if cap_ratio is not None:
        cap = max(min_per_group, int(round(cap_ratio * k)))
        overflow = 0
        for g in present:
            if quota[g] > cap:
                overflow += quota[g] - cap
                quota[g] = cap
        for g in sorted(present, key=lambda g: -counts[g]):
            if overflow <= 0:
                break
            room = min(cap - quota[g], counts[g] - quota[g])
            if room > 0:
                give = min(room, overflow)
                quota[g] += give
                overflow -= give
    for g in present:
        quota[g] = min(quota[g], counts[g])
    return quota


@dataclass
class Pick:
    index: int
    slot: str            # quota | diversity | fill
    group: int           # concept_id
    group_rank: int      # 그 그룹 후보(연사 dedup 후) 안의 점수 순위, 1부터. fill 은 0
    group_quota: int


def explain_selection(score: np.ndarray, emb: np.ndarray, groups: list[int], cluster_ids: list[int],
                      k: int, lam_mmr: float, min_per_group: int, cap_ratio: float | None,
                      exclude: set[int] | None = None, min_group_size: int = 2) -> list[Pick]:
    exclude = exclude or set()
    avail = [i for i in range(len(score)) if i not in exclude]
    if not avail or k <= 0:
        return []
    quota = coverage_quota([groups[i] for i in avail], k, min_per_group, cap_ratio, min_group_size)
    chosen: list[Pick] = []
    used_clusters: set[int] = set()

    def dedup(cand: list[int]) -> list[int]:
        best: dict[int, int] = {}
        for i in cand:
            c = cluster_ids[i]
            if c in used_clusters:
                continue
            if c not in best or score[i] > score[best[c]]:
                best[c] = i
        return list(best.values())

    for g, q in quota.items():
        if q <= 0:
            continue
        cand = dedup([i for i in avail if groups[i] == g])
        picked = mmr_select(score, emb, q, lam_mmr, candidates=np.array(cand)) if cand else []
        by_score = {i: r for r, i in enumerate(sorted(cand, key=lambda i: -score[i]), 1)}
        for i in picked:
            rank = by_score[i]
            chosen.append(Pick(i, "quota" if rank <= q else "diversity", g, rank, q))
        used_clusters.update(cluster_ids[i] for i in picked)

    if len(chosen) < k:
        taken = {p.index for p in chosen}
        avail_set = set(avail)
        rest = [int(i) for i in np.argsort(-score)
                if i in avail_set and i not in taken and cluster_ids[i] not in used_clusters]
        for i in rest[: k - len(chosen)]:
            chosen.append(Pick(i, "fill", groups[i], 0, quota.get(groups[i], 0)))
    return chosen[:k]
