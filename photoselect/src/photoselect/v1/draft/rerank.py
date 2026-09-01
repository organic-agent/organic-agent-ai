"""재랭킹 — 점수는 한 장의 성질이고, 중복·누락은 목록의 성질이다 (study/01 step4).

    coverage_quota        장면마다 최소 장수 + 비례 배분 + **상한**
    mmr_select            점수 높고 서로 안 닮은 K장 (그리디, O(K·N))
    select_with_coverage  쿼터 → 클러스터당 1장 → 장면 안에서 MMR → 남은 자리 점수순
    explain_selection     위와 같되 사진마다 **왜 뽑혔는지**(slot)를 함께 돌려준다 — 이유 문장 재료

slot 은 셋 중 하나다:
    quota      장면 쿼터 안에서 점수 순으로도 뽑혔을 컷 (scene_rank ≤ 쿼터)
    diversity  점수만으로는 쿼터 밖이었는데 MMR 이 "이미 고른 것과 안 닮아서" 끌어올린 컷
    fill       쿼터를 다 채운 뒤 남은 자리를 전체 점수 순으로 채운 컷

상한(scene_cap)은 01 lab에 없던 것이다. 08-25 실측 분포(snap 68%)에서 비례 배분이
쏠림을 재생산하는 것을 보고 넣었다 — study/01 step4 [D]③.
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


def coverage_quota(scenes: list[str], k: int, min_per_scene: int, cap_ratio: float | None) -> dict[str, int]:
    """장면별 쿼터. 존재하는 장면은 최소 min_per_scene, 나머지는 비례, 단 한 장면 ≤ cap_ratio·k."""
    present = sorted(set(scenes))
    counts = np.array([scenes.count(s) for s in present], dtype=float)
    quota = {s: min(min_per_scene, scenes.count(s)) for s in present}
    remaining = k - sum(quota.values())
    if remaining > 0:
        share = counts / counts.sum() * remaining
        base = np.floor(share).astype(int)
        for s, b in zip(present, base):
            quota[s] += int(b)
        left = remaining - int(base.sum())
        for s in [present[i] for i in np.argsort(-(share - base))][:left]:
            quota[s] += 1
    if cap_ratio is not None:
        cap = max(min_per_scene, int(round(cap_ratio * k)))
        overflow = 0
        for s in present:
            if quota[s] > cap:
                overflow += quota[s] - cap
                quota[s] = cap
        # 넘친 자리는 상한 미만인 장면에 크기순으로 나눠 준다
        for s in sorted(present, key=lambda s: -scenes.count(s)):
            if overflow <= 0:
                break
            room = min(cap - quota[s], scenes.count(s) - quota[s])
            if room > 0:
                give = min(room, overflow)
                quota[s] += give
                overflow -= give
    for s in present:
        quota[s] = min(quota[s], scenes.count(s))
    return quota


@dataclass
class Pick:
    index: int
    slot: str            # quota | diversity | fill
    scene: str
    scene_rank: int      # 그 장면 후보(클러스터 dedup 후) 안에서 점수 순위, 1부터
    scene_quota: int     # 그 장면에 배정된 자리 수


def explain_selection(score: np.ndarray, emb: np.ndarray, scenes: list[str], cluster_ids: list[int],
                      k: int, lam_mmr: float, min_per_scene: int, cap_ratio: float | None,
                      exclude: set[int] | None = None) -> list[Pick]:
    """초안 K장 + 선정 근거. exclude는 후보에서 뺄 인덱스(이미 담은 사진 — plan.md §3-B 후보 집합)."""
    exclude = exclude or set()
    n = len(score)
    avail = [i for i in range(n) if i not in exclude]
    if not avail:
        return []
    quota = coverage_quota([scenes[i] for i in avail], k, min_per_scene, cap_ratio)

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

    for scene, q in quota.items():
        if q <= 0:
            continue
        cand = dedup([i for i in avail if scenes[i] == scene])
        picked = mmr_select(score, emb, q, lam_mmr, candidates=np.array(cand)) if cand else []
        by_score = {i: r for r, i in enumerate(sorted(cand, key=lambda i: -score[i]), 1)}
        for i in picked:
            rank = by_score[i]
            chosen.append(Pick(i, "quota" if rank <= q else "diversity", scene, rank, q))
        used_clusters.update(cluster_ids[i] for i in picked)

    if len(chosen) < k:
        taken = {p.index for p in chosen}
        rest = [int(i) for i in np.argsort(-score) if i in set(avail) and i not in taken
                and cluster_ids[i] not in used_clusters]
        for i in rest[: k - len(chosen)]:
            sc = scenes[i]
            chosen.append(Pick(i, "fill", sc, 0, quota.get(sc, 0)))
    return chosen[:k]


def select_with_coverage(score: np.ndarray, emb: np.ndarray, scenes: list[str], cluster_ids: list[int],
                         k: int, lam_mmr: float, min_per_scene: int, cap_ratio: float | None,
                         exclude: set[int] | None = None) -> list[int]:
    """`explain_selection`의 인덱스만. 근거가 필요 없는 호출자(테스트·스크립트)용."""
    return [p.index for p in explain_selection(score, emb, scenes, cluster_ids, k, lam_mmr,
                                               min_per_scene, cap_ratio, exclude)]
