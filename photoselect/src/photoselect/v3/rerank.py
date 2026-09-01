"""v3 재랭킹 — 추천 단위가 갤러리가 아니라 **자식 폴더**다 (docs/plan-v3-folder-compare.md §2.1).

    folder_quota      폴더당 n_f = max(1, round(target·|f|/Σ|f'|)),  단 n_f ≤ ceil(|f|·cap_ratio)
    select_in_folder  폴더 안에서 연사 클러스터당 1장 → MMR(0.7·score − 0.3·maxCos) → n장

v2 rerank 와 달리 그룹 간 쿼터 경쟁(coverage_quota)이 없다 — 폴더마다 독립이라 배분식이
훨씬 단순하고, "폴더 화면에 이 폴더의 추천 n장"이라는 상태 그대로다.
"""

from __future__ import annotations

import math
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


def folder_quota(sizes: dict, target: int, cap_ratio: float) -> dict:
    """폴더별 목표 장수 n_f. target 비례·폴더당 최소 1·폴더 절반 상한 (§2.1·§2.2).

    target ≤ 0 (담은 사진이 목표에 닿은 뒤의 refine)이어도 폴더마다 1장은 남긴다 —
    폴더 화면에서 AI 마크가 사라지는 것이 더 이상하다. cap 은 |f|=1 이어도 최소 1."""
    total = sum(sizes.values())
    if total == 0:
        return {f: 0 for f in sizes}
    out = {}
    for f, n in sizes.items():
        n_f = max(1, round(max(target, 0) * n / total))
        out[f] = min(n_f, max(1, math.ceil(n * cap_ratio)))
    return out


@dataclass
class Pick:
    index: int           # ordered-row 인덱스
    folder_rank: int     # 폴더 안 점수 순위 (연사 dedup 후), 1부터
    quota: int           # 그 폴더의 n_f


def select_in_folder(score: np.ndarray, emb: np.ndarray, members: list[int],
                     cluster_ids: list[int], n: int, lam: float) -> list[Pick]:
    """폴더 안 선택 — 연사 클러스터당 1장 남기고 MMR 로 n장. 반환은 폴더 안 점수 순위와 함께."""
    if not members or n <= 0:
        return []
    best: dict[int, int] = {}
    for i in members:
        c = cluster_ids[i]
        if c not in best or score[i] > score[best[c]]:
            best[c] = i
    cand = list(best.values())
    picked = mmr_select(score, emb, n, lam, candidates=np.array(cand))
    by_score = {i: r for r, i in enumerate(sorted(cand, key=lambda i: -score[i]), 1)}
    picks = [Pick(index=i, folder_rank=by_score[i], quota=n) for i in picked]
    picks.sort(key=lambda p: p.folder_rank)
    return picks
