"""v3 점수식 — v2 와 같다 (docs/plan-v3-folder-compare.md §0: 점수식은 바뀌지 않는다).

    prior(p)  = w_t·technical_pct + w_a·aesthetic_pct          (갤러리 내 백분위)
    score(p)  = z(prior) [+ w_bal·z(deficit(type p)) + w_pref·z(affinity(type p))]

대괄호는 4단계 — `subjects_trusted` 이고 담은 사진이 충분할 때만 붙는다. 폴더별 추천에서도
점수는 **갤러리 내 백분위 그대로** 쓴다(폴더 안 재정규화 없음, §2.1).
"""

from __future__ import annotations

import numpy as np

from photoselect.v3.config import V3Knobs


def z(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def prior(technical_pct: np.ndarray, aesthetic_pct: np.ndarray, k: V3Knobs) -> np.ndarray:
    return k.w_technical * technical_pct + k.w_aesthetic * aesthetic_pct


def type_stats(types: list[str], selected_mask: np.ndarray) -> dict[str, dict[str, float]]:
    """유형별 집계 — 갤러리 비율, 담은 비율, 부족분(deficit), 선택 lift. 전부 [-1, 1] 안의 비율."""
    n = len(types)
    n_sel = int(selected_mask.sum())
    out: dict[str, dict[str, float]] = {}
    for t in sorted(set(types)):
        idx = np.array([i for i, x in enumerate(types) if x == t])
        pool = len(idx) / n if n else 0.0
        sel = float(selected_mask[idx].sum()) / n_sel if n_sel else 0.0
        p_sel_given_t = float(selected_mask[idx].mean()) if len(idx) else 0.0
        p_sel = n_sel / n if n else 0.0
        out[t] = {"pool_share": pool, "sel_share": sel, "deficit": max(0.0, pool - sel),
                  "select_lift": p_sel_given_t - p_sel, "pool_count": float(len(idx)),
                  "sel_count": float(selected_mask[idx].sum())}
    return out


def combine(prior_raw: np.ndarray, types: list[str] | None, stats: dict | None, k: V3Knobs
            ) -> tuple[np.ndarray, dict]:
    pz = z(prior_raw)
    bd = {"prior_z": pz, "balance_z": np.zeros_like(pz), "affinity_z": np.zeros_like(pz)}
    if not types or not stats:
        return pz, bd
    deficit = np.array([stats.get(t, {}).get("deficit", 0.0) for t in types])
    lift = np.array([stats.get(t, {}).get("select_lift", 0.0) for t in types])
    bz = z(deficit) if deficit.std() > 1e-12 else np.zeros_like(pz)
    az = z(lift) if lift.std() > 1e-12 else np.zeros_like(pz)
    bd["balance_z"], bd["affinity_z"] = bz, az
    return pz + k.w_balance * bz + k.w_pref * az, bd
