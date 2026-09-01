"""점수식 — plan.md §3-B.

    score(p) = (1−λ)·prior(p) + λ·pref(p)
    prior    = w_t·technical_pct + w_a·aesthetic_pct      (갤러리 내 백분위)
    pref     = Σ_axis conf(axis)·w_axis·x(p)              (증거 없으면 0)
    λ        = λ_min + (λ_max−λ_min)·n/(n+k),  n = 증거 가중합

prior와 pref는 갤러리 안에서 각각 z-표준화한 뒤 섞는다 — 스케일이 다르면 λ가 의미 없다.
커버리지·MMR은 여기가 아니라 rerank.py의 일이다(점수는 한 장의 성질, 중복은 목록의 성질).
"""

from __future__ import annotations

import numpy as np

from photoselect.v1.config import ScoreKnobs


def z(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def lambda_of(n_evidence: float, k: ScoreKnobs) -> float:
    return k.lambda_min + (k.lambda_max - k.lambda_min) * n_evidence / (n_evidence + k.lambda_k)


def prior(technical_pct: np.ndarray, aesthetic_pct: np.ndarray, k: ScoreKnobs) -> np.ndarray:
    return k.w_technical * technical_pct + k.w_aesthetic * aesthetic_pct


def combine(prior_raw: np.ndarray, pref_raw: np.ndarray | None, lam: float) -> tuple[np.ndarray, dict]:
    """(score, 사진별 breakdown 재료). pref가 None이면 λ와 무관하게 prior만."""
    pz = z(prior_raw)
    if pref_raw is None or not np.any(pref_raw):
        return pz, {"prior_z": pz, "pref_z": np.zeros_like(pz), "lambda": 0.0}
    fz = z(pref_raw)
    return (1 - lam) * pz + lam * fz, {"prior_z": pz, "pref_z": fz, "lambda": lam}
