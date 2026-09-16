"""수치 최적화 — 두 강도 L2 가중 로지스틱 회귀를 scipy L-BFGS 로 푼다. scipy.optimize 는 여기서만 쓴다.

라벨·표본 가중치를 어떻게 만들지는 `service/train.py` 의 일이고, 여기는 (X, y, w) 가 주어졌을 때 θ 를 찾는 것뿐이다.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import minimize

from preference.config.settings import Knobs

log = logging.getLogger(__name__)


def _block_scale(x: np.ndarray) -> float:
    """좌표당 평균 분산 = trace(Cov)/d. 0 이면 1(상수 블록)."""
    v = float(x.var(axis=0).mean())
    return v if v > 1e-12 else 1.0


def fit(xs: np.ndarray, xe: np.ndarray, y: np.ndarray, w: np.ndarray, knobs: Knobs) -> tuple[np.ndarray, np.ndarray, float]:
    """가중 로지스틱 손실 + λ_s‖w_s‖² + λ_e‖w_e‖². bias 는 정규화하지 않는다."""
    ns, ne = xs.shape[1], xe.shape[1]
    X = np.concatenate([xs, xe, np.ones((len(y), 1))], axis=1)
    # 정규화 강도를 블록의 분산 규모로 보정한다 — 임의 방향의 로짓 표준편차가 1 이 되는 ‖w‖ 를 기준 단위로.
    # 임베딩 1536차원은 좌표당 분산이 1/1536 수준이라 보정 없이 l2·‖w‖² 을 쓰면 값의 뜻이 차원 수에 묻힌다.
    reg = np.concatenate([np.full(ns, knobs.l2_scalar * _block_scale(xs)),
                          np.full(ne, knobs.l2_emb * _block_scale(xe)), [0.0]])
    wsum = w.sum()

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        m = X @ theta
        # log(1+e^m) − y·m, 수치 안정
        loss = np.logaddexp(0.0, m) - y * m
        p = 1.0 / (1.0 + np.exp(-m))
        total = (w * loss).sum() / wsum + (reg * theta * theta).sum()
        grad = X.T @ (w * (p - y)) / wsum + 2.0 * reg * theta
        return float(total), grad

    res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": knobs.max_iter})
    if not res.success:
        log.warning("L-BFGS 가 수렴 조건을 못 채웠다: %s (iter %d)", res.message, res.nit)
    theta = res.x
    return theta[:ns], theta[ns:ns + ne], float(theta[-1])
