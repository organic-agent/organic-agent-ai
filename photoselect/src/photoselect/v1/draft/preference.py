"""취향 학습 — Bradley-Terry(차이 벡터 로지스틱 회귀)와 축별 신뢰도 conf(axis).

study/01 step1·step3에서 합성 데이터로 검증한 코드를 옮겼다. 달라진 점 하나:
**conf(axis)를 leave-one-out으로 잰다.** study/02 step3 [B]가 밝힌 대로 학습 쌍에서 그대로
재면 낙관 편향이 크다(승률 0.86 vs LOO 0.55) — 증거가 적은 축일수록 심하다.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from photoselect.v1.axes import AXES, FEATURE_INDEX


def _nll_and_grad(w: np.ndarray, d: np.ndarray, reg: float) -> tuple[float, np.ndarray]:
    z = d @ w
    nll = np.logaddexp(0.0, -z).sum() + 0.5 * reg * w @ w
    sig = 1.0 / (1.0 + np.exp(-z))
    grad = -((1.0 - sig)[:, None] * d).sum(axis=0) + reg * w
    return float(nll), grad


def fit_bt(d: np.ndarray, reg: float) -> np.ndarray:
    """d: (K, D) chosen − rejected 차이 벡터. 절편 없는 L2 로지스틱 회귀의 MAP."""
    if len(d) == 0:
        return np.zeros(d.shape[1] if d.ndim == 2 else 0)
    res = minimize(_nll_and_grad, np.zeros(d.shape[1]), args=(d, reg), jac=True, method="L-BFGS-B")
    return res.x


def axis_confidence(
    X: np.ndarray, chosen: np.ndarray, rejected: np.ndarray, reg: float, prior_a: float,
    loo: bool = True,
) -> dict[str, float]:
    """축별 conf ∈ [0,1]. 그 축이 갈린 쌍에서 추정 방향이 맞은 비율을 베타-이항으로 누른 값.

    loo=True면 쌍마다 그 쌍을 뺀 모델로 방향을 판정한다(기본). 쌍이 많으면 K번 적합하므로
    상한(60)을 두고 그 이상이면 in-sample로 떨어진다 — 그 규모면 편향이 작다.
    """
    d_all = X[chosen] - X[rejected]
    K = len(d_all)
    if K == 0:
        return {ax: 0.0 for ax in AXES}
    use_loo = loo and K <= 60
    w_full = fit_bt(d_all, reg)
    w_minus = [fit_bt(np.delete(d_all, t, axis=0), reg) for t in range(K)] if use_loo else None

    conf = {}
    for ax, vals in AXES.items():
        cols = [FEATURE_INDEX[f"{ax}={v}"] for v in vals]
        n_ax = wins = 0
        for t in range(K):
            dc = d_all[t, cols]
            if not dc.any():          # 이 쌍은 이 축이 안 갈렸다
                continue
            w = (w_minus[t] if use_loo else w_full)[cols]
            n_ax += 1
            wins += int(dc @ w > 0)
        p_hat = (wins + prior_a) / (n_ax + 2 * prior_a) if n_ax else 0.5
        conf[ax] = max(0.0, 2 * p_hat - 1)
    return conf


def preference_scores(X: np.ndarray, w_hat: np.ndarray, conf: dict[str, float]) -> np.ndarray:
    """pref(p) = Σ_axis conf(axis)·(축 내부 중심화한 w)·x. 축 내부 상대값만 식별되므로 중심화."""
    w = w_hat.copy()
    for ax, vals in AXES.items():
        cols = [FEATURE_INDEX[f"{ax}={v}"] for v in vals]
        w[cols] = (w[cols] - w[cols].mean()) * conf.get(ax, 0.0)
    return X @ w


def top_preferences(w_hat: np.ndarray, conf: dict[str, float], min_conf: float = 0.4,
                    allowed_axes: set[str] | None = None) -> list[tuple[str, str]]:
    """이유 문장에 써도 되는 (축, 값). 두 관문을 다 넘어야 한다 —
    ① 그 축에 의견이 있다(conf ≥ min_conf, study/01 step2 §5)
    ② 그 축의 태그를 믿을 수 있다(allowed_axes, 정밀도 기준 — study/02 step8 [B])
    """
    out = []
    for ax, vals in AXES.items():
        if conf.get(ax, 0.0) < min_conf:
            continue
        if allowed_axes is not None and ax not in allowed_axes:
            continue
        cols = [FEATURE_INDEX[f"{ax}={v}"] for v in vals]
        best = vals[int(np.argmax(w_hat[cols] - w_hat[cols].mean()))]
        if best not in ("none", "unknown", "eyes_closed"):
            out.append((ax, best))
    return out
