"""평가 지표 — 쌍 예측 정확도(주지표)와 그 유의성, 그리고 랭킹 지표 nDCG.

설계안 §6의 지표를 실습 크기로 구현한 것. 스파이크 하네스
(`photoselect/scripts/spike/metrics.py`)의 AUC·recall@K는 08-22판에서 폐기됐고,
주지표가 여기 있는 pairwise_accuracy로 바뀌었다.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats


# ── 주지표: 홀드아웃 쌍 예측 정확도 ─────────────────────────────────────────
def pairwise_accuracy(
    w: np.ndarray, X: np.ndarray, chosen: np.ndarray, rejected: np.ndarray
) -> float:
    """학습된 취향 벡터 w가 홀드아웃 쌍의 선택을 맞히는 비율. 우연 = 0.5."""
    if len(chosen) == 0:
        return float("nan")
    margin = (X[chosen] - X[rejected]) @ w
    return float((margin > 0).mean())


def wilson_interval(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """비율의 신뢰구간 (Wilson). 표본이 작을 때 정규근사보다 정직하다.

    24표본에서 정확도 62%면 구간이 0.5를 포함하는지 여기서 바로 보인다.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (center - half, center + half)


def binom_p_value(k: int, n: int, p0: float = 0.5) -> float:
    """'우연(50%)보다 잘한다'의 단측 이항검정 p값. n이 작을 때 정규근사 쓰지 말 것."""
    if n == 0:
        return float("nan")
    return float(stats.binomtest(k, n, p0, alternative="greater").pvalue)


def required_pairs(p_true: float, p0: float = 0.5, alpha: float = 0.05, power: float = 0.8) -> int:
    """진짜 정확도가 p_true일 때, 우연과 구분하려면 홀드아웃 쌍이 몇 개 필요한가.

    단측 검정 정규근사. W1~2 실험 설계('몇 명 × 몇 쌍을 받아야 하나')의 계산 근거.
    """
    if p_true <= p0:
        return -1
    z_a = stats.norm.ppf(1 - alpha)
    z_b = stats.norm.ppf(power)
    num = z_a * math.sqrt(p0 * (1 - p0)) + z_b * math.sqrt(p_true * (1 - p_true))
    return int(math.ceil((num / (p_true - p0)) ** 2))


# ── 랭킹 지표 ───────────────────────────────────────────────────────────────
def dcg(gains: np.ndarray, k: int | None = None) -> float:
    """DCG@k. 순위가 낮을수록 log2(rank+1)로 할인. gains는 이미 추천 순서대로."""
    g = np.asarray(gains, dtype=float)[: k or len(gains)]
    discounts = np.log2(np.arange(2, len(g) + 2))
    return float((g / discounts).sum())


def ndcg(gains: np.ndarray, k: int | None = None) -> float:
    """nDCG@k = DCG / 이상적 순서의 DCG. 0~1. '설득력은 순서에 있다'의 계량."""
    ideal = np.sort(np.asarray(gains, dtype=float))[::-1]
    denom = dcg(ideal, k)
    return float(dcg(gains, k) / denom) if denom > 0 else float("nan")


def accept_rate(accepted: np.ndarray) -> float:
    """수락률 @round — 제시한 것 중 고객이 담은 비율."""
    return float(np.mean(accepted)) if len(accepted) else float("nan")


def regret_rate(accepted: np.ndarray, later_unselected: np.ndarray) -> float:
    """후회율 — 담았다가 뺀 비율. '전체 수락' 부풀림을 잡는 지표."""
    n_acc = int(accepted.sum())
    if n_acc == 0:
        return float("nan")
    return float((accepted & later_unselected).sum() / n_acc)
