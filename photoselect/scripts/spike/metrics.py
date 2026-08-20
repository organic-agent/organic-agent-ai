"""평가 지표 — 이진 골든셋 라벨(작가 최종 셀렉) 기준.

MOS가 아니라 선택/미선택 이진 라벨이므로 AUC와 recall@K가 1차 지표다.
러너 점수 축 간 겹침 확인용으로만 SRCC를 쓴다.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Mann-Whitney U 기반 AUC. labels는 0/1."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    u, _ = stats.mannwhitneyu(pos, neg, alternative="greater")
    return float(u / (len(pos) * len(neg)))


def recall_at_k(scores: np.ndarray, labels: np.ndarray, k: int | None = None) -> float:
    """점수 상위 K장 중 작가 셀렉 비율. K 기본값 = 실제 셀렉 장수."""
    n_selected = int(labels.sum())
    if n_selected == 0:
        return float("nan")
    if k is None:
        k = n_selected
    top_k = np.argsort(-scores)[:k]
    return float(labels[top_k].sum() / n_selected)


def srcc(a: np.ndarray, b: np.ndarray) -> float:
    r, _ = stats.spearmanr(a, b)
    return float(r)
