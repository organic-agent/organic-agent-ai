"""A-6 근접 중복 클러스터 — 연사(burst)를 한 덩어리로 묶는다.

union-find. 두 사진이 (a) 파일명 순서로 `window` 안에 있고 (b) 임베딩 코사인이
`threshold` 이상이면 같은 클러스터. 연사는 **연속 촬영**이므로 순서 제약이 있어야
"같은 장소에서 다른 순간에 찍은 비슷한 컷"이 잘못 묶이지 않는다.

임계값은 실측으로 정한다 — `similarity_profile()`이 이웃 유사도 분포를 돌려주고
analyze가 그것을 찍어 준다. 합성 가정(01 lab)은 같은 클러스터 0.9 / 다른 클러스터 0.4였다.
"""

from __future__ import annotations

import numpy as np


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def cluster_bursts(emb: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """(N,) cluster_id. 입력 순서 = 파일명 순서여야 한다. emb는 L2 정규화돼 있어야 한다."""
    n = len(emb)
    parent = list(range(n))
    for i in range(n):
        hi = min(n, i + window + 1)
        if hi <= i + 1:
            continue
        sims = emb[i + 1:hi] @ emb[i]
        for k, s in enumerate(sims):
            if s >= threshold:
                a, b = _find(parent, i), _find(parent, i + 1 + k)
                if a != b:
                    parent[b] = a
    roots = [_find(parent, i) for i in range(n)]
    # 등장 순서대로 0,1,2,… 로 다시 번호를 매긴다
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(r, len(remap)) for r in roots], dtype=int)


def similarity_profile(emb: np.ndarray, window: int) -> dict[str, float]:
    """이웃(순서상 window 내) 쌍의 코사인 분포. 임계값을 고르는 근거."""
    n = len(emb)
    if n < 2:
        return {}
    vals = []
    for i in range(n):
        hi = min(n, i + window + 1)
        if hi > i + 1:
            vals.extend((emb[i + 1:hi] @ emb[i]).tolist())
    v = np.array(vals)
    qs = np.percentile(v, [10, 25, 50, 75, 90, 95, 99])
    return {"n_pairs": float(len(v)), "p10": qs[0], "p25": qs[1], "p50": qs[2],
            "p75": qs[3], "p90": qs[4], "p95": qs[5], "p99": qs[6]}
