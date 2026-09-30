"""2층 그룹 — 한 1층(컨셉) 안에서 같은 배경·구도로 찍은 사진 묶음. 이름은 없다, 경계만 있다.

연사(`service/burst.py`, 코사인 ≥ 0.96·순서 창)보다 훨씬 느슨하게, 순서 제약 없이 임베딩만으로 묶는다.
2026-08-29 갤러리 1 실측(822장): 평균연결 계층 클러스터 코사인 거리 0.2에서 케이크 세트·해변·소파·정원·덩굴 아치가
촬영 세트와 일치했다. 0.3부터 다른 세트가 섞인다.

컨셉 구간화(2026-09-30) 뒤로는 갤러리 전체가 아니라 **1층마다** 따로 자른다 — 2층이 1층 경계를 넘지 않는다.
예전의 과병합 가드(최소 4그룹 · 한 그룹 50% 상한)는 갤러리 전체의 다양성을 위한 것이라 뺐다. 배경 하나로 찍은 컨셉은
그룹 하나가 맞다. 과분할 가드만 남는다.
"""

from __future__ import annotations

import numpy as np


def embed_groups(emb: np.ndarray, distance: float, frag_share: float = 0.35,
                 raise_cap: float = 0.15) -> tuple[np.ndarray, float]:
    """(N,) 그룹 id(등장 순서로 0..G-1), 실제로 쓴 거리 임계. emb 는 L2 정규화돼 있어야 한다.

    과분할 — 그룹 수가 max(1, n×frag_share) 를 넘으면 0.05씩 임계를 올린다(상한 distance+raise_cap).
    """
    n = len(emb)
    if n <= 1:
        return np.zeros(n, dtype=int), distance
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # linkage 에 원시 벡터를 주면 scipy 가 pdist(cosine) 을 스칼라 C 루프로 돌려 7,189장 × 1,536차원에 37s 가 든다(#113).
    # 정규화된 벡터의 코사인 거리는 1 − X·Xᵀ 라 BLAS 행렬곱 한 번(≈0.5s)이면 같은 값이다.
    X = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    G = 1.0 - X @ X.T
    np.fill_diagonal(G, 0.0)
    np.clip(G, 0.0, 2.0, out=G)          # 부동소수 -1e-16 방지 — linkage 는 음수 거리를 거부한다
    Z = linkage(squareform(G, checks=False), method="average")
    del G

    def cut(d: float):
        labels = fcluster(Z, t=d, criterion="distance")
        return labels, len(np.unique(labels))

    d = distance
    labels, g = cut(d)
    while g > max(1, n * frag_share) and d < round(distance + raise_cap, 2):
        d = round(d + 0.05, 2)
        labels, g = cut(d)
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(int(l), len(remap)) for l in labels], dtype=int), d


def detail_groups(X: np.ndarray, concept_of_row: list[int], distance: float,
                  frag_share: float = 0.35) -> np.ndarray:
    """1층마다 embed_groups 를 돌려 갤러리 전체에서 겹치지 않는 그룹 id 를 매긴다(1층 번호 순 → 1층 안 등장 순)."""
    out = np.full(len(concept_of_row), -1, dtype=int)
    offset = 0
    for ci in sorted(set(concept_of_row)):
        idx = [i for i, c in enumerate(concept_of_row) if c == ci]
        labels, _ = embed_groups(X[idx], distance, frag_share)
        out[idx] = labels + offset
        offset += int(labels.max()) + 1
    return out


def group_profile(labels: np.ndarray) -> dict[str, float]:
    if len(labels) == 0:
        return {}
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    return {"embedGroups": float(len(sizes)), "maxSize": float(sizes.max()),
            "maxShare": float(sizes.max() / len(labels)), "singletons": float((sizes == 1).sum())}
