"""컨셉 그룹 — 같은 배경·구도로 찍은 사진 묶음. 이름은 없다, 경계만 있다.

VLM scene 태그의 자리를 대신한다. 연사 클러스터(`service/cluster.py`, 코사인 ≥ 0.96·순서 창)보다
훨씬 느슨하게, 순서 제약 없이 임베딩만으로 묶는다. 2026-08-29 갤러리 1 실측(DINOv2, 822장):
평균연결 계층 클러스터 코사인 거리 0.2에서 70그룹 — 케이크 세트 50·해변 41·소파 39·정원 37·
덩굴 아치 36장으로 촬영 세트와 일치했다. 0.3부터 다른 세트가 섞인다.

갤러리마다 분포가 달라 그룹 수가 너무 적거나(전부 한 덩어리) 한 그룹이 절반을 넘으면 임계를
낮춰 다시 자른다 — 다양성 쿼터가 의미를 잃지 않게.
"""

from __future__ import annotations

import numpy as np


def concept_groups(emb: np.ndarray, distance: float, min_groups: int = 4,
                   max_share: float = 0.5, frag_share: float = 0.35,
                   raise_cap: float = 0.15) -> tuple[np.ndarray, float]:
    """(N,) 그룹 id(등장 순서로 0..G-1), 실제로 쓴 거리 임계. emb 는 L2 정규화돼 있어야 한다.

    적응 규칙은 대칭이다 (review-v3-design.md (10)):
      · 과병합 — 그룹이 min_groups 미만이거나 한 그룹이 n×max_share 초과 → 0.05씩 하강, 하한 0.05
      · 과분할 — 하강이 불필요했는데 그룹 수가 n×frag_share 초과 → 0.05씩 상승,
        상한 distance+raise_cap. 상승이 과병합 조건을 깨면 한 걸음 되돌리고 멈춘다
    """
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int), distance
    if n == 1:
        return np.zeros(1, dtype=int), distance
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    # linkage 에 원시 벡터를 주면 scipy 가 pdist(cosine) 을 스칼라 C 루프로 돌려 7,189장 × 1,536차원에 37s 가 든다(#113).
    # 정규화된 벡터의 코사인 거리는 1 − X·Xᵀ 라 BLAS 행렬곱 한 번(≈0.5s)이면 같은 값이다. linkage 자체는 O(N²) 로 싸다.
    X = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)   # 이미 정규화돼 있으면 no-op
    G = 1.0 - X @ X.T
    np.fill_diagonal(G, 0.0)
    np.clip(G, 0.0, 2.0, out=G)          # 부동소수 -1e-16 방지 — linkage 는 음수 거리를 거부한다
    Z = linkage(squareform(G, checks=False), method="average")
    del G

    def cut(d: float):
        labels = fcluster(Z, t=d, criterion="distance")
        sizes = np.bincount(labels)
        return labels, int((sizes > 0).sum()), int(sizes.max())

    def overmerged(g: int, biggest: int) -> bool:
        return g < min(min_groups, n) or biggest > max(1, n * max_share)

    d = distance
    labels, g, biggest = cut(d)
    if overmerged(g, biggest):
        while overmerged(g, biggest) and d > 0.05:
            d = round(d - 0.05, 2)
            labels, g, biggest = cut(d)
    else:
        while g > max(min_groups, n * frag_share) and d < round(distance + raise_cap, 2):
            nd = round(d + 0.05, 2)
            nl, ng, nb = cut(nd)
            if overmerged(ng, nb):
                break
            d, labels, g, biggest = nd, nl, ng, nb
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(int(l), len(remap)) for l in labels], dtype=int), d


def group_profile(labels: np.ndarray) -> dict[str, float]:
    if len(labels) == 0:
        return {}
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    return {"groups": float(len(sizes)), "maxSize": float(sizes.max()),
            "maxShare": float(sizes.max() / len(labels)), "singletons": float((sizes == 1).sum())}
