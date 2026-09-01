"""컨셉 그룹 — 같은 배경·구도로 찍은 사진 묶음. 이름은 없다, 경계만 있다.

VLM scene 태그의 자리를 대신한다. 연사 클러스터(`photoselect.cluster`, 코사인 ≥ 0.96·순서 창)보다
훨씬 느슨하게, 순서 제약 없이 임베딩만으로 묶는다. 2026-08-29 갤러리 1 실측(DINOv2, 822장):
평균연결 계층 클러스터 코사인 거리 0.2에서 70그룹 — 케이크 세트 50·해변 41·소파 39·정원 37·
덩굴 아치 36장으로 촬영 세트와 일치했다. 0.3부터 다른 세트가 섞인다.

갤러리마다 분포가 달라 그룹 수가 너무 적거나(전부 한 덩어리) 한 그룹이 절반을 넘으면 임계를
낮춰 다시 자른다 — 다양성 쿼터가 의미를 잃지 않게.
"""

from __future__ import annotations

import numpy as np


def concept_groups(emb: np.ndarray, distance: float, min_groups: int = 4,
                   max_share: float = 0.5) -> tuple[np.ndarray, float]:
    """(N,) 그룹 id(등장 순서로 0..G-1), 실제로 쓴 거리 임계. emb 는 L2 정규화돼 있어야 한다."""
    n = len(emb)
    if n == 0:
        return np.zeros(0, dtype=int), distance
    if n == 1:
        return np.zeros(1, dtype=int), distance
    from scipy.cluster.hierarchy import fcluster, linkage

    Z = linkage(emb, method="average", metric="cosine")
    d = distance
    while True:
        labels = fcluster(Z, t=d, criterion="distance")
        sizes = np.bincount(labels)
        g = int((sizes > 0).sum())
        if (g >= min(min_groups, n) and sizes.max() <= max(1, n * max_share)) or d <= 0.05:
            break
        d = round(d - 0.05, 2)
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(int(l), len(remap)) for l in labels], dtype=int), d


def group_profile(labels: np.ndarray) -> dict[str, float]:
    if len(labels) == 0:
        return {}
    sizes = np.bincount(labels)
    sizes = sizes[sizes > 0]
    return {"groups": float(len(sizes)), "maxSize": float(sizes.max()),
            "maxShare": float(sizes.max() / len(labels)), "singletons": float((sizes == 1).sum())}
