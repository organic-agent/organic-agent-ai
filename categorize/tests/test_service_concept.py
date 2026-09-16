"""임베딩 그룹 — 대칭 적응 규칙 (review-v3-design.md (10))."""

from __future__ import annotations

import numpy as np

from categorize.service import concept
from tests.helpers import unit


def _two_cluster_emb(n_per=10, noise=0.05, seed=3):
    rng = np.random.default_rng(seed)
    centers = [unit(rng.normal(size=16)) for _ in range(2)]
    return np.stack([unit(c + noise * rng.normal(size=16))
                     for c in centers for _ in range(n_per)])


def test_concept_groups_raises_threshold_on_fragmentation():
    emb = _two_cluster_emb()
    labels, used = concept.concept_groups(emb, 0.01, min_groups=2)
    assert used > 0.01
    assert len(set(labels.tolist())) == 2


def test_concept_groups_raise_steps_back_before_overmerge():
    emb = _two_cluster_emb()
    labels, used = concept.concept_groups(emb, 0.01, min_groups=4)
    g = len(set(labels.tolist()))
    assert g >= 4
    assert np.bincount(labels).max() <= len(emb) * 0.5


def test_concept_groups_matmul_distance_matches_scipy_cosine_linkage():
    """거리 행렬을 BLAS 로 만들어도(#113) scipy 의 원시-벡터 cosine linkage 와 같은 분할이어야 한다 — 결과 계약 불변."""
    from scipy.cluster.hierarchy import fcluster, linkage

    rng = np.random.default_rng(13)
    centers = [unit(rng.normal(size=48)) for _ in range(12)]
    emb = np.stack([unit(c + 0.04 * rng.normal(size=48)) for c in centers for _ in range(25)])
    labels, used = concept.concept_groups(emb, 0.2, min_groups=4)

    ref = fcluster(linkage(emb, method="average", metric="cosine"), t=used, criterion="distance")
    pairs = set(zip(labels.tolist(), ref.tolist()))
    assert len(pairs) == len(set(labels.tolist())) == len(set(ref.tolist()))   # 1:1 대응 = 동일 분할
    assert 4 <= len(pairs) <= 12 * 3

    # 정규화 안 된 입력도 안에서 정규화한다 — 스케일이 달라도 같은 분할
    scaled, used2 = concept.concept_groups(emb * 7.0, 0.2, min_groups=4)
    assert used2 == used and set(zip(labels.tolist(), scaled.tolist())).__len__() == len(pairs)
