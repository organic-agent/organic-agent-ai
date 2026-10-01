"""2층 그룹 — 과분할 가드 · BLAS 거리 = scipy cosine · 1층 경계를 넘지 않는 전역 id."""

from __future__ import annotations

import numpy as np

from categorize.service import grouping
from tests.helpers import unit


def _clusters(k=2, n_per=10, noise=0.05, seed=3, dim=16):
    rng = np.random.default_rng(seed)
    centers = [unit(rng.normal(size=dim)) for _ in range(k)]
    return np.stack([unit(c + noise * rng.normal(size=dim)) for c in centers for _ in range(n_per)])


def test_embed_groups_raises_threshold_on_fragmentation():
    emb = _clusters()
    labels, used = grouping.embed_groups(emb, 0.01)
    assert used > 0.01
    assert len(set(labels.tolist())) == 2


def test_one_background_stays_one_group():
    """과병합 가드를 뺐다 — 배경 하나로 찍은 컨셉은 그룹 하나가 맞다."""
    labels, _ = grouping.embed_groups(_clusters(k=1), 0.2)
    assert set(labels.tolist()) == {0}


def test_embed_groups_matmul_distance_matches_scipy_cosine_linkage():
    """거리 행렬을 BLAS 로 만들어도(#113) scipy 의 원시-벡터 cosine linkage 와 같은 분할이어야 한다."""
    from scipy.cluster.hierarchy import fcluster, linkage

    emb = _clusters(k=12, n_per=25, noise=0.04, seed=13, dim=48)
    labels, used = grouping.embed_groups(emb, 0.2)
    ref = fcluster(linkage(emb, method="average", metric="cosine"), t=used, criterion="distance")
    pairs = set(zip(labels.tolist(), ref.tolist()))
    assert len(pairs) == len(set(labels.tolist())) == len(set(ref.tolist()))
    scaled, used2 = grouping.embed_groups(emb * 7.0, 0.2)
    assert used2 == used and len(set(zip(labels.tolist(), scaled.tolist()))) == len(pairs)


def test_detail_groups_never_cross_concepts_and_ids_are_global():
    emb = _clusters(k=1, n_per=20)                          # 전부 같은 배경이어도
    concept_of = [0] * 10 + [1] * 10                         # 1층이 다르면 다른 그룹
    ids = grouping.detail_groups(emb, concept_of, 0.2)
    assert set(ids[:10].tolist()) == {0} and set(ids[10:].tolist()) == {1}
