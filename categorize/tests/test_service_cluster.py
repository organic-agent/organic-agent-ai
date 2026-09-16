"""연사 클러스터 — 카메라 파티션 (review-v3-design.md (1))."""

from __future__ import annotations

import numpy as np

from categorize.service import cluster
from tests.helpers import unit


def test_partition_order_splits_cameras_and_sorts_by_taken_at():
    cams = ["A", "B", "A", "B", None]
    times = [3, 1, 1, 2, None]
    parts = cluster.partition_order(cams, times)
    assert parts == [[2, 0], [1, 3], [4]]     # 파티션은 첫 등장 순, 안은 taken_at 순


def test_partition_order_keeps_input_order_when_taken_at_missing():
    assert cluster.partition_order(["A", "A", "A"], [5, None, 1]) == [[0, 1, 2]]


def test_partitioned_bursts_never_cross_cameras():
    rng = np.random.default_rng(2)
    base = unit(rng.normal(size=16))
    emb = np.stack([unit(base + 0.01 * rng.normal(size=16)) for _ in range(6)])
    cams = ["A", "B", "A", "B", "A", "B"]
    assert len(set(cluster.cluster_bursts(emb, 0.9, 8))) == 1   # 순서만으론 전부 한 덩어리
    parts = cluster.partition_order(cams, list(range(6)))
    cids = cluster.cluster_bursts_partitioned(emb, parts, 0.9, 8)
    assert (cids >= 0).all()
    a_ids = {cids[i] for i in range(6) if cams[i] == "A"}
    b_ids = {cids[i] for i in range(6) if cams[i] == "B"}
    assert a_ids.isdisjoint(b_ids)
    assert len(a_ids) == 1 and len(b_ids) == 1
