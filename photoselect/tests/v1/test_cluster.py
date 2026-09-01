import numpy as np

from photoselect.v1 import cluster
from photoselect.v1.analyze import represent
from photoselect.v1.store import PhotoAnalysis


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def test_bursts_join_neighbors_only():
    a = _unit([1, 0, 0])
    b = _unit([1, 0.05, 0])        # a와 거의 같다
    c = _unit([0, 1, 0])           # 다르다
    d = _unit([1, 0.05, 0.02])     # a와 닮았지만 순서상 멀다
    emb = np.stack([a, b, c] + [_unit([0, 0, 1])] * 10 + [d])
    ids = cluster.cluster_bursts(emb, threshold=0.95, window=2)
    assert ids[0] == ids[1]
    assert ids[2] != ids[0]
    assert ids[-1] != ids[0], "window 밖이면 닮아도 같은 클러스터가 아니다"


def test_representative_prefers_open_eyes_then_quality():
    rows = [
        PhotoAnalysis(photo_id="a", cluster_id=0, technical_pct=90, aesthetic_pct=50,
                      face_boxes={"face_count": 1}, sub_scores={"eyes_open": 0.2}),
        PhotoAnalysis(photo_id="b", cluster_id=0, technical_pct=60, aesthetic_pct=50,
                      face_boxes={"face_count": 1}, sub_scores={"eyes_open": 0.9}),
        PhotoAnalysis(photo_id="c", cluster_id=1, technical_pct=10, aesthetic_pct=10,
                      face_boxes={"face_count": 0}, sub_scores={"eyes_open": float("nan")}),
    ]
    represent.assign_ranks(rows)
    by = {r.photo_id: r for r in rows}
    assert by["b"].cluster_rank == 0 and by["b"].rank_reason_code == "eyes_open"
    assert by["a"].cluster_rank == 1
    assert by["c"].cluster_rank == 0 and by["c"].rank_reason_code == "single"
