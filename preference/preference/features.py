"""특징 x — 스칼라 11 + 임베딩 1536. 순서가 wes 와의 계약이다(`FEATURE_SPEC = pref-v1`).

스칼라는 이미 갤러리 내 백분위(0~1)거나 비율이라 그대로. 임베딩은 행마다 L2 정규화한 DINOv3 ⊕ CLIP 에서 **갤러리 평균을
뺀다** — 촬영 장소·작가 스타일은 갤러리마다 다르므로 절대 위치가 아니라 "그 갤러리 안에서 어느 쪽"을 배운다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from preference.store import GalleryData

SUBJECTS = ("bride", "groom", "couple", "group")
SCALAR_NAMES = (
    "technical_pct", "aesthetic_pct", "sharpness_pct",
    "subj_bride", "subj_groom", "subj_couple", "subj_group",
    "log_group_share", "timeline_pos", "cluster_size_rel", "is_burst",
)
N_SCALAR = len(SCALAR_NAMES)
N_EMB = 768 * 2


@dataclass
class Features:
    scalar: np.ndarray   # (n, 11)
    emb: np.ndarray      # (n, 1536), 갤러리 평균 제거
    emb_mean: np.ndarray  # (1536,) — wes 가 같은 값을 빼야 하므로 함께 둔다(추론 시 갤러리에서 다시 계산)


def _l2(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)


def cluster_sizes(cluster_id: np.ndarray) -> np.ndarray:
    """행마다 자기 클러스터 크기. 음수 id 는 단독(크기 1)."""
    size = np.ones(len(cluster_id), dtype=float)
    valid = cluster_id >= 0
    if valid.any():
        ids, inv, counts = np.unique(cluster_id[valid], return_inverse=True, return_counts=True)
        size[valid] = counts[inv]
    return size


def build(gd: GalleryData) -> Features:
    n = gd.n
    subj = np.zeros((n, len(SUBJECTS)))
    for i, s in enumerate(gd.subjects):
        if s in SUBJECTS:
            subj[i, SUBJECTS.index(s)] = 1.0
    gsize = np.ones(n)
    valid = gd.embed_group_id >= 0
    if valid.any():
        _, inv, counts = np.unique(gd.embed_group_id[valid], return_inverse=True, return_counts=True)
        gsize[valid] = counts[inv]
    csize = cluster_sizes(gd.cluster_id)
    order = np.argsort(np.argsort(gd.display_order, kind="stable"), kind="stable")
    scalar = np.column_stack([
        gd.technical_pct / 100.0,
        gd.aesthetic_pct / 100.0,
        gd.sharpness_pct / 100.0,
        subj,
        np.log(gsize / n),
        order / max(n - 1, 1),
        csize / csize.max(),
        (csize > 1).astype(float),
    ])
    emb = np.concatenate([_l2(gd.embedding.astype(np.float64)), _l2(gd.clip_embedding.astype(np.float64))], axis=1)
    mean = emb.mean(axis=0)
    return Features(scalar=scalar, emb=emb - mean, emb_mean=mean)
