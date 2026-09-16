"""학습 재료 — 갤러리 하나의 행렬(GalleryData)과 거기에 부부의 선택을 붙인 학습 단위(LabeledGallery).

repository 가 만들고 service 가 읽는다. 로직은 길이 검증과 id → 행 인덱스 변환뿐이다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GalleryData:
    """갤러리 하나의 학습 재료. 행 순서 = display_order, id. 전부 같은 길이 n."""

    gallery_id: str
    photo_ids: list[str]
    file_names: list[str]
    technical_pct: np.ndarray      # (n,) 0~100
    aesthetic_pct: np.ndarray      # (n,)
    sharpness_pct: np.ndarray      # (n,) sub_scores.sharpness_pct, 없으면 50
    subjects: list[str]            # bride · groom · couple · group · unknown
    cluster_id: np.ndarray         # (n,) int — 연사 클러스터. 음수면 단독
    cluster_rank: np.ndarray       # (n,) int — 0 이 대표
    embed_group_id: np.ndarray     # (n,) int — 컨셉 그룹
    display_order: np.ndarray      # (n,) int
    embedding: np.ndarray          # (n, 768) DINOv3
    clip_embedding: np.ndarray     # (n, 768) CLIP
    embedding_model: str = ""
    model_version: str = ""
    shoot_type: str | None = None

    def __post_init__(self) -> None:
        n = len(self.photo_ids)
        for name in ("file_names", "technical_pct", "aesthetic_pct", "sharpness_pct", "subjects", "cluster_id",
                     "cluster_rank", "embed_group_id", "display_order", "embedding", "clip_embedding"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"GalleryData.{name} 길이 {len(getattr(self, name))} ≠ {n}")

    @property
    def n(self) -> int:
        return len(self.photo_ids)

    def index_of(self, photo_ids: list[str]) -> np.ndarray:
        """photo_id 목록 → 행 인덱스. 없는 id 는 조용히 버린다(삭제된 사진 등)."""
        by_id = {p: i for i, p in enumerate(self.photo_ids)}
        return np.array(sorted({by_id[p] for p in photo_ids if p in by_id}), dtype=int)


@dataclass
class LabeledGallery:
    """학습 단위 — 갤러리 재료 + 그 부부의 최종 선택(photo_id)."""

    data: GalleryData
    selected_ids: list[str] = field(default_factory=list)

    @property
    def positive_idx(self) -> np.ndarray:
        return self.data.index_of(self.selected_ids)
