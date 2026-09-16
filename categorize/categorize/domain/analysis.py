"""분석 레코드와 저장소 인터페이스. 로직이 없다.

    PhotoAnalysis       `photo_analysis` 한 행 — repository ↔ service
    ConceptAssignment   `ai_concept_assignments` 한 행 — service(naming) → repository
    GalleryRead         갤러리 한 번 읽기(행 + 벡터 두 종류) — repository → service
    Store               service 가 보는 저장소 프로토콜. 구현은 repository/analysis.py(DbStore) · repository/local.py(LocalStore)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class PhotoAnalysis:
    """`photo_analysis` 한 행. 컬럼 이름을 그대로 필드로 쓴다."""

    photo_id: str
    subjects: str = "unknown"
    technical_pct: float = 50.0
    aesthetic_pct: float = 50.0
    sub_scores: dict = field(default_factory=dict)   # technical_score, aesthetic_score, sharpness, rank_reason, clip_parent …
    cluster_id: int = -1
    cluster_rank: int = 0
    embed_group_id: int = -1
    model_version: str = ""


@dataclass
class ConceptAssignment:
    """`ai_concept_assignments` 한 행 — 임베딩 그룹 → (큰 분류, 컨셉)."""

    embed_group_id: int
    parent_name: str
    concept_name: str
    confidence: float
    assigned_by: str                     # 'vlm' | 'nearest'
    proposed_parent: str | None = None   # parent_name='기타'일 때 VLM 제안
    clip_parent: str | None = None       # CLIP zero-shot 다수결 (검증)
    needs_review: bool = False


@dataclass
class GalleryRead:
    """갤러리 한 번 읽기 — score 가 지난 분석 행과 벡터 두 종류. 파이프라인 한 실행에 한 번만 만든다."""

    rows: list[PhotoAnalysis]                 # model_version 있는 행, 화면 순(display_order, id)
    embeddings: dict[str, np.ndarray]         # DINOv3 — 없으면 빈 dict(로컬 데이터셋 모드)
    clip_embeddings: dict[str, np.ndarray]    # CLIP ViT-L/14


# ── 인터페이스 ───────────────────────────────────────────────────────────────
class Store(Protocol):
    def read_gallery(self, gallery: str) -> GalleryRead: ...
    def write_groups(self, gallery: str, rows: list[PhotoAnalysis]) -> None: ...
    def write_assignments(self, gallery: str, job_id: int | None,
                          rows: list[ConceptAssignment]) -> None: ...
    def preview_paths(self, gallery: str, photo_ids: list[str]) -> dict[str, str]:
        """대표 사진들의 로컬 JPEG 경로. 없는 사진은 빠진다 — 호출자가 nearest 배정으로 넘긴다."""
        ...
