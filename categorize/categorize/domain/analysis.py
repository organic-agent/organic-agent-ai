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
    """`photo_analysis` 한 행. 필드 이름은 용어집(WES-DOCS glossary)을 따르고, 이름이 다른 컬럼은 repository 가 매핑한다
    (`burst_id`→`cluster_id`, `burst_rank`→`cluster_rank`, `pipeline_version`→`model_version`). 컬럼 이름은 wes 용어 2단계에서 맞춘다."""

    photo_id: str
    subjects: str = "unknown"
    technical_pct: float = 50.0
    aesthetic_pct: float = 50.0
    sub_scores: dict = field(default_factory=dict)   # technical_score, aesthetic_score, sharpness, rank_reason, clip_parent(score 의 키) …
    # [GLOSSARY-1 2026-09-27] cluster_id → burst_id, cluster_rank → burst_rank (용어집: 연사), model_version → pipeline_version (D3)
    burst_id: int = -1
    burst_rank: int = 0
    embed_group_id: int = -1
    pipeline_version: str = ""


@dataclass
class ConceptAssignment:
    """`ai_concept_assignments` 한 행 — 임베딩 그룹 → (큰 분류, 세부 이름).

    필드는 wes 의 폴더 층 이름을 따른다(1층 = concept, 2층 = detail). 컬럼 이름과 다른 필드는 repository/analysis.py 가
    옮긴다: concept_name → parent_name, detail_name → concept_name, proposed_concept → proposed_parent,
    clip_concept → clip_parent 컬럼.
    """

    embed_group_id: int
    concept_name: str
    detail_name: str
    confidence: float
    assigned_by: str                     # 'vlm' | 'nearest'
    proposed_concept: str | None = None  # concept_name='기타'일 때 VLM 이 제안한 1층 이름
    clip_concept: str | None = None      # CLIP zero-shot 1층 라벨의 그룹 다수결 (검증)
    needs_review: bool = False


@dataclass
class GalleryRead:
    """갤러리 한 번 읽기 — score 가 지난 분석 행과 벡터 두 종류. 파이프라인 한 실행에 한 번만 만든다."""

    rows: list[PhotoAnalysis]                 # pipeline_version 있는 행, 화면 순(display_order, id)
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
