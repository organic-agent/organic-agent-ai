"""한 번의 실행이 층 사이로 넘기는 결과. service 가 만들어 controller · naming 으로 돌려준다.

    Grouped            그룹화(service/pipeline) → naming(service/naming). 갤러리를 다시 읽지 않기 위한 전달물.
                       구간(segmentation)까지 담는다 — 1층 묶기의 입력.
    CategorizeResult   그룹화 결과. `to_dict()` 는 Lambda 응답·CLI stdout 용.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from categorize.domain.analysis import PhotoAnalysis

if TYPE_CHECKING:
    from categorize.service.segment import Segmentation


@dataclass
class Grouped:
    """구간까지 끝난 갤러리 — naming 의 입력. [rows] · [X] · [E] 는 같은 순서(화면 순)다."""

    rows: list[PhotoAnalysis]   # 점수·벡터가 다 있는 사진, burst_id 채워짐 (embed_group_id 는 1층이 정해진 뒤)
    X: np.ndarray               # concat(DINOv3 ⊕ CLIP) 정규화 공간 — 2층 그룹
    E: np.ndarray               # DINOv3 정규화 — 이미지 모드 구간 · 빠진 구간 배정
    segmentation: "Segmentation"


@dataclass
class CategorizeResult:
    gallery: str
    pipeline: str = "v4-concept-segments"
    mode: str = "categorize"
    photos: int = 0
    # [GLOSSARY-1 2026-09-27] clusters → bursts, groups → embed_groups (결과 키도 "bursts"·"embedGroups")
    bursts: int = 0
    segment_mode: str = ""
    segments: int = 0
    concepts: dict = field(default_factory=dict)
    embed_groups: dict = field(default_factory=dict)
    similarity_profile: dict = field(default_factory=dict)
    embeddings_source: str = "dinov3"
    naming: dict | str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "gallery": self.gallery, "pipeline": self.pipeline, "mode": self.mode,
            "photos": self.photos, "bursts": self.bursts,
            "segmentMode": self.segment_mode, "segments": self.segments, "concepts": self.concepts,
            "embedGroups": {k: round(v, 3) for k, v in self.embed_groups.items()},
            "similarityProfile": {k: round(v, 3) for k, v in self.similarity_profile.items()},
            "embeddingsSource": self.embeddings_source,
            "naming": self.naming,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
        }
