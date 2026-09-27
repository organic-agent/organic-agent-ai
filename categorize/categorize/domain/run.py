"""한 번의 실행이 층 사이로 넘기는 결과. service 가 만들어 controller · naming 으로 돌려준다.

    Grouped            그룹화(service/pipeline) → naming(service/naming). 갤러리를 다시 읽지 않기 위한 전달물.
    CategorizeResult   그룹화 결과. `to_dict()` 는 Lambda 응답·CLI stdout 용.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from categorize.domain.analysis import PhotoAnalysis


@dataclass
class Grouped:
    """그룹화가 끝난 갤러리 — naming 의 입력. [rows] 와 [X] 는 같은 순서(화면 순)다."""

    rows: list[PhotoAnalysis]   # 점수·벡터가 다 있는 사진, burst_id·embed_group_id 채워짐
    X: np.ndarray               # concat(DINOv3 ⊕ CLIP) 정규화 공간, rows 와 행이 맞는다


@dataclass
class CategorizeResult:
    gallery: str
    pipeline: str = "v3"
    mode: str = "categorize"
    photos: int = 0
    # [GLOSSARY-1 2026-09-27] clusters → bursts, groups → embed_groups (결과 키도 "bursts"·"embedGroups")
    bursts: int = 0
    embed_groups: dict = field(default_factory=dict)
    group_distance: float = 0.0
    similarity_profile: dict = field(default_factory=dict)
    embeddings_source: str = "dinov3"
    naming: dict | str | None = None
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "gallery": self.gallery, "pipeline": self.pipeline, "mode": self.mode,
            "photos": self.photos, "bursts": self.bursts,
            "embedGroups": {k: round(v, 3) for k, v in self.embed_groups.items()},
            "groupDistance": self.group_distance,
            "similarityProfile": {k: round(v, 3) for k, v in self.similarity_profile.items()},
            "embeddingsSource": self.embeddings_source,
            "naming": self.naming,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
        }
