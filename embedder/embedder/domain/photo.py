"""층 사이를 오가는 사진 값 타입. 로직이 없고 아무것도 import 하지 않는다.

    PhotoRef         repository → service. 사진이 S3 어디에 있나.
    PhotoMetadata    service → repository. photos 의 EXIF 컬럼과 1:1.
    EmbeddingResult  service → repository. 한 장의 처리 결과 전부.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True)
class PhotoRef:
    photo_id: int
    storage_key: str


@dataclass(frozen=True)
class PhotoMetadata:
    """photos 의 EXIF 컬럼과 1:1. 컬럼이 전부 nullable 이라 값이 없으면 None."""

    taken_at: datetime | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    exposure_time: str | None = None
    f_number: float | None = None
    iso: int | None = None
    width: int | None = None
    height: int | None = None
    byte_size: int | None = None


@dataclass(frozen=True)
class EmbeddingResult:
    """사진 한 장을 끝까지 처리한 결과. `vector` 는 S3 에 올라간 `preview_key` 파일에서 계산한 값이다.

    `metadata` 만 None 일 수 있다 — EXIF 추출은 best-effort 라 실패해도 벡터·미리보기는 적재한다.
    """

    ref: PhotoRef
    vector: "np.ndarray"
    preview_key: str
    metadata: PhotoMetadata | None
