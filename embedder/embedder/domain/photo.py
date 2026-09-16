"""층 사이를 오가는 사진 값 타입. 로직이 없고 아무것도 import 하지 않는다.

    PhotoRef         repository 가 만들어 service 로 — "이 사진은 S3 어디에 있나"
    PhotoMetadata    service 가 만들어 repository 로 — photos 의 EXIF 컬럼과 1:1
    EmbeddingResult  service 가 만들어 repository 로 — 한 장의 처리 결과 전부(벡터·미리보기 위치·촬영 정보)

의존 방향은 한쪽이다: controller · service · repository · infrastructure 가 여기를 import 하고, 여기는 어디도
import 하지 않는다. 예전에는 PhotoRef 가 db.py 에, PhotoMetadata 가 metadata.py 에 있어서 repository 가 service 를
거꾸로 import 했다(#121).
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
    """photos의 EXIF 컬럼과 1:1로 대응한다. 값이 없으면 None -- 컬럼도 전부 nullable이다."""

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
    """사진 한 장을 끝까지 처리한 결과. `db.store_embeddings` 가 이 단위로 적재한다.

    `metadata` 만 None 일 수 있다 — EXIF 추출은 best-effort 라 실패해도 벡터·미리보기는 적재한다(job.py).
    `preview_key` 는 S3 에 실제로 올라간 파일이고, `vector` 는 그 파일에서 계산한 값이다(#24).
    """

    ref: PhotoRef
    vector: "np.ndarray"
    preview_key: str
    metadata: PhotoMetadata | None
