"""층 사이를 오가는 사진 참조. 로직이 없고 아무것도 import 하지 않는다.

    PhotoRef   repository(photos) → service. 갤러리 안의 사진 하나 — id · 파일 위치 · 연사 순서 키.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhotoRef:
    photo_id: str     # 갤러리 안에서 유일. 로컬은 상대 경로, DB는 photos.id
    path: str | None  # 로컬 파일 경로 (DB 모드에서는 임시 다운로드 경로, download=False 면 None)
    #: EXIF 촬영 시각(datetime)·카메라 바디("make model"). 연사 클러스터의 순서·파티션 키 —
    #: 임베더가 채운 photos.taken_at/camera_make/camera_model. 로컬 모드는 None(파일명 순 폴백).
    taken_at: object | None = None
    camera: str | None = None
