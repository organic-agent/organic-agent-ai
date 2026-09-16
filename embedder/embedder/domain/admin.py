"""관리자 사진 교체 이벤트 값 타입. wes `StageCallDto.ExactPhoto` 의 필드와 1:1.

JSON 페이로드가 어떻게 생겼는지는 여기서 모른다 — 검증·변환은 `controller/admin_event.py` 의 몫이다.
그래서 wes 계약(키 이름·허용 jobType)이 바뀌어도 controller 만 고친다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdminPhotoEvent:
    job_id: int
    attempt_count: int
    job_type: str
    photo_id: int
    gallery_id: int
    storage_key: str
    revision_id: int
