"""관리자 사진 교체(exact-photo) 경로의 값 타입.

    AdminPhotoEvent   wes `StageCallDto.ExactPhoto` 와 1:1. controller 가 만들어 service · repository 로.
    AdminJobResult    한 건의 결과. service 가 만들어 controller 로.

JSON 검증·변환은 `controller/admin.py` 의 몫이라 여기는 페이로드 모양을 모른다.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class AdminPhotoEvent:
    job_id: int
    attempt_count: int
    job_type: str
    photo_id: int
    gallery_id: int
    storage_key: str
    revision_id: int


@dataclass(frozen=True)
class AdminJobResult:
    """`status` 는 SUCCEEDED · FAILED · IGNORED. `detail` 은 성공이면 `previewKey` 또는 `embeddingDimension`,
    아니면 `failureCode`."""

    event: AdminPhotoEvent
    status: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "jobId": self.event.job_id,
            "attemptCount": self.event.attempt_count,
            "jobType": self.event.job_type,
            "photoId": self.event.photo_id,
            "revisionId": self.event.revision_id,
            "status": self.status,
            **self.detail,
        }
