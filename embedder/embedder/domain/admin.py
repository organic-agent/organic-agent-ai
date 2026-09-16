"""관리자 사진 교체(exact-photo) 경로의 값 타입.

    AdminPhotoEvent   wes `StageCallDto.ExactPhoto` 의 필드와 1:1. controller 가 만들어 service · repository 로.
    AdminJobResult    한 건의 결과. service 가 만들어 controller 로.

JSON 페이로드가 어떻게 생겼는지는 여기서 모른다 — 검증·변환은 `controller/admin.py` 의 몫이다.
그래서 wes 계약(키 이름·허용 jobType)이 바뀌어도 controller 만 고친다. `to_dict()` 의 camelCase 키는 지금 읽는 쪽이
없다(EVENT 호출) — 계약이 되면 직렬화를 controller 로 뺀다.
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
    """관리자 사진 교체 한 건의 결과. `status` 는 SUCCEEDED · FAILED · IGNORED 셋 중 하나다.

    `detail` 은 상태에 따라 다르다 — 성공이면 `previewKey` 또는 `embeddingDimension`, 실패·무시면 `failureCode`.
    """

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
