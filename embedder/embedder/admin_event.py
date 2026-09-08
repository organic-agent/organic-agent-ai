"""관리자 사진 교체 outbox가 보내는 exact-photo 이벤트 계약."""

from __future__ import annotations

from dataclasses import dataclass


#: wes V15(#100)가 QUALITY_ANALYSIS 잡과 그 점수 컬럼을 지웠다 — 관리자 사진 교체는 파생본과 벡터 둘뿐이다.
SUPPORTED_JOB_TYPES = frozenset({"DERIVATIVE", "EMBEDDING"})


class InvalidAdminPhotoEvent(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class AdminPhotoEvent:
    job_id: int
    attempt_count: int
    job_type: str
    photo_id: int
    gallery_id: int
    storage_key: str
    revision_id: int

    @staticmethod
    def from_payload(payload: dict) -> "AdminPhotoEvent":
        if not isinstance(payload, dict):
            raise InvalidAdminPhotoEvent("INVALID_EVENT")
        job_type = payload.get("jobType")
        if job_type not in SUPPORTED_JOB_TYPES:
            raise InvalidAdminPhotoEvent("UNSUPPORTED_JOB_TYPE")
        storage_key = payload.get("storageKey")
        if not isinstance(storage_key, str) or not storage_key.strip() or len(storage_key) > 500:
            raise InvalidAdminPhotoEvent("INVALID_STORAGE_KEY")
        return AdminPhotoEvent(
            job_id=_positive_int(payload, "jobId"),
            attempt_count=_positive_int(payload, "attemptCount"),
            job_type=job_type,
            photo_id=_positive_int(payload, "photoId"),
            gallery_id=_positive_int(payload, "galleryId"),
            storage_key=storage_key,
            revision_id=_positive_int(payload, "revisionId"),
        )


def _positive_int(payload: dict, name: str) -> int:
    value = payload.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise InvalidAdminPhotoEvent(f"INVALID_{name.upper()}")
    if isinstance(value, str) and (not value.isascii() or not value.isdecimal()):
        raise InvalidAdminPhotoEvent(f"INVALID_{name.upper()}")
    parsed = int(value)
    if parsed <= 0:
        raise InvalidAdminPhotoEvent(f"INVALID_{name.upper()}")
    return parsed
