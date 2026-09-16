"""관리자 사진 교체 뒤 한 사진·한 리비전만 처리하는 비동기 작업."""

from __future__ import annotations

import logging
import re

from embedder.config.settings import Settings
from embedder.domain.admin import AdminPhotoEvent
from embedder.infrastructure import model
from embedder.repository import db
from embedder.repository.storage import PhotoStorage
from embedder.service import images, metadata

log = logging.getLogger(__name__)


def run(event: AdminPhotoEvent, settings: Settings) -> dict:
    try:
        # 검증 SELECT가 여는 transaction은 이미지 다운로드·디코딩·모델 추론 전에 끝낸다.
        # 검증 뒤 대상이 바뀌는 race는 아래 final transaction의 exact attempt/revision CAS가
        # 막는다. 따라서 긴 S3/CPU 구간에는 DB connection도 transaction도 잡지 않는다.
        with db.connect(settings) as verification_connection:
            if not db.verify_admin_photo_event(verification_connection, event):
                raise AdminPhotoProcessingError("TARGET_REVISION_MISMATCH")

        storage = PhotoStorage(settings.s3_bucket)
        data = storage.read(event.storage_key)
        original = images.open_original(data)
        prepared = images.prepare(data, settings.resize_long_edge)

        if event.job_type == "DERIVATIVE":
            preview_key = images.preview_key_for(event.storage_key)
            storage.write(
                preview_key,
                images.to_jpeg(prepared, settings.preview_quality),
                "image/jpeg",
            )
            try:
                photo_metadata = metadata.extract(original, len(data))
            except Exception:
                log.exception("exact-photo EXIF 추출 실패: jobId=%s", event.job_id)
                photo_metadata = None
            result = {"previewKey": preview_key}
        elif event.job_type == "EMBEDDING":
            # DERIVATIVE 경로는 이 줄을 지나지 않는다. 모델은 첫 EMBEDDING에서만
            # 올라가고 웜 스타트에서는 model.get_embedder 캐시를 재사용한다.
            vector = model.load_from(settings).encode([prepared])[0]
            result = {"embeddingDimension": len(vector)}
        else:  # AdminPhotoEvent가 막지만 타입 계약을 이 함수에도 남긴다.
            raise AdminPhotoProcessingError("UNSUPPORTED_JOB_TYPE")

        # 결과 계산 뒤 새 connection/transaction에서 사진 결과와 job terminal CAS를 함께
        # commit한다. 취소·재시도·새 리비전이 먼저 이기면 둘 다 rollback된다.
        with db.connect(settings) as final_connection:
            if event.job_type == "DERIVATIVE":
                db.complete_admin_derivative(final_connection, event, preview_key, photo_metadata)
            else:
                db.complete_admin_embedding(final_connection, event, vector, settings.model_id)
            final_connection.commit()
        return _response(event, "SUCCEEDED", result)
    except db.AdminJobClaimLost as error:
        code = _failure_code(AdminPhotoProcessingError(str(error)))
        updated = _persist_failure(event, settings, code, "CAS 실패 상태")
        if updated == 1:
            log.warning("exact-photo 결과 CAS 실패: jobId=%s code=%s", event.job_id, code)
            return _response(event, "FAILED", {"failureCode": code})
        log.info("이미 끝났거나 취소된 exact-photo job 무시: jobId=%s code=%s", event.job_id, code)
        return _response(event, "IGNORED", {"failureCode": code})
    except Exception as error:
        code = _failure_code(error)
        updated = _persist_failure(event, settings, code, "실패 상태")
        if updated != 1:
            log.info("실패 전이 전에 claim이 사라진 exact-photo job 무시: jobId=%s", event.job_id)
            return _response(event, "IGNORED", {"failureCode": code})
        log.exception("exact-photo 처리 실패: jobId=%s type=%s code=%s", event.job_id, event.job_type, code)
        return _response(event, "FAILED", {"failureCode": code})


def _persist_failure(event: AdminPhotoEvent, settings: Settings, code: str, label: str) -> int:
    try:
        with db.connect(settings) as failure_connection:
            updated = db.fail_admin_photo_job(failure_connection, event, code)
            failure_connection.commit()
            return updated
    except Exception:
        log.exception("exact-photo %s 저장 실패: jobId=%s", label, event.job_id)
        raise


class AdminPhotoProcessingError(RuntimeError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _failure_code(error: Exception) -> str:
    if isinstance(error, AdminPhotoProcessingError):
        raw = error.code
    else:
        response = getattr(error, "response", None)
        raw = None
        if isinstance(response, dict):
            raw = response.get("Error", {}).get("Code")
        raw = raw or type(error).__name__ or "PROCESSING_FAILED"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(raw)).upper()[:80]


def _response(event: AdminPhotoEvent, status: str, result: dict) -> dict:
    return {
        "jobId": event.job_id,
        "attemptCount": event.attempt_count,
        "jobType": event.job_type,
        "photoId": event.photo_id,
        "revisionId": event.revision_id,
        "status": status,
        **result,
    }
