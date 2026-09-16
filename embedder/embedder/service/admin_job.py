"""관리자 사진 교체 뒤 한 사진·한 리비전만 처리한다."""

from __future__ import annotations

import logging
import re

from embedder.config.settings import Settings
from embedder.domain.admin import AdminPhotoEvent
from embedder.domain.admin import AdminJobResult
from embedder.infrastructure import model
from embedder.repository import admin_jobs, connection
from embedder.repository.storage import PhotoStorage
from embedder.service import images, metadata

log = logging.getLogger(__name__)


def run(event: AdminPhotoEvent, settings: Settings) -> dict:
    try:
        # 긴 S3/CPU 구간 동안 DB 트랜잭션을 잡지 않는다. 검증 뒤 대상이 바뀌는 race 는 마지막 CAS 가 막는다.
        with connection.connect(settings) as verification_connection:
            if not admin_jobs.verify_admin_photo_event(verification_connection, event):
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
            vector = model.load_from(settings).encode([prepared])[0]
            result = {"embeddingDimension": len(vector)}
        else:
            raise AdminPhotoProcessingError("UNSUPPORTED_JOB_TYPE")

        # 사진 결과와 job 상태를 새 트랜잭션에서 함께 CAS 한다. 먼저 이긴 쪽이 있으면 둘 다 rollback.
        with connection.connect(settings) as final_connection:
            if event.job_type == "DERIVATIVE":
                admin_jobs.complete_admin_derivative(final_connection, event, preview_key, photo_metadata)
            else:
                admin_jobs.complete_admin_embedding(final_connection, event, vector, settings.model_id)
            final_connection.commit()
        return AdminJobResult(event, "SUCCEEDED", result).to_dict()
    except admin_jobs.AdminJobClaimLost as error:
        code = _failure_code(AdminPhotoProcessingError(str(error)))
        updated = _persist_failure(event, settings, code, "CAS 실패 상태")
        if updated == 1:
            log.warning("exact-photo 결과 CAS 실패: jobId=%s code=%s", event.job_id, code)
            return AdminJobResult(event, "FAILED", {"failureCode": code}).to_dict()
        log.info("이미 끝났거나 취소된 exact-photo job 무시: jobId=%s code=%s", event.job_id, code)
        return AdminJobResult(event, "IGNORED", {"failureCode": code}).to_dict()
    except Exception as error:
        code = _failure_code(error)
        updated = _persist_failure(event, settings, code, "실패 상태")
        if updated != 1:
            log.info("실패 전이 전에 claim이 사라진 exact-photo job 무시: jobId=%s", event.job_id)
            return AdminJobResult(event, "IGNORED", {"failureCode": code}).to_dict()
        log.exception("exact-photo 처리 실패: jobId=%s type=%s code=%s", event.job_id, event.job_type, code)
        return AdminJobResult(event, "FAILED", {"failureCode": code}).to_dict()


def _persist_failure(event: AdminPhotoEvent, settings: Settings, code: str, label: str) -> int:
    try:
        with connection.connect(settings) as failure_connection:
            updated = admin_jobs.fail_admin_photo_job(failure_connection, event, code)
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
