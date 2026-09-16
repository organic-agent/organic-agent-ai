"""admin_processing_jobs · admin_photo_revisions 읽기/쓰기 — 관리자 사진 교체 경로.

사진 결과와 잡 상태를 한 트랜잭션에서 attempt·revision 으로 CAS 한다. 취소·재시도·새 리비전이
먼저 이기면 `AdminJobClaimLost` 로 둘 다 버린다.

사진 결과 컬럼의 SET 절 순서는 `photos.metadata_params` 와 같아야 한다.
"""

from __future__ import annotations

import numpy as np
import psycopg

from embedder.domain.admin import AdminPhotoEvent
from embedder.domain.photo import PhotoMetadata
from embedder.repository.photos import metadata_params


class AdminJobClaimLost(RuntimeError):
    """이벤트가 가리키던 attempt/revision 이 더는 현재 작업이 아니다."""


def verify_admin_photo_event(connection: psycopg.Connection, event: AdminPhotoEvent) -> bool:
    """job·attempt·현재 사진·리비전이 이벤트와 전부 같은지 확인한다."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT 1
            FROM admin_processing_jobs j
            JOIN photos p ON p.id = j.target_id
            JOIN galleries g ON g.id = p.gallery_id
            JOIN admin_photo_revisions r ON r.id = j.revision_id AND r.photo_id = p.id
            WHERE j.id = %s AND j.attempt_count = %s AND j.job_type = %s
              AND j.target_type = 'PHOTO' AND j.target_id = %s AND j.revision_id = %s
              AND j.status IN ('DISPATCHING', 'DISPATCHED')
              AND p.id = %s AND p.gallery_id = %s AND p.storage_key = %s
              AND p.status <> 'PENDING' AND p.deleted_at IS NULL AND g.deleted_at IS NULL
              AND r.storage_key = %s
              AND j.payload ->> 'galleryId' = %s
              AND j.payload ->> 'storageKey' = %s
            """,
            (
                event.job_id,
                event.attempt_count,
                event.job_type,
                event.photo_id,
                event.revision_id,
                event.photo_id,
                event.gallery_id,
                event.storage_key,
                event.storage_key,
                str(event.gallery_id),
                event.storage_key,
            ),
        )
        return cursor.fetchone() is not None


def complete_admin_derivative(
    connection: psycopg.Connection,
    event: AdminPhotoEvent,
    preview_key: str,
    meta: PhotoMetadata | None,
) -> None:
    params = (preview_key, *metadata_params(meta), *_photo_identity_params(event))
    _complete_admin_photo_job(
        connection,
        event,
        """
        UPDATE photos p
        SET preview_key = %s,
            taken_at = COALESCE(%s, taken_at),
            camera_make = COALESCE(%s, camera_make),
            camera_model = COALESCE(%s, camera_model),
            exposure_time = COALESCE(%s, exposure_time),
            f_number = COALESCE(%s, f_number),
            iso = COALESCE(%s, iso),
            width = COALESCE(%s, width),
            height = COALESCE(%s, height),
            byte_size = COALESCE(%s, byte_size),
            version = version + 1,
            updated_at = now()
        WHERE p.id = %s AND p.gallery_id = %s AND p.storage_key = %s AND p.deleted_at IS NULL
          AND EXISTS (
              SELECT 1 FROM admin_photo_revisions r
              WHERE r.id = %s AND r.photo_id = p.id AND r.storage_key = p.storage_key
          )
        """,
        params,
    )


def complete_admin_embedding(
    connection: psycopg.Connection,
    event: AdminPhotoEvent,
    vector: np.ndarray,
    model_id: str,
) -> None:
    # CTE 한 문장인 이유: 사진 CAS 가 빗나가면 벡터 upsert 도 0행이어야 rowcount 로 판정할 수 있다.
    # photos.status 는 쓰지 않는다(권한 없음). version 만 올려 CAS 한다.
    _complete_admin_photo_job(
        connection,
        event,
        """
        WITH target AS (
            UPDATE photos p
            SET version = version + 1, updated_at = now()
            WHERE p.id = %s AND p.gallery_id = %s AND p.storage_key = %s AND p.deleted_at IS NULL
              AND EXISTS (
                  SELECT 1 FROM admin_photo_revisions r
                  WHERE r.id = %s AND r.photo_id = p.id AND r.storage_key = p.storage_key
              )
            RETURNING p.id
        )
        INSERT INTO photo_analysis (photo_id, embedding, embedding_model, created_at, updated_at)
        SELECT id, %s, %s, now(), now() FROM target
        ON CONFLICT (photo_id) DO UPDATE
        SET embedding = EXCLUDED.embedding,
            embedding_model = EXCLUDED.embedding_model,
            version = photo_analysis.version + 1,
            updated_at = now()
        """,
        (*_photo_identity_params(event), vector, model_id),
    )


def fail_admin_photo_job(
    connection: psycopg.Connection,
    event: AdminPhotoEvent,
    failure_code: str,
) -> int:
    """현재 attempt 만 FAILED 로 바꾼다. 취소·완료된 행은 덮어쓰지 않는다."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            UPDATE admin_processing_jobs
            SET status = 'FAILED', failure_code = %s, last_run_at = now(), updated_at = now()
            WHERE id = %s AND attempt_count = %s AND job_type = %s
              AND target_type = 'PHOTO' AND target_id = %s AND revision_id = %s
              AND status IN ('DISPATCHING', 'DISPATCHED')
              AND payload ->> 'galleryId' = %s
              AND payload ->> 'storageKey' = %s
            """,
            (
                failure_code[:80],
                event.job_id,
                event.attempt_count,
                event.job_type,
                event.photo_id,
                event.revision_id,
                str(event.gallery_id),
                event.storage_key,
            ),
        )
        return cursor.rowcount


def _complete_admin_photo_job(
    connection: psycopg.Connection,
    event: AdminPhotoEvent,
    photo_sql: str,
    photo_params: tuple,
) -> None:
    """사진 결과와 job SUCCEEDED 를 호출자의 한 트랜잭션 안에서 CAS 한다."""
    with connection.cursor() as cursor:
        cursor.execute(photo_sql, photo_params)
        if cursor.rowcount != 1:
            raise AdminJobClaimLost("PHOTO_REVISION_MISMATCH")
        cursor.execute(
            """
            UPDATE admin_processing_jobs
            SET status = 'SUCCEEDED', failure_code = NULL, updated_at = now()
            WHERE id = %s AND attempt_count = %s AND job_type = %s
              AND target_type = 'PHOTO' AND target_id = %s AND revision_id = %s
              AND status IN ('DISPATCHING', 'DISPATCHED')
              AND payload ->> 'galleryId' = %s
              AND payload ->> 'storageKey' = %s
            """,
            (
                event.job_id,
                event.attempt_count,
                event.job_type,
                event.photo_id,
                event.revision_id,
                str(event.gallery_id),
                event.storage_key,
            ),
        )
        if cursor.rowcount != 1:
            raise AdminJobClaimLost("JOB_ATTEMPT_MISMATCH")


def _photo_identity_params(event: AdminPhotoEvent) -> tuple:
    return (event.photo_id, event.gallery_id, event.storage_key, event.revision_id)
