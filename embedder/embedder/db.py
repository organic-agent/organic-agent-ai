"""photos·photo_analysis 테이블 읽기/쓰기.

스키마는 앱(Flyway)이 소유한다. 이 모듈이 건드리는 것은 `photo_analysis`의 벡터(`embedding`·
`embedding_model`)와 `photos`의 파생본 위치(`preview_key`), 촬영 정보(EXIF) 컬럼들, 그리고
`status`·`updated_at`이다. 전부 앱이 채울 수 없는 값이라는 공통점이 있다 -- 이미지 바이트가
앱을 거치지 않기 때문이다.

벡터가 `photos`가 아니라 `photo_analysis`에 있는 이유는 생명주기다(V29). EXIF는 업로드 때 한 번
정해지지만 벡터는 모델을 바꿀 때마다 다시 적는다. AI 분석 배치가 같은 행에 태그·점수·클러스터를
채우므로, 여기서는 그 컬럼을 건드리지 않도록 벡터 두 컬럼만 `ON CONFLICT DO UPDATE` 한다.

접속은 원래 **RDS IAM 인증 토큰**을 썼다. 토큰 생성(`generate_db_auth_token`)은 로컬 서명
연산이라 네트워크를 타지 않는다 -- NAT도 인터페이스 엔드포인트도 없는 이 서브넷에서
자격증명을 얻을 수 있는 유일한 방법이었고, 덕분에 비밀번호가 어디에도 남지 않았다.

**지금은 비밀번호를 쓴다.** 조직 SCP가 `rds-db:connect`를 계정 전체에서 거부하기 때문이다.
이 계정은 조직의 멤버 계정이라 여기서는 풀 수 없다. 원복 절차는 인프라 레포
`docs/runbook.md`의 "SCP 차단" 절에 있다.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from embedder.config import Settings
from embedder.metadata import PhotoMetadata

if TYPE_CHECKING:
    from embedder.admin_event import AdminPhotoEvent
    from embedder.quality import TechnicalQuality

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PhotoRef:
    photo_id: int
    storage_key: str


class AdminJobClaimLost(RuntimeError):
    """이 이벤트가 가리키던 attempt/revision이 더는 현재 작업이 아닐 때."""


def connect(settings: Settings) -> psycopg.Connection:
    # SCP가 풀리면 아래 password 인자를 지우고 이 토큰 생성으로 되돌린다 (import boto3 필요):
    #
    #     token = boto3.client("rds").generate_db_auth_token(
    #         DBHostname=settings.db_auth_host,
    #         Port=settings.db_port,
    #         DBUsername=settings.db_user,
    #     )
    #
    # 그때 DB 쪽에서 `GRANT rds_iam TO embedder;`도 함께 해줘야 한다. 반대로 지금은 그 GRANT가
    # 있으면 안 된다 -- pg_hba가 `hostssl all +rds_iam pam`을 먼저 매칭해서 비밀번호를 아예
    # 보지 않고 PAM으로 보낸다. 그 상태의 증상은 `PAM authentication failed`다.
    connection = psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        # TLS는 IAM 인증 때문만이 아니다. RDS PostgreSQL 15+ 는 rds.force_ssl이 기본 1이라
        # 평문 접속 자체를 거부한다. 비밀번호로 바뀐 지금도 그대로 필요하다.
        sslmode=settings.db_sslmode,
        sslrootcert=settings.db_sslrootcert,
        connect_timeout=10,
    )

    # 이걸 해야 파이썬 리스트/ndarray를 vector 컬럼에 그대로 바인딩할 수 있다. 없으면
    # '[0.1,0.2,...]' 문자열을 손으로 조립하게 되는데, 표기가 조금만 어긋나도 예외가 아니라
    # 파싱 실패로 나타난다.
    register_vector(connection)
    return connection


def fetch_targets(
    connection: psycopg.Connection, gallery_id: int, force: bool, since: str | None = None,
) -> list[PhotoRef]:
    """이번 실행이 처리할 사진.

    기본값은 아직 임베딩이 없는 것만 고른다. 그래서 중간에 죽은 실행을 다시 부르면 남은 것만
    이어서 처리하고, 재시도 로직을 따로 짤 필요가 없다. force는 모델이나 전처리를 바꿔 전량
    다시 계산할 때만 쓴다.

    `since`(ISO 시각)가 있으면 "벡터 있음"을 "그 시각 이후 적재된 벡터 있음"으로 좁힌다(#56). force 실행이
    시작 시각을 샤드·데드라인 재호출에 넘기는 값이라, 재호출이 force 를 잃어도 이번 실행 전의 벡터는
    다시 계산되고 이번 실행이 적은 벡터만 건너뛴다. 보는 컬럼은 store_embeddings 가 ON CONFLICT 에서
    갱신하는 `photo_analysis.updated_at` — wes 의 EMBED 진행 관측과 같은 기준이다. since 가 있으면 force 는 뜻이 없다.

    PENDING은 건너뛴다 -- 업로드 URL만 발급되고 S3에 객체가 아직 없을 수 있는 상태다.

    휴지통(deleted_at)에 있는 사진과 갤러리도 건너뛴다. 앱은 @SQLRestriction으로 그 행을
    아예 보지 않으므로, 여기서 계산해 봐야 쓰이지 않고 purge 때 원본과 함께 사라진다.
    갤러리가 없거나 휴지통이면 대상이 0장이라 잡은 아무것도 하지 않고 끝난다.
    """
    sql = """
        SELECT p.id, p.storage_key
        FROM photos p
        JOIN galleries g ON g.id = p.gallery_id
        WHERE p.gallery_id = %s
          AND p.status <> 'PENDING'
          AND p.deleted_at IS NULL
          AND g.deleted_at IS NULL
    """
    params: tuple = (gallery_id,)
    if since is not None:
        sql += (
            " AND NOT EXISTS (SELECT 1 FROM photo_analysis a"
            " WHERE a.photo_id = p.id AND a.embedding IS NOT NULL AND a.updated_at >= %s)"
        )
        params = (gallery_id, since)
    elif not force:
        sql += (
            " AND NOT EXISTS (SELECT 1 FROM photo_analysis a"
            " WHERE a.photo_id = p.id AND a.embedding IS NOT NULL)"
        )
    # 순서가 계약이다 — 샤드가 같은 목록에서 위치 % total 로 자기 몫을 고른다(job.Shard).
    sql += " ORDER BY p.id"

    with connection.cursor() as cursor:
        cursor.execute(sql, params)
        return [PhotoRef(photo_id=row[0], storage_key=row[1]) for row in cursor.fetchall()]


def fetch_by_ids(connection: psycopg.Connection, photo_ids: list[int]) -> list[PhotoRef]:
    """사진 id 목록을 그대로 대상으로(#73). wes 스위퍼가 UPLOADED 사진을 50장씩 배정해 부르는 v2 경로.

    status 는 보지 않는다 — 어느 사진을 임베딩할지는 배정한 쪽(wes)이 정했고, v2 에서 status 는 "S3 에 있나"만 답한다.
    휴지통(사진·갤러리)과 storage_key 없는 행만 거른다. 순서는 요청 순서. 없는 id 는 조용히 빠진다(결과 장수로 드러난다).
    """
    if not photo_ids:
        return []
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT p.id, p.storage_key
            FROM photos p
            JOIN galleries g ON g.id = p.gallery_id
            WHERE p.id = ANY(%s)
              AND p.storage_key IS NOT NULL
              AND p.deleted_at IS NULL
              AND g.deleted_at IS NULL
            """,
            (list(photo_ids),),
        )
        found = {row[0]: row[1] for row in cursor.fetchall()}
    return [PhotoRef(photo_id=pid, storage_key=found[pid]) for pid in photo_ids if pid in found]


def store_embeddings(
    connection: psycopg.Connection,
    results: Iterable[tuple[PhotoRef, np.ndarray, str, PhotoMetadata | None]],
    model_id: str,
    set_status: str | None = None,
) -> int:
    """계산된 벡터와 파생본 위치, 촬영 정보를 배치로 적재한다.

    벡터 차원은 vector(n) 컬럼이 강제한다. 모델을 바꿔 폭이 달라지면 여기서 DB 에러로
    떨어진다 -- 조용히 틀린 값이 들어가지 않는다는 뜻이라 굳이 앞단에서 또 막지 않는다.

    두 문장이지만 한 트랜잭션이다(커밋은 호출자가 배치 단위로 한다). 따로 커밋하면 "벡터는
    있는데 상태는 UPLOADED인" 중간 상태가 생기고, 앱의 대상 수 집계와 fetch_targets가 서로
    다른 답을 낸다. 미리보기·촬영 정보도 같은 이유로 같은 트랜잭션에 있다.

    photo_analysis는 INSERT ... ON CONFLICT DO UPDATE다. 재실행(--force)이면 행이 이미 있고,
    AI 분석 배치가 태그·점수를 채워 둔 행일 수도 있다 -- 그 컬럼은 건드리지 않고 벡터 둘만
    갈아 끼운다. 분석 배치는 model_version으로 재분석 대상을 판별하므로 여기서 지울 것이 없다.

    **status 는 기본으로 건드리지 않는다(#83).** wes V15 부터 `photos.status` 는 "S3 에 있나"(PENDING·UPLOADED)만 답하고
    임베딩 여부는 `photo_analysis.embedding` 이 말한다 — embedder 역할에 status UPDATE 권한도 없다. 옛 계약(EMBEDDED)이
    필요하면 set_status 로 켠다; 그때도 벡터와 미리보기가 둘 다 있는 행에만 찍힌다. 벡터는 S3에 올라간
    미리보기 JPEG에서 계산하므로(job.py) 여기 도착한 사진은 반드시 preview_key를 갖는다.
    그래서 preview_key는 COALESCE 없이 덮어쓴다 -- 이번 실행이 올린 파일이 곧 벡터의 원본이라
    이전 실행의 키를 지킬 이유가 없다.

    EXIF 컬럼만 COALESCE다. 촬영 정보 추출만 실패하면 그 자리에 None이 오는데, 그때 이전
    실행이 남긴 멀쩡한 값을 지우면 안 된다. 값이 원래 없던 사진에는 NULL이 NULL로 덮이는
    것이라 달라지는 것이 없다.

    대상 선별 뒤 운영자가 사진을 교체할 수 있으므로 id만으로 갱신하면 안 된다. 선별 당시
    storage_key와 활성 사진·갤러리 조건을 함께 CAS해, 구 원본의 늦은 결과는 0행 갱신으로
    무시한다.
    """
    results = list(results)
    analysis_rows: Sequence[tuple] = [
        (ref.photo_id, vector, model_id)
        for ref, vector, _, _ in results
    ]
    rows: Sequence[tuple] = [
        (
            preview_key,
            *_metadata_params(meta),
            ref.photo_id,
            ref.storage_key,
        )
        for ref, _, preview_key, meta in results
    ]
    if not rows:
        return 0

    with connection.cursor() as cursor:
        cursor.executemany(
            """
            INSERT INTO photo_analysis (photo_id, embedding, embedding_model, created_at, updated_at)
            VALUES (%s, %s, %s, now(), now())
            ON CONFLICT (photo_id) DO UPDATE
            SET embedding = EXCLUDED.embedding,
                embedding_model = EXCLUDED.embedding_model,
                version = photo_analysis.version + 1,
                updated_at = now()
            """,
            analysis_rows,
        )
        status_clause = _status_clause(set_status)
        cursor.executemany(
            f"""
            UPDATE photos
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
                {status_clause}
                version = version + 1,
                updated_at = now()
            WHERE id = %s AND storage_key = %s AND deleted_at IS NULL
              AND EXISTS (
                  SELECT 1 FROM galleries g
                  WHERE g.id = photos.gallery_id AND g.deleted_at IS NULL
              )
            """,
            rows,
        )
        # fetch 뒤 사진 교체·휴지통 이동이 먼저 끝났다면 id는 같아도 old storage_key의
        # 결과를 새 사진에 쓰지 않는다. executemany rowcount는 실제 갱신 합계이므로 stale
        # 행은 processed에서 빠지고, 다음 현재 작업이 새 storage_key를 처리한다.
        return max(cursor.rowcount, 0)


#: 갤러리 잡 advisory lock의 앞쪽 키. 같은 DB를 쓰는 다른 프로세스(wes·photoselect)와
#: 키 공간이 겹치지 않게 이 모듈만의 상수를 쓴다. 뒤쪽 키가 gallery_id다.
GALLERY_LOCK_NAMESPACE = 0x454D42  # 'EMB'
#: 뒤쪽 키 = gallery_id * LOCK_STRIDE + shard_index (#56). 샤드가 최대 8 이라 여유 있게 64 — score 와 같은 모양.
#: 샤드 없는 실행(조정자·N=1)은 샤드 0 과 같은 키라 둘이 겹쳐 돌지 않는다.
LOCK_STRIDE = 64


def try_lock_gallery(connection: psycopg.Connection, gallery_id: int, shard_index: int = 0) -> bool:
    """같은 갤러리(·샤드)의 임베딩 잡이 이미 돌고 있으면 False.

    세션 수준 advisory lock이라 배치마다 commit해도 유지되고, 연결이 닫히면 풀린다 -- Lambda가
    타임아웃으로 죽어도 잠금이 남지 않는다. 버튼 연타와 타임아웃 뒤 자기 재호출이 겹칠 때
    같은 사진을 두 번 처리하지 않게 한다. UPSERT라 두 번 해도 결과는 같지만 시간과 S3 PUT이
    낭비된다.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT pg_try_advisory_lock(%s, %s)",
            (GALLERY_LOCK_NAMESPACE, gallery_id * LOCK_STRIDE + shard_index),
        )
        row = cursor.fetchone()
    return bool(row and row[0])


def verify_admin_photo_event(connection: psycopg.Connection, event: "AdminPhotoEvent") -> bool:
    """job·attempt·현재 사진·보존 리비전이 이벤트의 exact target과 모두 같은지 확인한다."""
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
    event: "AdminPhotoEvent",
    preview_key: str,
    meta: PhotoMetadata | None,
) -> None:
    params = (preview_key, *_metadata_params(meta), *_photo_identity_params(event))
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


def _status_clause(set_status: str | None) -> str:
    """`status = 'X',` 또는 빈 문자열(#73·#83). 값은 파라미터가 아니라 상수로 넣는다 — psycopg 가 status 를 문자열로
    바인딩해도 동작하지만, 문장 자체를 갈라 두는 편이 "무엇을 쓰는지"가 SQL 에 그대로 보인다. 그래서 값을 검사한다."""
    if set_status is not None and not re.fullmatch(r"[A-Z_]{1,32}", set_status):
        raise ValueError(f"EMBED_SET_STATUS 값이 이상하다: {set_status!r}")
    return f"status = '{set_status}'," if set_status else ""


def complete_admin_embedding(
    connection: psycopg.Connection,
    event: "AdminPhotoEvent",
    vector: np.ndarray,
    model_id: str,
    set_status: str | None = None,
) -> None:
    # 벡터는 photo_analysis에 산다(V45). CTE 한 문장인 이유 -- _complete_admin_photo_job이
    # rowcount 1로 CAS 성공을 판정하므로, 사진 CAS가 빗나가면 벡터 upsert도 0행이어야 한다.
    # status 는 store_embeddings 와 같은 규칙(#83): 기본은 안 쓰고(version 만 올려 CAS), set_status 가 있을 때만 찍는다.
    _complete_admin_photo_job(
        connection,
        event,
        f"""
        WITH target AS (
            UPDATE photos p
            SET {_status_clause(set_status)} version = version + 1, updated_at = now()
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


def complete_admin_quality(
    connection: psycopg.Connection,
    event: "AdminPhotoEvent",
    result: "TechnicalQuality",
) -> None:
    _complete_admin_photo_job(
        connection,
        event,
        """
        UPDATE photos p
        SET technical_quality_score = %s,
            technical_quality_signals = CAST(%s AS JSONB),
            quality_analyzed_at = now(),
            version = version + 1,
            updated_at = now()
        WHERE p.id = %s AND p.gallery_id = %s AND p.storage_key = %s AND p.deleted_at IS NULL
          AND EXISTS (
              SELECT 1 FROM admin_photo_revisions r
              WHERE r.id = %s AND r.photo_id = p.id AND r.storage_key = p.storage_key
          )
        """,
        (result.score, json.dumps(result.signals, separators=(",", ":")), *_photo_identity_params(event)),
    )


def fail_admin_photo_job(
    connection: psycopg.Connection,
    event: "AdminPhotoEvent",
    failure_code: str,
) -> int:
    """현재 exact attempt만 명시 실패로 바꾼다. 취소·완료된 행은 덮어쓰지 않는다."""
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
    event: "AdminPhotoEvent",
    photo_sql: str,
    photo_params: tuple,
) -> None:
    """사진 결과와 job SUCCEEDED를 호출자의 한 DB transaction 안에서 CAS한다."""
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


def _photo_identity_params(event: "AdminPhotoEvent") -> tuple:
    return (event.photo_id, event.gallery_id, event.storage_key, event.revision_id)


def _metadata_params(meta: PhotoMetadata | None) -> tuple:
    """EXIF 컬럼에 들어갈 값들. 순서는 위 UPDATE의 SET 절과 같아야 한다.

    추출 자체가 실패했으면(meta is None) 전부 NULL로 보낸다. COALESCE가 받아 기존 값을
    그대로 두므로, 다시 돌려 성공했을 때 채워진다.
    """
    if meta is None:
        return (None,) * 9

    return (
        meta.taken_at,
        meta.camera_make,
        meta.camera_model,
        meta.exposure_time,
        meta.f_number,
        meta.iso,
        meta.width,
        meta.height,
        meta.byte_size,
    )
