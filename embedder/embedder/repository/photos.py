"""photos·photo_analysis 테이블 읽기/쓰기 — 일반(배치 임베딩) 경로.

스키마는 앱(Flyway)이 소유한다. 이 모듈이 건드리는 것은 `photo_analysis`의 벡터(`embedding`·
`embedding_model`)와 `photos`의 파생본 위치(`preview_key`), 촬영 정보(EXIF) 컬럼들, 그리고
`updated_at`이다. 전부 앱이 채울 수 없는 값이라는 공통점이 있다 -- 이미지 바이트가
앱을 거치지 않기 때문이다.

벡터가 `photos`가 아니라 `photo_analysis`에 있는 이유는 생명주기다(V29). EXIF는 업로드 때 한 번
정해지지만 벡터는 모델을 바꿀 때마다 다시 적는다. AI 분석 배치가 같은 행에 태그·점수·클러스터를
채우므로, 여기서는 그 컬럼을 건드리지 않도록 벡터 두 컬럼만 `ON CONFLICT DO UPDATE` 한다.

접속은 `connection.py`, 관리자 사진 교체 잡은 `admin_jobs.py`(#123).
"""

from __future__ import annotations

from typing import Iterable, Sequence

import psycopg

from embedder.domain.photo import EmbeddingResult, PhotoMetadata, PhotoRef


def fetch_targets(connection: psycopg.Connection, gallery_id: int) -> list[PhotoRef]:
    """갤러리에서 아직 벡터가 없는 사진(로컬 CLI 전용, #100).

    운영은 wes 스위퍼가 목록을 배정해 `fetch_by_ids` 로 온다 — 이 경로는 `python -m embedder --gallery-id N`
    (wes `scripts/local-ai.sh`)이 갤러리 하나를 통째로 밀어 볼 때만 쓴다. 다시 부르면 남은 것만 이어서 한다.
    재계산이 필요하면 `photo_analysis` 행을 지운다(관리자 재처리) — force 플래그는 v2 에 없다.

    PENDING 은 건너뛴다: 업로드 URL 만 발급되고 S3 에 객체가 아직 없을 수 있는 상태다.
    휴지통(deleted_at)의 사진·갤러리도 건너뛴다 — 앱이 보지 않고 purge 때 원본과 함께 사라진다.
    """
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT p.id, p.storage_key
            FROM photos p
            JOIN galleries g ON g.id = p.gallery_id
            WHERE p.gallery_id = %s
              AND p.status <> 'PENDING'
              AND p.deleted_at IS NULL
              AND g.deleted_at IS NULL
              AND NOT EXISTS (SELECT 1 FROM photo_analysis a
                              WHERE a.photo_id = p.id AND a.embedding IS NOT NULL)
            ORDER BY p.id
            """,
            (gallery_id,),
        )
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
    results: Iterable[EmbeddingResult],
    model_id: str,
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

    **status 는 건드리지 않는다(#83·#100).** wes V15 부터 `photos.status` 는 "S3 에 있나"(PENDING·UPLOADED)만 답하고
    임베딩 여부는 `photo_analysis.embedding` 이 말한다 — embedder 역할에 그 컬럼 UPDATE 권한도 없다. 벡터는 S3에 올라간
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
        (result.ref.photo_id, result.vector, model_id)
        for result in results
    ]
    rows: Sequence[tuple] = [
        (
            result.preview_key,
            *metadata_params(result.metadata),
            result.ref.photo_id,
            result.ref.storage_key,
        )
        for result in results
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


def metadata_params(meta: PhotoMetadata | None) -> tuple:
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
