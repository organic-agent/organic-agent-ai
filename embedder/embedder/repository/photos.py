"""photos · photo_analysis 읽기/쓰기 — 일반(배치 임베딩) 경로.

스키마는 wes(Flyway)가 소유한다. 여기서 쓰는 컬럼은 `photo_analysis` 의 벡터 둘(`embedding` · `embedding_model`)과
`photos` 의 `preview_key` · EXIF 컬럼뿐이다. 이미지 바이트가 앱을 거치지 않아 앱이 채울 수 없는 값들이다.

`photos.status` 는 쓰지 않는다 — 임베딩 여부는 `photo_analysis.embedding` 이 말한다.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import psycopg

from embedder.domain.photo import EmbeddingResult, PhotoMetadata, PhotoRef


def fetch_targets(connection: psycopg.Connection, gallery_id: int) -> list[PhotoRef]:
    """갤러리에서 아직 벡터가 없는 사진. 로컬 CLI 전용이며 운영은 `fetch_by_ids` 로 온다.

    PENDING(S3 에 객체가 아직 없을 수 있음)과 휴지통(deleted_at)의 사진·갤러리는 건너뛴다.
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
    """wes 스위퍼가 배정한 사진 id 목록을 그대로 대상으로.

    어느 사진을 임베딩할지는 배정한 쪽이 정했으므로 status 는 보지 않는다. 휴지통과 storage_key 없는 행만 거른다.
    순서는 요청 순서, 없는 id 는 조용히 빠진다.
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
    """벡터 · 파생본 위치 · 촬영 정보를 배치로 적재한다. 커밋은 호출자가 한다.

    - photo_analysis 는 upsert. 다른 배치가 채운 태그·점수 컬럼은 건드리지 않고 벡터 둘만 갈아 끼운다.
    - preview_key 는 덮어쓴다. 이번에 올린 파일이 곧 벡터의 원본이라 이전 키를 지킬 이유가 없다.
    - EXIF 컬럼만 COALESCE. 추출만 실패한 경우 이전 실행의 값을 지우지 않기 위해서다.
    - id 만으로 갱신하지 않고 storage_key 와 활성 조건을 함께 CAS 한다. 대상 선별 뒤 사진이 교체되면
      구 원본의 늦은 결과는 0행 갱신으로 무시된다.
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
        # executemany rowcount 는 실제 갱신 합계라 CAS 에 빗나간 stale 행은 processed 에서 빠진다.
        return max(cursor.rowcount, 0)


def metadata_params(meta: PhotoMetadata | None) -> tuple:
    """EXIF 컬럼 값. 순서는 위 UPDATE 의 SET 절과 같아야 한다. 추출 실패(None)면 전부 NULL → COALESCE 가 기존 값을 지킨다."""
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
