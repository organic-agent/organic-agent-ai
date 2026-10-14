"""photo_analysis · photos 쓰기 — 메시지 수백 장을 문장 하나로.

embedder 의 `repository/photos.py:store_embeddings` 와 같은 규칙이다. 다른 점은 행마다 executemany 대신 배열 파라미터를
`unnest` 로 펼쳐 한 문장으로 쓰는 것뿐이다(#286: 8장마다 INSERT 평균 86ms × 동시 72 → 수백 장 / 트랜잭션 하나).

- 원본이 그대로인 사진만 쓴다: `photos.id = photoId AND storage_key = storageKey AND deleted_at IS NULL` 이고 갤러리가
  살아 있어야 한다. 대상 배정 뒤 원본이 바뀐 늦은 결과는 0행이 된다(store_embeddings 의 CAS 와 같은 규칙). 계약 v1 은
  photo_analysis 도 같은 조건으로 거른다 — 지워진 사진에 벡터 행을 새로 만들지 않는다.
- photo_analysis 는 upsert. 점수·그룹 컬럼은 건드리지 않고 벡터 둘만 갈아 끼운다.
- photos 는 preview_key 덮어쓰기, EXIF 는 COALESCE(추출만 실패한 경우 이전 값을 지킨다), version + 1, updated_at.

쓰는 테이블·컬럼은 embedder DB 역할의 권한 안이다(wes V15 EMBEDDER_GRANT_CONTRACT): photo_analysis SELECT·INSERT·UPDATE,
photos SELECT(id, gallery_id, storage_key, deleted_at, EXIF, version) · UPDATE(preview_key, EXIF, version, updated_at),
galleries SELECT(id, deleted_at).

커밋은 호출자(handler)가 한다. UPSERT 라 같은 메시지를 두 번 써도 결과가 같다(version 만 한 번 더 오른다).
"""

from __future__ import annotations

from typing import Iterable

import psycopg

from embedwriter.message import PhotoRow

#: unnest 열 순서 = PhotoRow 필드 순서 = SQL 의 t(...) 별칭 순서. 셋이 같이 움직여야 한다.
_COLUMNS = (
    ("photo_id", "bigint"),
    ("storage_key", "text"),
    ("preview_key", "text"),
    ("embedding", "text"),
    ("embedding_model", "text"),
    ("taken_at", "timestamp"),
    ("camera_make", "text"),
    ("camera_model", "text"),
    ("exposure_time", "text"),
    ("f_number", "float8"),
    ("iso", "int"),
    ("width", "int"),
    ("height", "int"),
    ("byte_size", "bigint"),
)

_UNNEST = ", ".join(f"%({name})s::{pg_type}[]" for name, pg_type in _COLUMNS)
_ALIASES = ", ".join(name for name, _ in _COLUMNS)

#: 데이터 변경 CTE(analysis)는 바깥 문장이 읽지 않아도 끝까지 실행된다(PostgreSQL WITH 규칙).
#: 바깥 UPDATE 에도 CAS 조건을 다시 건다 — live 를 읽은 뒤 다른 트랜잭션이 사진을 바꿨으면 재검사로 빠진다.
SQL = f"""
WITH incoming AS (
    SELECT * FROM unnest({_UNNEST}) AS t({_ALIASES})
),
live AS (
    SELECT i.*
    FROM incoming i
    JOIN photos p ON p.id = i.photo_id
                 AND p.storage_key = i.storage_key
                 AND p.deleted_at IS NULL
    JOIN galleries g ON g.id = p.gallery_id
                    AND g.deleted_at IS NULL
),
analysis AS (
    INSERT INTO photo_analysis (photo_id, embedding, embedding_model, created_at, updated_at)
    SELECT photo_id, embedding::vector, embedding_model, now(), now()
    FROM live
    ORDER BY photo_id
    ON CONFLICT (photo_id) DO UPDATE
    SET embedding = EXCLUDED.embedding,
        embedding_model = EXCLUDED.embedding_model,
        version = photo_analysis.version + 1,
        updated_at = now()
    RETURNING photo_id
)
UPDATE photos p
SET preview_key = i.preview_key,
    taken_at = COALESCE(i.taken_at, p.taken_at),
    camera_make = COALESCE(i.camera_make, p.camera_make),
    camera_model = COALESCE(i.camera_model, p.camera_model),
    exposure_time = COALESCE(i.exposure_time, p.exposure_time),
    f_number = COALESCE(i.f_number, p.f_number),
    iso = COALESCE(i.iso, p.iso),
    width = COALESCE(i.width, p.width),
    height = COALESCE(i.height, p.height),
    byte_size = COALESCE(i.byte_size, p.byte_size),
    version = p.version + 1,
    updated_at = now()
FROM live i
WHERE p.id = i.photo_id
  AND p.storage_key = i.storage_key
  AND p.deleted_at IS NULL
"""


def write(connection: psycopg.Connection, rows: Iterable[PhotoRow]) -> int:
    """적재하고 실제로 갱신된 사진 수를 돌려준다(CAS 에 빗나간 늦은 결과는 빠진다). 커밋은 호출자가 한다."""
    params = build_params(rows)
    if not params["photo_id"]:
        return 0
    with connection.cursor() as cursor:
        cursor.execute(SQL, params)
        return max(cursor.rowcount, 0)


def build_params(rows: Iterable[PhotoRow]) -> dict[str, list]:
    """열 배열 파라미터. 같은 사진이 두 번 오면(SQS 중복 전달·재배정) 뒤의 것만 남긴다 —
    ON CONFLICT DO UPDATE 는 한 문장에서 같은 행을 두 번 건드리면 오류다. photo_id 순으로 정렬해 writer 둘이 겹쳐도
    잠금 순서가 같게 한다."""
    latest: dict[int, PhotoRow] = {}
    for row in rows:
        latest[row.photo_id] = row
    ordered = [latest[photo_id] for photo_id in sorted(latest)]
    return {name: [getattr(row, name) for row in ordered] for name, _ in _COLUMNS}
