"""잡 상태 전이 — `ai_analysis_jobs`·`ai_selection_jobs`.

Store 인터페이스에 일부러 없다(파이프라인 코드는 잡을 모른다). 진입점(CLI·handler·워커)이
잡을 집고(claim) 파이프라인을 돌린 뒤 닫는다(finish/fail). wes는 PENDING만 만들고 이후 전이는
전부 여기다.
"""

from __future__ import annotations

import json

import psycopg

ANALYSIS = "ai_analysis_jobs"
SELECTION = "ai_selection_jobs"


def claim(conn: psycopg.Connection, table: str, job_id: int) -> bool:
    """PENDING → RUNNING. 이미 누가 집었거나 없는 잡이면 False — 두 워커가 같은 잡을 돌리지 않게."""
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET status = 'RUNNING', started_at = now(), updated_at = now(), "
            f"version = version + 1 WHERE id = %s AND status = 'PENDING'",
            (job_id,),
        )
        claimed = cur.rowcount == 1
    conn.commit()
    return claimed


def claim_next(conn: psycopg.Connection, table: str) -> int | None:
    """가장 오래된 PENDING 하나를 집는다. 워커 루프용. SKIP LOCKED라 워커가 여럿이어도 겹치지 않는다."""
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET status = 'RUNNING', started_at = now(), updated_at = now(), "
            f"version = version + 1 WHERE id = (SELECT id FROM {table} WHERE status = 'PENDING' "
            f"ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING id",
            (),
        )
        row = cur.fetchone()
    conn.commit()
    return int(row[0]) if row else None


def finish(conn: psycopg.Connection, table: str, job_id: int, result: dict, round_no: int | None = None) -> None:
    sets = "status = 'DONE', finished_at = now(), updated_at = now(), version = version + 1, result = %s::jsonb"
    params: list = [json.dumps(result, ensure_ascii=False)]
    if table == SELECTION:
        sets += ", round = %s"
        params.append(round_no)
    params.append(job_id)
    with conn.cursor() as cur:
        cur.execute(f"UPDATE {table} SET {sets} WHERE id = %s", params)
    conn.commit()


def fail(conn: psycopg.Connection, table: str, job_id: int, error: str) -> None:
    conn.rollback()   # 파이프라인이 반쯤 쓴 것은 버린다 — 잡 행의 실패 표시만 남긴다
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {table} SET status = 'FAILED', finished_at = now(), updated_at = now(), "
            f"version = version + 1, error = %s WHERE id = %s",
            (error[:4000], job_id),
        )
    conn.commit()


def selection_job_info(conn: psycopg.Connection, job_id: int) -> tuple[int, str] | None:
    """(selection_id, mode). handler가 이벤트 대신 잡 행을 믿는 데 쓴다."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT selection_id, mode FROM {SELECTION} WHERE id = %s", (job_id,))
        row = cur.fetchone()
    return (int(row[0]), str(row[1])) if row else None


def analysis_job_info(conn: psycopg.Connection, job_id: int) -> tuple[int, str] | None:
    """(gallery_id, mode). mode 는 V45 의 'FULL'|'NAMING' — v3 워커가 분기한다. V45 이전
    스키마(컬럼 없음)를 위해 실패하면 FULL 로 둔다."""
    with conn.cursor() as cur:
        try:
            cur.execute(f"SELECT gallery_id, mode FROM {ANALYSIS} WHERE id = %s", (job_id,))
            row = cur.fetchone()
            return (int(row[0]), str(row[1])) if row else None
        except psycopg.errors.UndefinedColumn:
            conn.rollback()
            cur.execute(f"SELECT gallery_id FROM {ANALYSIS} WHERE id = %s", (job_id,))
            row = cur.fetchone()
            return (int(row[0]), "FULL") if row else None
