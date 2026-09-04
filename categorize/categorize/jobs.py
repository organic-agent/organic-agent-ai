"""잡 상태 전이 — `ai_analysis_jobs`. categorize 가 맡는 절반: 체인의 끝이라 DONE/FAILED 를 찍는다.

FULL 잡은 score 가 RUNNING 으로 열고 `result.score` 를 기록한 뒤 여기로 넘어온다. NAMING 잡은 wes 가 만든
PENDING 을 여기서 바로 연다. `finish` 는 기존 result 에 합친다 — score 의 부분 결과가 살아남는다.
"""

from __future__ import annotations

import json

import psycopg

ANALYSIS = "ai_analysis_jobs"


def start(conn: psycopg.Connection, job_id: int) -> str:
    """PENDING 이면 RUNNING 으로. 이미 RUNNING(score 가 열어 둔 FULL 잡)이면 그대로. 현재 상태를 돌려준다."""
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {ANALYSIS} SET status = 'RUNNING', started_at = now(), updated_at = now(), "
            f"version = version + 1 WHERE id = %s AND status = 'PENDING'",
            (job_id,),
        )
        cur.execute(f"SELECT status FROM {ANALYSIS} WHERE id = %s", (job_id,))
        row = cur.fetchone()
    conn.commit()
    return str(row[0]) if row else "MISSING"


def finish(conn: psycopg.Connection, job_id: int, result: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {ANALYSIS} SET status = 'DONE', finished_at = now(), updated_at = now(), "
            f"version = version + 1, result = COALESCE(result, '{{}}'::jsonb) || %s::jsonb WHERE id = %s",
            (json.dumps(result, ensure_ascii=False), job_id),
        )
    conn.commit()


def fail(conn: psycopg.Connection, job_id: int, error: str) -> None:
    conn.rollback()   # 파이프라인이 반쯤 쓴 것은 버린다 — 잡 행의 실패 표시만 남긴다
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {ANALYSIS} SET status = 'FAILED', finished_at = now(), updated_at = now(), "
            f"version = version + 1, error = %s WHERE id = %s",
            (error[:4000], job_id),
        )
    conn.commit()
