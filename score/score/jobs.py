"""잡 상태 전이 — `ai_analysis_jobs`. score 가 맡는 절반.

wes 는 PENDING 행을 만들고(운영은 이어서 Lambda 를 EVENT 로 부른다), 이후 전이는 AI 쪽이다.
score 는 잡을 **열고**(PENDING→RUNNING) 자기 결과를 `result` 에 **부분 기록**만 한다 — DONE 은
categorize 가 찍는다(체인의 끝). 실패하면 여기서 FAILED.

파이프라인 코드(pipeline.py)는 잡을 모른다 — 진입점(job.py)이 잡을 열고 닫는다.
"""

from __future__ import annotations

import json

import psycopg

ANALYSIS = "ai_analysis_jobs"


def start(conn: psycopg.Connection, job_id: int) -> str:
    """PENDING 이면 RUNNING 으로 올리고, 현재 상태를 돌려준다.

    이미 RUNNING 이면 그대로 둔다 — 데드라인 뒤 자기 재호출은 같은 잡으로 다시 들어온다.
    DONE·FAILED 이거나 없는 잡이면 호출자가 거부한다.
    """
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


def claim_next(conn: psycopg.Connection) -> int | None:
    """가장 오래된 PENDING 하나를 집는다. 로컬 폴링 워커용. SKIP LOCKED 라 워커가 여럿이어도 겹치지 않는다."""
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {ANALYSIS} SET status = 'RUNNING', started_at = now(), updated_at = now(), "
            f"version = version + 1 WHERE id = (SELECT id FROM {ANALYSIS} WHERE status = 'PENDING' "
            f"ORDER BY id FOR UPDATE SKIP LOCKED LIMIT 1) RETURNING id",
            (),
        )
        row = cur.fetchone()
    conn.commit()
    return int(row[0]) if row else None


def record(conn: psycopg.Connection, job_id: int, partial: dict) -> None:
    """상태는 두고 `result` 에 이 단계의 결과를 합친다 — categorize 가 DONE 을 찍을 때 같이 남는다."""
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {ANALYSIS} SET result = COALESCE(result, '{{}}'::jsonb) || %s::jsonb, "
            f"updated_at = now(), version = version + 1 WHERE id = %s",
            (json.dumps(partial, ensure_ascii=False), job_id),
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


# ── 샤딩(#54): result.scoreShards = {total, done, startedAt, results: {"<index>": 샤드 결과}} ──
def shards_begin(conn: psycopg.Connection, job_id: int, total: int, started_at: str | None) -> None:
    """조정자가 샤드 수를 적는다. 얕은 || 라 scoreShards 전체를 새로 쓴다(이전 실행의 카운터가 남지 않게)."""
    record(conn, job_id, {"scoreShards": {"total": total, "done": 0, "startedAt": started_at, "results": {}}})


def record_shard(conn: psycopg.Connection, job_id: int, index: int, partial: dict) -> None:
    """샤드 하나의 결과를 result.scoreShards.results.<index> 에 — 다른 샤드의 결과를 덮지 않게 jsonb_set."""
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {ANALYSIS} SET result = jsonb_set(COALESCE(result, '{{}}'::jsonb), %s::text[], %s::jsonb, true), "
            f"updated_at = now(), version = version + 1 WHERE id = %s",
            (["scoreShards", "results", str(index)], json.dumps(partial, ensure_ascii=False), job_id),
        )
    conn.commit()


def shard_done(conn: psycopg.Connection, job_id: int) -> tuple[int, int]:
    """done 을 원자적으로 +1 하고 (done, total) 을 돌려준다. UPDATE 한 문장이라 동시에 끝난 샤드끼리 겹치지 않는다 —
    done == total 을 본 딱 한 샤드가 categorize 를 연다."""
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {ANALYSIS} SET result = jsonb_set(result, '{{scoreShards,done}}', "
            f"to_jsonb(COALESCE((result #>> '{{scoreShards,done}}')::int, 0) + 1), true), "
            f"updated_at = now(), version = version + 1 WHERE id = %s "
            f"RETURNING (result #>> '{{scoreShards,done}}')::int, (result #>> '{{scoreShards,total}}')::int",
            (job_id,),
        )
        row = cur.fetchone()
    conn.commit()
    return (int(row[0]), int(row[1])) if row and row[1] is not None else (1, 1)


def info(conn: psycopg.Connection, job_id: int) -> tuple[int, str] | None:
    """(gallery_id, mode). mode 는 wes 의 'FULL' | 'NAMING'."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT gallery_id, mode FROM {ANALYSIS} WHERE id = %s", (job_id,))
        row = cur.fetchone()
    return (int(row[0]), str(row[1])) if row else None
