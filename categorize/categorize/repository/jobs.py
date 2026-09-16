"""잡 실패 표시 — `ai_analysis_jobs.error` 한 컬럼(#95, wes V16).

V16(2026-09-08) 부터 잡의 상태 기계는 **wes 가 소유한다**: wes 가 `ANALYZING → CATEGORIZING` 으로 옮기며 이 Lambda 를
`{galleryId, jobId}` 로 부르고, 결과(배정 행 `ai_concept_assignments(job_id)` + 백분위)를 관측해 DONE 을 찍는다.
`error` 가 채워져 있으면 FAILED 로 간다. 그래서 Lambda 가 쓰는 것은 `error` 하나다 —
photoselect 역할에도 `GRANT UPDATE (error, updated_at)` 만 있고, `status`·`started_at`·`finished_at`·`result` 는
CHECK 위반·없는 컬럼·권한 부족으로 전부 실패한다.

옛 계약(RUNNING 으로 열고 DONE 을 찍던 `start`·`finish`)은 V16 과 함께 사라졌다.
"""

from __future__ import annotations

import psycopg

ANALYSIS = "ai_analysis_jobs"


def fail(conn: psycopg.Connection, job_id: int, error: str) -> None:
    """실패를 잡 행에 남긴다. 잡을 FAILED 로 닫는 것은 이 값을 본 wes 다.

    먼저 rollback 하는 이유: 파이프라인이 반쯤 쓴 것은 버리고 실패 표시만 남긴다."""
    conn.rollback()
    with conn.cursor() as cur:
        cur.execute(
            f"UPDATE {ANALYSIS} SET error = %s, updated_at = now() WHERE id = %s",
            (error[:4000], job_id),
        )
    conn.commit()
