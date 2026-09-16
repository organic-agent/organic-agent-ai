"""잡 실패 표시 — `ai_analysis_jobs.error` 한 컬럼(#95, wes V16)."""

from __future__ import annotations

from categorize.repository import jobs
from tests.db_fakes import JobConn


def test_jobs_only_writes_error_column():
    """#95: V16 은 status·started_at·finished_at·result 를 지웠고 photoselect 에 UPDATE (error, updated_at) 만 준다."""
    assert not hasattr(jobs, "start") and not hasattr(jobs, "finish")   # 옛 계약은 사라졌다
    conn = JobConn()
    jobs.fail(conn, 3, "RuntimeError: boom")

    sql, params = conn.executed[0]
    assert sql == "UPDATE ai_analysis_jobs SET error = %s, updated_at = now() WHERE id = %s"
    assert params == ("RuntimeError: boom", 3)
    assert conn.rollbacks == 1 and conn.commits == 1
    for banned in ("status", "started_at", "finished_at", "result", "version"):
        assert banned not in sql
