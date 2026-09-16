"""잡 계약 (wes V16) — 성공 경로는 잡 테이블을 건드리지 않고, 실패 경로는 error 한 컬럼만 쓴다."""

from __future__ import annotations

import pytest

from categorize.config.settings import Settings
from categorize.domain.photo import PhotoRef
from categorize.service import job as job_mod
from tests.db_fakes import JobConn


def _settings(tmp_path):
    return Settings(out_root=tmp_path, dataset_root=tmp_path, work_dir=tmp_path)


def test_job_run_does_not_open_or_close_the_job(monkeypatch, tmp_path):
    """성공 경로는 잡 테이블을 건드리지 않는다 — 여는 것도 닫는 것도 wes."""
    conn = JobConn()
    monkeypatch.setattr(job_mod.connection, "connect", lambda s: conn)
    monkeypatch.setattr(job_mod, "load_db", lambda *a, **k: [PhotoRef(photo_id="1", path=None)])
    monkeypatch.setattr(job_mod, "DbStore", lambda settings, connection: "STORE")
    monkeypatch.setattr(job_mod.pipeline, "run", lambda *a, **k: {"gallery": "7"})

    result = job_mod.run(7, settings=_settings(tmp_path), job_id=3, llm="LLM")

    assert result["gallery"] == "7" and "elapsedSeconds" in result
    assert conn.executed == []                                          # 잡 테이블 접근 0


def test_job_run_writes_error_when_pipeline_fails(monkeypatch, tmp_path):
    conn = JobConn()
    monkeypatch.setattr(job_mod.connection, "connect", lambda s: conn)
    monkeypatch.setattr(job_mod, "load_db", lambda *a, **k: [PhotoRef(photo_id="1", path=None)])
    monkeypatch.setattr(job_mod, "DbStore", lambda settings, connection: "STORE")

    def boom(*a, **k):
        raise RuntimeError("no vectors")

    monkeypatch.setattr(job_mod.pipeline, "run", boom)

    with pytest.raises(RuntimeError):
        job_mod.run(7, settings=_settings(tmp_path), job_id=3, llm="LLM")

    sql, params = conn.executed[0]
    assert "SET error = %s" in sql and params == ("RuntimeError: no vectors", 3)
