"""Lambda 진입점 — 페이로드 `{galleryId, jobId?, conceptCount?}` 를 service.job.run 에 넘기고 Bedrock 클라이언트를 주입한다."""

from __future__ import annotations

from categorize.controller import handler


def test_handler_passes_gallery_job_and_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(handler.job, "run", lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(handler.bedrock, "bedrock_client", lambda s: "LLM")

    result = handler.handler({"galleryId": "7", "jobId": "3", "conceptCount": "10"}, None)

    assert result == {"ok": True}
    assert calls[0]["gallery_id"] == 7 and calls[0]["job_id"] == 3 and calls[0]["llm"] == "LLM"
    assert calls[0]["concept_count"] == 10

    handler.handler({"galleryId": 8}, None)
    assert calls[1]["job_id"] is None and calls[1]["concept_count"] is None


def test_handler_routes_rank_mode_without_job_or_llm(monkeypatch):
    """wes #274 2물결 — `{galleryId, mode: "rank"}` 는 잡·Bedrock 없이 순위만."""
    ranked, full = [], []
    monkeypatch.setattr(handler.job, "rank", lambda **kw: ranked.append(kw) or {"mode": "rank"})
    monkeypatch.setattr(handler.job, "run", lambda **kw: full.append(kw) or {})

    result = handler.handler({"galleryId": "7", "mode": "rank"}, None)

    assert result == {"mode": "rank"}
    assert ranked[0]["gallery_id"] == 7 and full == []


def test_handler_rejects_unknown_mode():
    import pytest

    with pytest.raises(ValueError, match="mode"):
        handler.handler({"galleryId": 7, "mode": "naming"}, None)
