"""Lambda 진입점 — 페이로드 `{galleryId, jobId?}` 를 service.job.run 에 넘기고 Bedrock 클라이언트를 주입한다."""

from __future__ import annotations

from categorize.controller import handler


def test_handler_passes_gallery_job_and_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(handler.job, "run", lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(handler.bedrock, "bedrock_client", lambda s: "LLM")

    result = handler.handler({"galleryId": "7", "jobId": "3"}, None)

    assert result == {"ok": True}
    assert calls[0]["gallery_id"] == 7 and calls[0]["job_id"] == 3 and calls[0]["llm"] == "LLM"

    handler.handler({"galleryId": 8}, None)
    assert calls[1]["job_id"] is None
