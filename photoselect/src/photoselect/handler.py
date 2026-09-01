"""Lambda 진입점 (B). 이벤트: {"jobId": M} 또는 {"selectionId": N, "mode": "draft"|"refine"}.

A(전수 분석)는 Lambda가 아니라 워커(`python -m photoselect worker`)가 돈다. 여기는 B만 받는다.
파이프라인은 환경변수 PHOTOSELECT_PIPELINE(기본 v2), 근거 문장은 LLM_REASONS=1 이면 Bedrock(v2 는 사진도 보낸다 — global. 크로스 리전).
"""

from __future__ import annotations

import logging
import os

from photoselect import jobs, worker
from photoselect.config import Settings as BaseSettings

logging.getLogger().setLevel(logging.INFO)
_SETTINGS = worker.version_settings(BaseSettings.from_env())
_PIPE = worker.pipeline_of(_SETTINGS)
_LLM = None


def _llm():
    global _LLM
    if _LLM is None and os.environ.get("LLM_REASONS", "").lower() in ("1", "true"):
        _LLM = _PIPE.bedrock_client(_SETTINGS)
    return _LLM


def handler(event: dict, context) -> dict:
    job_id = event.get("jobId")
    selection_id = event.get("selectionId")
    mode = event.get("mode", "draft")

    st = _PIPE.store.DbStore(_SETTINGS, selection_id=selection_id)
    conn = st.conn
    if job_id is not None:
        info = jobs.selection_job_info(conn, int(job_id))
        if info is None:
            raise ValueError(f"ai_selection_jobs 에 없는 잡: {job_id}")
        selection_id, mode = info
        st.selection_id = selection_id
        if not jobs.claim(conn, jobs.SELECTION, int(job_id)):
            logging.warning("잡 %s 은 PENDING 이 아니다 — 건너뜀", job_id)
            return {"skipped": True, "jobId": job_id}
    if selection_id is None:
        raise ValueError("페이로드에 jobId 또는 selectionId 가 없습니다")
    if mode not in ("draft", "refine"):
        raise ValueError(f"mode는 draft|refine: {mode}")

    return worker.run_selection_job(
        st, int(job_id) if job_id is not None else None, selection_id, _SETTINGS,
        llm=_llm(), round_no=1 if mode == "draft" else None,
    )
