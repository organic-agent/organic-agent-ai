"""Lambda 진입점. 이벤트: {"jobId": M} 또는 {"selectionId": N, "mode": "draft"|"refine"}
또는 동기 비교샷 {"mode": "compare", "selectionId": N, "photoA": A, "photoB": B} (RequestResponse).

폴더화(FULL·NAMING)는 Lambda가 아니라 워커(`python -m photoselect_v1 worker`)가 돈다 —
여기는 추천과 compare만 받는다. 근거 문장은 LLM_REASONS=1 이면 Bedrock
(사진도 보낸다 — global. 크로스 리전).
"""

from __future__ import annotations

import logging
import os

from photoselect_v1 import jobs, llm as llm_mod, store, worker
from photoselect_v1.config import Settings

logging.getLogger().setLevel(logging.INFO)
_SETTINGS = Settings.from_env()
_LLM = None


def _llm():
    global _LLM
    if _LLM is None and os.environ.get("LLM_REASONS", "").lower() in ("1", "true"):
        _LLM = llm_mod.bedrock_client(_SETTINGS)
    return _LLM


def handler(event: dict, context) -> dict:
    job_id = event.get("jobId")
    selection_id = event.get("selectionId")
    mode = event.get("mode", "draft")

    # 비교샷 — 유일한 동기(RequestResponse) 모드. 잡 테이블을 거치지 않고 응답을 바로 돌려준다.
    if mode == "compare":
        from photoselect_v1.compare import verdict

        photo_a, photo_b = event.get("photoA"), event.get("photoB")
        if selection_id is None or photo_a is None or photo_b is None:
            raise ValueError("compare 페이로드는 selectionId·photoA·photoB 가 필요하다")
        st = store.DbStore(_SETTINGS, selection_id=selection_id)
        gallery = st.gallery_of_selection(selection_id)
        return verdict.run(st, gallery, _SETTINGS, str(photo_a), str(photo_b),
                           selection_id=str(selection_id), llm=llm_mod.compare_client(_SETTINGS))

    st = store.DbStore(_SETTINGS, selection_id=selection_id)
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
        raise ValueError(f"mode는 draft|refine|compare: {mode}")

    return worker.run_selection_job(
        st, int(job_id) if job_id is not None else None, selection_id, _SETTINGS,
        llm=_llm(), round_no=1 if mode == "draft" else None,
    )
