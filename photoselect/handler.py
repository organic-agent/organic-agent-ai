"""Lambda 진입점 (B·C). 이벤트: {"selectionId": N, "jobId": M, "mode": "draft"|"refine"}.

A(전수 분석)는 Lambda가 아니라 GPU EC2의 systemd 루프가 `python -m photoselect analyze`를
부른다(tech-stack.md §4). 여기는 B만 받는다.

지금은 DbStore가 없으므로 호출하면 NotImplementedError가 난다 — wes V22 머지 후
`store.DbStore`를 채우면 이 파일은 바뀌지 않는다.
"""

from __future__ import annotations

import logging

from photoselect import store
from photoselect.config import Settings
from photoselect.draft import job as draft_job

logging.getLogger().setLevel(logging.INFO)
_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    selection_id = event.get("selectionId")
    if selection_id is None:
        raise ValueError("페이로드에 selectionId가 없습니다")
    mode = event.get("mode", "draft")
    if mode not in ("draft", "refine"):
        raise ValueError(f"mode는 draft|refine: {mode}")

    st = store.DbStore(_SETTINGS)
    # DB 모드에서는 selection → gallery 매핑을 store가 안다. 지금은 자리만.
    gallery = str(event.get("galleryId", ""))
    return draft_job.run(
        st, gallery, settings=_SETTINGS,
        selection_id=str(selection_id), round_no=1 if mode == "draft" else None,
    )
