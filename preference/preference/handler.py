"""Lambda 진입점.

페이로드는 `{"galleryId": N}` — wes 가 갤러리를 CLOSED 로 바꾼 뒤 EVENT 로 부른다(2단계). galleryId 는 로그용이다:
학습은 언제나 마감된 라벨 갤러리 **전부**를 다시 모아서 한다(갤러리 하나가 아니라 누적 데이터가 모델이므로).
결과는 `preference_models` 한 행. 게이트를 통과한 실행만 active 가 되어 wes 추천에 붙는다.
"""

from __future__ import annotations

import logging

from preference import job
from preference.config import Settings
from preference.store import DbStore

log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    gallery_id = event.get("galleryId")
    log.info("preference 학습 시작 — 트리거 갤러리 %s", gallery_id)
    store = DbStore(_SETTINGS)
    try:
        result = job.run_train(store, _SETTINGS)
    finally:
        store.conn.close()
    result["trigger_gallery_id"] = gallery_id
    return result
