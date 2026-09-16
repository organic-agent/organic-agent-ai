"""일반 경로 — wes 스위퍼의 배치 임베딩 호출 `{"galleryId": 1, "photoIds": [101, …]}`.

배정받은 사진 목록만 임베딩한다. 잠금·샤딩·자기 재호출은 없고, 남거나 실패한 장은 wes 가 다시 배정한다.
갤러리 전체는 로컬 CLI 의 일이라 Lambda 로는 받지 않는다.
"""

from __future__ import annotations

import json
import logging

from embedder.config.settings import Settings
from embedder.service import job

log = logging.getLogger(__name__)


def handle(event: dict, context, settings: Settings) -> dict:
    gallery_id = event.get("galleryId")
    if gallery_id is None:
        raise ValueError("페이로드에 galleryId가 없습니다")
    if event.get("photoIds") is None:
        raise ValueError("페이로드에 photoIds 가 없습니다 — v2 계약은 {galleryId, photoIds} 하나다(#100). "
                         "갤러리 전체는 로컬 CLI(python -m embedder --gallery-id N)로 돌린다")

    gallery_id = int(gallery_id)
    photo_ids = [int(pid) for pid in event["photoIds"]]
    log.info("갤러리 %s: 사진 %s장 배치", gallery_id, len(photo_ids))
    result = job.run(gallery_id=gallery_id, settings=settings, remaining_seconds=_remaining_seconds(context),
                     photo_ids=photo_ids)
    log.info("결과: %s", json.dumps(result, ensure_ascii=False))
    return result


def _remaining_seconds(context):
    """Lambda 컨텍스트의 남은 실행 시간(초). 로컬 CLI 에는 데드라인이 없다."""
    if context is None or not hasattr(context, "get_remaining_time_in_millis"):
        return None
    return lambda: context.get_remaining_time_in_millis() / 1000.0
