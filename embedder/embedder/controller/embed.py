"""일반 경로 — wes 스위퍼의 배치 임베딩 호출 `{"galleryId": 1, "photoIds": [101, …]}`.

wes 스위퍼가 배정한 사진 목록만 임베딩한다(운영 유일 경로, #73·#100). `LambdaStageInvoker.payloadOf` 의
`StageCall.Embed` 가 이 모양으로만 보낸다. 잠금·샤딩·자기 재호출 없음 — 남거나 실패한 장은 wes 가
`photos.dispatched_at` 을 되돌려 다시 배정한다.

갤러리 전체는 로컬 CLI(`python -m embedder --gallery-id N`)의 일이다 — Lambda 로는 받지 않는다.
15분 타임아웃 앞에서 배치 경계에서 멈춘다(`job.run` 의 `remaining_seconds`). 50장은 그 근처에 가지 않지만 코드 경로는 같다.
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
