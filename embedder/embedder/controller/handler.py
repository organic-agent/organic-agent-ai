"""Lambda 진입점.

페이로드는 두 가지다(#100).

- **`{"galleryId": 1, "photoIds": [101, 102, …]}`** — wes 스위퍼가 배정한 사진 목록만 임베딩한다(운영 유일 경로).
  `LambdaStageInvoker.payloadOf` 의 `StageCall.Embed` 가 이 모양으로만 보낸다. 잠금·샤딩·자기 재호출 없음 —
  남거나 실패한 장은 wes 가 `photos.dispatched_at` 을 되돌려 다시 배정한다.
- **`{"jobId": …}`** — 관리자 사진 교체 outbox(DERIVATIVE·EMBEDDING). wes 가 그래서 분석 페이로드에는 `jobId` 를 넣지 않는다.

갤러리 전체는 로컬 CLI(`python -m embedder --gallery-id N`)의 일이다 — Lambda 로는 받지 않는다.
15분 타임아웃 앞에서 배치 경계에서 멈춘다(`job.run` 의 `remaining_seconds`). 50장은 그 근처에 가지 않지만 코드 경로는 같다.
"""

from __future__ import annotations

import json
import logging

from embedder.config.settings import Settings
from embedder.controller.admin_event import parse_admin_photo_event
from embedder.service import admin_job, job

log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

# 설정은 가볍게 읽되 모델은 올리지 않는다. DERIVATIVE 는 torch 나 DINO 를 전혀 쓰지 않고,
# EMBEDDING 의 첫 호출만 model.load_from 의 프로세스 캐시를 채운다.
_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    if "jobId" in event:
        return admin_job.run(parse_admin_photo_event(event), _SETTINGS)

    gallery_id = event.get("galleryId")
    if gallery_id is None:
        raise ValueError("페이로드에 galleryId가 없습니다")
    if event.get("photoIds") is None:
        raise ValueError("페이로드에 photoIds 가 없습니다 — v2 계약은 {galleryId, photoIds} 하나다(#100). "
                         "갤러리 전체는 로컬 CLI(python -m embedder --gallery-id N)로 돌린다")

    gallery_id = int(gallery_id)
    photo_ids = [int(pid) for pid in event["photoIds"]]
    log.info("갤러리 %s: 사진 %s장 배치", gallery_id, len(photo_ids))
    result = job.run(gallery_id=gallery_id, settings=_SETTINGS, remaining_seconds=_remaining_seconds(context),
                     photo_ids=photo_ids)
    log.info("결과: %s", json.dumps(result, ensure_ascii=False))
    return result


def _remaining_seconds(context):
    """Lambda 컨텍스트의 남은 실행 시간(초). 로컬 CLI 에는 데드라인이 없다."""
    if context is None or not hasattr(context, "get_remaining_time_in_millis"):
        return None
    return lambda: context.get_remaining_time_in_millis() / 1000.0
