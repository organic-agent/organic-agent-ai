"""Lambda 진입점.

페이로드는 **`{"galleryId": 12, "photoIds": [101, 102, …]}` 하나다**(#98). wes 스위퍼가 GPU 워커를 못 쓰거나
워커가 못 따라갈 때 미점수 사진을 50장씩 보낸다(`LambdaStageInvoker.payloadOf` 의 `StageCall.Score`).
점수만 쓰고 끝난다 — 잡 테이블·categorize 체인·샤딩·자기 재호출은 wes 가 소유한다(V16 이후 그 컬럼과 권한이 없다).

15분 타임아웃 앞에서 배치 경계에서 멈춘다(`job.run` 의 `remaining_seconds`). 50장은 그 근처에 가지 않지만 코드 경로는 남긴다 —
멈추면 처리한 만큼은 commit 돼 있고, 남은 사진은 다음 스윕이 다시 보낸다.

옛 갤러리 페이로드(`{"galleryId"}` 만, 또는 `jobId`·`shard`)는 **받지 않는다**: 조용히 갤러리 전수를 스캔하는 대신 에러로 드러낸다.
갤러리 전체를 돌리는 것은 로컬 CLI(`python -m score --gallery-id N`)와 벤치마크(`score.sagemaker`)의 일이다.
"""

from __future__ import annotations

import json
import logging

from score import job
from score.config import Settings

log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    gallery_id = event.get("galleryId")
    if gallery_id is None:
        raise ValueError("페이로드에 galleryId가 없습니다")
    if event.get("photoIds") is None:
        raise ValueError("페이로드에 photoIds 가 없습니다 — v2 계약은 {galleryId, photoIds} 하나다(#98). "
                         "갤러리 전체는 로컬 CLI(python -m score --gallery-id N)로 돌린다")

    gallery_id = int(gallery_id)
    photo_ids = [int(pid) for pid in event["photoIds"]]
    log.info("갤러리 %s: 사진 %d장 배치(폴백)", gallery_id, len(photo_ids))
    result = job.run(gallery_id=gallery_id, settings=_SETTINGS, remaining_seconds=_remaining_seconds(context),
                     photo_ids=photo_ids)
    log.info("결과: %s", json.dumps(result, ensure_ascii=False))
    return result


def _remaining_seconds(context):
    """Lambda 컨텍스트의 남은 실행 시간(초). 로컬 CLI 에는 데드라인이 없다."""
    if context is None or not hasattr(context, "get_remaining_time_in_millis"):
        return None
    return lambda: context.get_remaining_time_in_millis() / 1000.0
