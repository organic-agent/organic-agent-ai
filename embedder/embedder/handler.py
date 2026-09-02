"""Lambda 진입점.

앱이 `POST /api/v1/galleries/{id}/embeddings/run`을 받으면 이 함수를 EVENT(비동기)로 부른다.
페이로드는 `{"galleryId": 1, "force": false}`.
"""

from __future__ import annotations

import logging

from embedder import admin_job, job
from embedder.admin_event import AdminPhotoEvent
from embedder.config import Settings

logging.getLogger().setLevel(logging.INFO)

# 설정은 가볍게 읽되 모델은 올리지 않는다. DERIVATIVE/QUALITY_ANALYSIS는 torch나 DINO를
# 전혀 쓰지 않고, EMBEDDING의 첫 호출만 model.load_from의 프로세스 캐시를 채운다.
_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    if "jobId" in event:
        return admin_job.run(AdminPhotoEvent.from_payload(event), _SETTINGS)

    gallery_id = event.get("galleryId")
    if gallery_id is None:
        raise ValueError("페이로드에 galleryId가 없습니다")

    return job.run(
        gallery_id=int(gallery_id),
        force=bool(event.get("force", False)),
        settings=_SETTINGS,
    )
