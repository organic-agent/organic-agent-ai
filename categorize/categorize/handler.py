"""Lambda 진입점.

페이로드는 `{"galleryId": 12, "jobId": 34}`. score Lambda 가 체인으로 부르거나(FULL), wes 가 직접 부른다(NAMING).
`jobId` 가 없으면 잡 계약 밖의 그룹화(naming 포함, 배정은 잡이 없어 저장하지 않는다) — 확인용.

Bedrock 을 부른다. NAT 없는 서브넷이라 `bedrock-runtime` 인터페이스 VPC 엔드포인트와 실행 역할의
`bedrock:InvokeModel`(크로스 리전 프로필 포함)이 있어야 한다. 데드라인은 보지 않는다 — 갤러리 전체가 수 초 +
Bedrock 몇 번이라 15분을 넘길 일이 없고, 넘기면 그건 버그다.
"""

from __future__ import annotations

import logging

from categorize import job, llm
from categorize.config import Settings

log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    gallery_id = event.get("galleryId")
    if gallery_id is None:
        raise ValueError("페이로드에 galleryId가 없습니다")
    gallery_id = int(gallery_id)
    job_id = int(event["jobId"]) if event.get("jobId") is not None else None

    return job.run(
        gallery_id=gallery_id,
        settings=_SETTINGS,
        job_id=job_id,
        llm=llm.bedrock_client(_SETTINGS),
    )
