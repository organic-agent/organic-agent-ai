"""Lambda 진입점. 페이로드 모양으로 두 경로 중 하나로 넘긴다.

- `{"galleryId", "photoIds", "storageKeys"?}` → `controller/embed.py` — wes 스위퍼의 배치 임베딩
- `{"jobId", …}`                             → `controller/admin.py` — 관리자 사진 교체 outbox

wes 는 배치 페이로드에 `jobId` 를 넣지 않는다. 결과 큐 모드(#286)에서는 두 경로가 함수도 갈린다 — 배치는
`<prefix>-embedder`(VPC 밖, RESULT_QUEUE_URL), 관리자는 같은 이미지의 `<prefix>-embedder-admin`(VPC 안, DB).
"""

from __future__ import annotations

import logging

from embedder.config.settings import Settings
from embedder.controller import admin, embed

logging.getLogger().setLevel(logging.INFO)

# 설정만 읽고 모델은 올리지 않는다. 모델은 첫 EMBEDDING 호출에서 캐시된다.
_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    if "jobId" in event:
        return admin.handle(event, _SETTINGS)
    return embed.handle(event, context, _SETTINGS)
