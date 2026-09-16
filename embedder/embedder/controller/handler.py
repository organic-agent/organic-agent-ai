"""Lambda 진입점. 페이로드 모양만 보고 두 경로 중 하나로 넘긴다(#123).

- **`{"galleryId": 1, "photoIds": [101, …]}`** → `controller/embed.py` — wes 스위퍼의 배치 임베딩(운영 유일 경로).
- **`{"jobId": …}`** → `controller/admin.py` — 관리자 사진 교체 outbox(DERIVATIVE·EMBEDDING).
  wes 는 그래서 배치 페이로드에 `jobId` 를 넣지 않는다.

함수는 하나다 — wes `AnalysisProperties.functionNameOf` 가 `Embed` 와 `ExactPhoto` 를 같은 embedder 함수로 보낸다.
나중에 둘로 나누면 이 파일을 둘로 복사하고 Dockerfile CMD 만 바꾸면 된다.
"""

from __future__ import annotations

import logging

from embedder.config.settings import Settings
from embedder.controller import admin, embed

logging.getLogger().setLevel(logging.INFO)

# 설정은 가볍게 읽되 모델은 올리지 않는다. DERIVATIVE 는 torch 나 DINO 를 전혀 쓰지 않고,
# EMBEDDING 의 첫 호출만 model.load_from 의 프로세스 캐시를 채운다.
_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    if "jobId" in event:
        return admin.handle(event, _SETTINGS)
    return embed.handle(event, context, _SETTINGS)
