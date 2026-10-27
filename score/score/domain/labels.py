"""사진 라벨 어휘(wes #348 · V46 `photo_labels`) — 라벨 코드와 이 모듈이 쓰는 행의 source. 계산은 `service/labels.py`.

라벨 코드는 wes `AgentLabel` 과 같은 목록이어야 한다 — wes 는 밖의 값을 경고 로그로 무시한다. 어휘를 바꾸면 relabel 로 다시 쓴다.
"""

from __future__ import annotations

#: 라벨 코드. 순서는 wes V46 주석과 같다.
LABELS = ("EYES_CLOSED", "SMILE", "VEIL", "BOUQUET", "PROP", "CLOSE_UP", "HALF_BODY", "FULL_BODY", "BACK_VIEW")

#: `photo_labels.source` — score 의 CLIP 층이 쓴 행. 다른 층(PROBE·VLM·FACE)이 맡은 라벨은 덮지 않는다.
CLIP_SOURCE = "CLIP"
