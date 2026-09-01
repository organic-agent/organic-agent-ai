"""Bedrock 클라이언트 래퍼 — structured output 한 종류만 쓴다.

tech-stack.md §5는 `AnthropicBedrockMantle`을 적었지만 **ap-northeast-2에는 Mantle 엔드포인트가 없다**
(`bedrock-mantle.ap-northeast-2.api.aws` DNS 없음, us-east-1은 있음 — 2026-08-25 확인). 그래서 legacy
`AnthropicBedrock`(bedrock-runtime, InvokeModel)을 쓴다. 이 경로에서도 `output_config` JSON 스키마가
동작함을 실호출로 확인했다. 서울 온디맨드에 Haiku 4.5가 없어 `global.` 크로스 리전 프로필을 쓴다 —
텍스트만 보내므로 국외 라우팅이 무방하다. tech-stack.md §5 갱신 대상.

`LlmClient`는 프로토콜이다. 실제 구현은 `BedrockClient`, 테스트는 `FakeClient`(tests/test_llm.py).
호출부(reasons·feedback)는 이 프로토콜만 본다.
"""

from __future__ import annotations

import json
import logging
from typing import Protocol

log = logging.getLogger(__name__)


class LlmClient(Protocol):
    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> dict: ...


class BedrockClient:
    """anthropic SDK 1.x · `output_config.format`으로 JSON 스키마를 강제한다. aws_region 필수(1.x)."""

    def __init__(self, aws_region: str, model_id: str) -> None:
        from anthropic import AnthropicBedrock

        self._client = AnthropicBedrock(aws_region=aws_region)
        self.model_id = model_id

    def complete_json(self, system: str, user: str, schema: dict, max_tokens: int) -> dict:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("LLM refusal")
        text = next(b.text for b in response.content if b.type == "text")
        log.info("bedrock %s · in %s / out %s tokens", self.model_id,
                 response.usage.input_tokens, response.usage.output_tokens)
        return json.loads(text)
