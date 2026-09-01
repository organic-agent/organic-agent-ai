"""Bedrock 클라이언트 래퍼 — structured output 한 종류만 쓴다. 텍스트 + (선택) 이미지 블록.

tech-stack.md §5는 `AnthropicBedrockMantle`을 적었지만 **ap-northeast-2에는 Mantle 엔드포인트가 없다**
(`bedrock-mantle.ap-northeast-2.api.aws` DNS 없음, us-east-1은 있음 — 2026-08-25 확인). 그래서 legacy
`AnthropicBedrock`(bedrock-runtime, InvokeModel)을 쓴다. 이 경로에서도 `output_config` JSON 스키마가
동작함을 실호출로 확인했다(이미지 블록 포함, Sonnet 4.6, 2026-08-30). 서울 온디맨드에 Sonnet이 없어 `global.`
크로스 리전 프로필을 쓴다(`aws bedrock list-inference-profiles --region ap-northeast-2` 로 확인).

이미지도 보낸다 — 이유 문장이 사진을 직접 보고 말하게 하기 위해서다(reasons.py). 원본이 아니라
embedder 미리보기를 다시 줄인 JPEG(긴 변 768 안팎)이고, 크로스 리전 프로필이므로 **국외로 나간다**.
`LlmKnobs.reasons_vision=False` 로 끄면 텍스트만 간다.

`LlmClient`는 프로토콜이다. 실제 구현은 `BedrockClient`, 테스트는 tests 의 Fake.
호출부(reasons·feedback)는 이 프로토콜만 본다.
"""

from __future__ import annotations

import base64
import io
import json
import logging
from typing import Protocol

log = logging.getLogger(__name__)

#: user 메시지 한 조각. text 는 문자열, image 는 JPEG 바이트.
Part = tuple[str, str | bytes]   # ("text", str) | ("image", bytes)


class LlmClient(Protocol):
    def complete_json(self, system: str, user: str | list[Part], schema: dict, max_tokens: int) -> dict: ...


def jpeg_bytes(path: str, long_edge: int = 768, quality: int = 80) -> bytes:
    """LLM 에 보낼 크기로 다시 줄인 JPEG. EXIF 회전 반영."""
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def to_content(user: str | list[Part]) -> str | list[dict]:
    """프로토콜의 user 를 Messages API content 로. 문자열은 그대로, 조각 목록은 블록 배열."""
    if isinstance(user, str):
        return user
    blocks: list[dict] = []
    for kind, payload in user:
        if kind == "text":
            blocks.append({"type": "text", "text": payload})
        elif kind == "image":
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg",
                "data": base64.standard_b64encode(payload).decode("ascii")}})
        else:
            raise ValueError(f"알 수 없는 조각: {kind}")
    return blocks


class BedrockClient:
    """anthropic SDK 1.x · `output_config.format`으로 JSON 스키마를 강제한다. aws_region 필수(1.x)."""

    def __init__(self, aws_region: str, model_id: str,
                 timeout: float | None = None, max_retries: int | None = None) -> None:
        """timeout·max_retries 는 동기 경로(compare)용 — 배치는 SDK 기본값(재시도 포함)을 쓴다."""
        from anthropic import AnthropicBedrock

        kwargs: dict = {"aws_region": aws_region}
        if timeout is not None:
            kwargs["timeout"] = timeout
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        self._client = AnthropicBedrock(**kwargs)
        self.model_id = model_id

    def complete_json(self, system: str, user: str | list[Part], schema: dict, max_tokens: int) -> dict:
        response = self._client.messages.create(
            model=self.model_id,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": to_content(user)}],
            output_config={"format": {"type": "json_schema", "schema": schema}},
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("LLM refusal")
        if response.stop_reason == "max_tokens":
            raise RuntimeError(f"LLM 출력이 max_tokens={max_tokens} 에서 잘림")
        text = next(b.text for b in response.content if b.type == "text")
        log.info("bedrock %s · in %s / out %s tokens", self.model_id,
                 response.usage.input_tokens, response.usage.output_tokens)
        return json.loads(text)
