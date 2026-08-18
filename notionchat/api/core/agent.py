"""tool use 루프 — Bedrock Converse API 호출 (현재 모델: gpt-oss).

Converse는 모델 제조사와 무관한 Bedrock 공통 인터페이스라, 계정에서 Claude가
개방되면 NOTIONCHAT_MODEL_ID만 anthropic.* 로 바꿔도 이 루프 그대로 동작한다.
(그때 Anthropic 전용 기능(프롬프트 캐싱 등)이 필요하면 Mantle 클라이언트 도입을
검토한다 — .claude/rules/bedrock.md 참고.)

이 루프 구조는 P2(보정 요청 구조화)가 물려받는다.
"""

from collections.abc import Callable

from api import config
from api.core.prompts import SYSTEM_PROMPT
from api.notion.client import NotionAPIError, NotionClient
from api.notion.tools import TOOL_DEFINITIONS, run_tool


class ModelResponseError(Exception):
    """모델이 정상 종료(end_turn/tool_use)하지 못한 경우."""

    def __init__(self, stop_reason: str, detail: str = ""):
        self.stop_reason = stop_reason
        super().__init__(f"stop_reason={stop_reason} {detail}".strip())


def tool_config() -> dict:
    """Anthropic 형식 도구 정의를 Converse toolSpec 형식으로 변환한다."""
    return {
        "tools": [
            {
                "toolSpec": {
                    "name": t["name"],
                    "description": t["description"],
                    "inputSchema": {"json": t["input_schema"]},
                }
            }
            for t in TOOL_DEFINITIONS
        ]
    }


def _system_blocks() -> list[dict]:
    system = [{"text": SYSTEM_PROMPT}]
    if config.PROMPT_CACHE:
        # 안정 프리픽스(도구 정의+시스템 프롬프트)를 캐시 — 매 호출 재전송 비용 절감
        system.append({"cachePoint": {"type": "default"}})
    return system


def _with_cache_point(messages: list[dict]) -> list[dict]:
    """요청 시점에만 마지막 메시지 끝에 cachePoint를 붙인다 (저장되는 히스토리는 비오염).

    루프 내 다음 호출과 대화의 다음 턴이 직전까지의 프리픽스를 캐시에서 읽게 된다.
    """
    if not config.PROMPT_CACHE:
        return messages
    last = messages[-1]
    return [*messages[:-1], {**last, "content": [*last["content"], {"cachePoint": {"type": "default"}}]}]


_USAGE_KEYS = ("inputTokens", "outputTokens", "cacheReadInputTokens", "cacheWriteInputTokens")


def ask(
    runtime,  # boto3 bedrock-runtime client
    notion: NotionClient,
    messages: list[dict],
    on_log: Callable[[str], None],
) -> tuple[str, list[dict], dict]:
    """Converse 형식 messages로 한 턴을 수행하고 (답변, 갱신된 히스토리, 토큰 usage 합계)를 반환한다."""
    tool_calls = 0
    usage = dict.fromkeys(_USAGE_KEYS, 0)

    while True:
        response = runtime.converse(
            modelId=config.MODEL_ID,
            system=_system_blocks(),
            messages=_with_cache_point(messages),
            toolConfig=tool_config(),
            inferenceConfig={"maxTokens": config.MAX_TOKENS},
        )
        for key in _USAGE_KEYS:
            usage[key] += response.get("usage", {}).get(key) or 0
        message = response["output"]["message"]
        messages.append(message)
        stop_reason = response.get("stopReason")

        if stop_reason == "max_tokens":
            raise ModelResponseError("max_tokens", f"maxTokens={config.MAX_TOKENS} 초과로 응답이 잘렸습니다")
        if stop_reason != "tool_use":
            answer = "\n".join(c["text"] for c in message["content"] if "text" in c)
            return answer, messages, usage

        tool_results = []
        for content in message["content"]:
            tool_use = content.get("toolUse")
            if not tool_use:
                continue
            tool_calls += 1
            if tool_calls > config.MAX_TOOL_CALLS:
                on_log(f"도구 호출 상한({config.MAX_TOOL_CALLS}회) 도달 — 지금까지의 정보로 답하도록 요청")
                result, is_error = (
                    f"도구 호출 상한({config.MAX_TOOL_CALLS}회)에 도달했다. 지금까지 얻은 정보로 답하라.",
                    True,
                )
            else:
                on_log(f"{tool_use['name']}({tool_use['input']})")
                try:
                    result, is_error = run_tool(notion, tool_use["name"], dict(tool_use["input"])), False
                except NotionAPIError as e:
                    result, is_error = f"Notion API 오류: {e}", True
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": tool_use["toolUseId"],
                        "content": [{"text": result}],
                        "status": "error" if is_error else "success",
                    }
                }
            )
        messages.append({"role": "user", "content": tool_results})
