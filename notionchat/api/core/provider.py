"""대화 진입점 — 무상태 함수 run_turn과 로컬 UI용 ChatSession.

서버리스(FastAPI)는 히스토리를 요청으로 받아 run_turn에 넘기고,
로컬 CLI는 ChatSession이 히스토리를 대신 들고 같은 함수를 쓴다.
"""

import sys

from botocore.exceptions import BotoCoreError, ClientError

from api import config
from api.core.agent import ModelResponseError, ask
from api.notion.client import NotionClient


class ChatError(Exception):
    """UI에 그대로 보여줄 수 있는 한국어 오류 메시지."""


def _default_log(message: str) -> None:
    print(f"  ⚙ {message}", file=sys.stderr, flush=True)


# 프로세스 수명 동안 재사용하는 클라이언트 (서버리스 웜 스타트에서도 유효)
_clients: dict = {}


def _notion_client() -> NotionClient:
    if "notion" not in _clients:
        _clients["notion"] = NotionClient()
    return _clients["notion"]


def _runtime_client():
    if "runtime" not in _clients:
        import boto3

        # Vercel은 표준 AWS_* 환경변수명을 예약해 막으므로 전용 이름을 사용한다.
        # 미설정(None)이면 boto3 기본 자격증명 체인(로컬 ~/.aws)으로 폴백
        _clients["runtime"] = boto3.client(
            "bedrock-runtime",
            region_name=config.aws_region(),
            aws_access_key_id=config.env("NOTIONCHAT_AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=config.env("NOTIONCHAT_AWS_SECRET_ACCESS_KEY"),
        )
    return _clients["runtime"]


def _log_usage(usage: dict) -> None:
    """턴별 토큰 사용량을 서버 로그로 남긴다 (Vercel 함수 로그에서 비용 실측용)."""
    print(
        f"[usage] model={config.model_id()} in={usage['inputTokens']} out={usage['outputTokens']}"
        f" cache_read={usage['cacheReadInputTokens']} cache_write={usage['cacheWriteInputTokens']}",
        flush=True,
    )


def run_turn(
    question: str,
    history: list,
    on_log=_default_log,
    notion: NotionClient | None = None,
) -> tuple[str, list, dict]:
    """질문 1턴을 수행한다. history는 JSON 직렬화 가능한 Converse 포맷 리스트.

    성공 시 (답변, 갱신된 history, 토큰 usage 합계)를 반환하고,
    실패 시 history를 질문 이전으로 되돌린 뒤 ChatError를 던진다.
    """
    notion = notion or _notion_client()
    checkpoint = len(history)
    history.append({"role": "user", "content": [{"text": question}]})
    try:
        answer, history, usage = ask(_runtime_client(), notion, history, on_log=on_log)
        _log_usage(usage)
        return answer, history, usage
    except ModelResponseError as e:
        del history[checkpoint:]
        raise ChatError(f"모델 응답 실패: {e}") from e
    except ClientError as e:
        del history[checkpoint:]
        raise ChatError(f"Bedrock 오류: {e.response.get('Error', {}).get('Message', str(e))}") from e
    except BotoCoreError as e:
        del history[checkpoint:]
        raise ChatError(f"AWS 호출 실패: {e}") from e


class ChatSession:
    """히스토리를 들고 있는 로컬 CLI용 래퍼. 마지막 턴 usage는 last_usage로 노출."""

    def __init__(self, notion: NotionClient | None = None):
        self._notion = notion
        self.messages: list = []
        self.last_usage: dict = {}

    def reset(self) -> None:
        self.messages = []

    def ask(self, question: str, on_log=_default_log) -> str:
        answer, self.messages, self.last_usage = run_turn(
            question, self.messages, on_log=on_log, notion=self._notion
        )
        return answer
