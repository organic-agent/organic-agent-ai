"""CLI 채팅 — python -m notionchat.api

필요 환경변수:
  NOTION_TOKEN           Notion Integration 토큰
  AWS_*                  Bedrock 호출용 AWS 자격증명 (프로필/환경변수)
  NOTIONCHAT_MODEL_ID    (선택) 기본값은 config.py 참고. anthropic.* 모델은 Mantle,
                         그 외(gpt-oss/Nova)는 Converse 경로로 자동 분기
"""

import readline  # noqa: F401 — input()에 라인 편집/히스토리 부여
import sys

from notionchat.api import config
from notionchat.api.core.provider import ChatError, ChatSession


def main() -> None:
    session = ChatSession()

    print(f"notionchat — 팀 Notion 조회 챗봇 (model: {config.MODEL_ID}, region: {config.AWS_REGION})")
    print("질문을 입력하세요. 종료: exit / Ctrl-D\n")

    while True:
        try:
            question = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            return

        try:
            answer = session.ask(question)
        except ChatError as e:
            print(f"[오류] {e}", file=sys.stderr)
            continue

        u = session.last_usage
        print(f"\n{answer}\n")
        if u:
            print(
                f"  📊 토큰 — 입력 {u['inputTokens']} · 출력 {u['outputTokens']}"
                f" · 캐시 읽기 {u['cacheReadInputTokens']}\n"
            )


if __name__ == "__main__":
    main()
