"""notionchat 설정 — 모델 ID는 여기 한 곳에만 둔다 (.claude/rules/bedrock.md)."""

import os
from pathlib import Path


def _load_dotenv() -> None:
    """repo 루트 .env를 읽어, 아직 없는 환경변수만 채운다. 이미 export된 값이 우선."""
    path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip().removeprefix("export ")
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

# 메인 모델: Claude Haiku 4.5 (교차 리전 추론 프로필 — us. 접두사 필수).
# Converse API라 다른 Bedrock 모델로 교체 시 이 값만 바꾸면 된다.
# 현 AWS 계정은 최신 세대(Opus 4.7+, Sonnet 5, GPT-5.6)만 차단 — 개방되면 평가로 비교 후 교체.
# NOTIONCHAT_AWS_REGION이 우선 — Vercel(Lambda)이 자체 AWS_REGION을 주입하기 때문
AWS_REGION = os.environ.get("NOTIONCHAT_AWS_REGION") or os.environ.get("AWS_REGION", "us-east-1")
MODEL_ID = os.environ.get("NOTIONCHAT_MODEL_ID", "us.anthropic.claude-haiku-4-5-20251001-v1:0")

# Bedrock 프롬프트 캐싱은 Claude 계열만 지원 — cachePoint 삽입 여부 결정
PROMPT_CACHE = "anthropic." in MODEL_ID

MAX_TOKENS = 8192

# 질의당 도구 호출 상한 — 폭주 방지 (docs/notion-chatbot-plan.md §2)
MAX_TOOL_CALLS = 8

# read_page 블록 재귀 깊이 제한 — 깊게 중첩된 페이지의 조회 지연 방지
MAX_BLOCK_DEPTH = 3

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"

# 워크스페이스 URL 슬러그 (예: app.notion.com/p/yasic/... 의 "yasic").
# 설정하면 출처 링크를 app.notion.com/p/<slug>/<id> 형식으로 만든다.
NOTION_WORKSPACE = os.environ.get("NOTION_WORKSPACE", "")

# web 프론트를 별도 도메인으로 배포할 때 허용할 출처 (쉼표 구분). 미설정 시 전체 허용
CORS_ORIGINS = [o.strip() for o in os.environ.get("NOTIONCHAT_CORS_ORIGINS", "*").split(",")]


def notion_token() -> str:
    token = os.environ.get("NOTION_TOKEN")
    if not token:
        raise SystemExit(
            "NOTION_TOKEN 환경변수가 필요합니다. "
            "Notion Integration 토큰을 발급해 설정하세요 (docs/notion-chatbot-plan.md §2 구현 순서 2)."
        )
    return token
