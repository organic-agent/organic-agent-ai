"""도구 정의(스키마)와 실행을 함께 둔다 (docs/notion-chatbot-plan.md §2 아키텍처)."""

import json
from pathlib import Path

from api import config
from api.notion.blocks import blocks_to_markdown, properties_to_text, rich_text
from api.notion.client import NotionClient

TOOL_DEFINITIONS = [
    {
        "name": "search_notion",
        "description": (
            "Notion 워크스페이스를 키워드로 검색한다. 페이지·데이터베이스의 제목/URL/최종 수정 시각을 반환한다. "
            "질문에 답할 문서를 찾을 때 가장 먼저 호출한다. 키워드 중심 검색이므로 "
            "결과가 비면 다른 표현(동의어, 더 짧은 키워드)으로 다시 검색한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색 키워드"},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_page",
        "description": (
            "페이지 본문 전체를 마크다운으로 읽는다. search_notion으로 찾은 페이지의 내용을 확인할 때 호출한다. "
            "페이지 URL과 last_edited_time을 함께 반환하므로 여러 버전이 있으면 최신 수정본을 우선하라."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "page_id": {"type": "string", "description": "Notion 페이지 ID"},
            },
            "required": ["page_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "query_database",
        "description": (
            "Notion 데이터베이스의 행을 조회해 각 행의 속성을 문장으로 반환한다. "
            "filter는 Notion API filter 객체(JSON) 형식이며 생략하면 전체 조회한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "database_id": {"type": "string", "description": "Notion 데이터베이스 ID"},
                "filter": {
                    "type": "string",
                    "description": "Notion API filter 객체를 JSON 문자열로. 예: {\"property\": \"상태\", \"select\": {\"equals\": \"확정\"}}",
                },
            },
            "required": ["database_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "read_expense_guide",
        "description": (
            "소마(AI·SW마에스트로) 공식 비용 지원 규정 전문을 반환한다: 프로젝트 활동비"
            "(차수별 1~5차 신청 기간·승인·정산 일정, 지원 항목·금액 한도, 신청 절차, AWS 클라우드 비용, 증빙 규칙)와 "
            "자기주도형 학습비(1인당 한도, 강의 지원 방식, 월별 신청 일정). "
            "활동비/학습비/비용 규정 질문은 Notion 검색 전에 반드시 여기부터. "
            "'다음 신청 기간이 언제냐'도 여기 일정표로 답한다. 팀이 실제 쓴 비용 내역만 Notion의 비용 처리 DB를 본다. "
            "공식 문서의 스냅샷이므로 문서 상단의 스냅샷 날짜를 확인하라."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_children",
        "description": (
            "페이지/블록 바로 아래의 하위 블록·하위 페이지 목록을 반환한다. "
            "워크스페이스 구조를 탐색하거나, read_page가 깊이 제한으로 생략한 부분을 확장할 때 호출한다."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "block_id": {"type": "string", "description": "페이지 또는 블록 ID"},
            },
            "required": ["block_id"],
            "additionalProperties": False,
        },
    },
]


def run_tool(client: NotionClient, name: str, tool_input: dict) -> str:
    if name == "search_notion":
        return _search(client, tool_input["query"])
    if name == "read_page":
        return _read_page(client, tool_input["page_id"])
    if name == "query_database":
        return _query_database(client, tool_input["database_id"], tool_input.get("filter"))
    if name == "list_children":
        return _list_children(client, tool_input["block_id"])
    if name == "read_expense_guide":
        return _read_expense_guide()
    raise ValueError(f"알 수 없는 도구: {name}")


def _read_expense_guide() -> str:
    data = Path(__file__).resolve().parent.parent / "data"
    snapshot = data / "asm_expense_guide.md"
    if not snapshot.exists():
        return "활동비 규정 스냅샷이 없다. scripts/snapshot_asm_expenses.py 실행이 필요하다고 안내하라."
    parts = [snapshot.read_text()]
    # 이미지로만 게시된 표의 수동 변환본, 자기주도형 학습비 안내(로그인 페이지 수동 스냅샷)
    for name in ("asm_expense_guide_images.md", "asm_self_study_guide.md"):
        extra = data / name
        if extra.exists():
            parts.append(extra.read_text())
    return "\n\n".join(parts)


def _page_url(obj: dict) -> str:
    """API의 url 필드 대신 페이지 ID로 접근 가능한 링크를 만든다.

    API가 반환하는 url에는 워크스페이스 슬러그가 빠져 있어 브라우저에서 접근이 안 된다.
    NOTION_WORKSPACE가 있으면 app.notion.com/p/<slug>/<id>, 없으면 www.notion.so/<id> 폴백.
    """
    obj_id = (obj.get("id") or "").replace("-", "")
    if not obj_id:
        return obj.get("url", "")
    workspace = config.notion_workspace()
    if workspace:
        return f"https://app.notion.com/p/{workspace}/{obj_id}"
    return f"https://www.notion.so/{obj_id}"


def _title_of(result: dict) -> str:
    if result.get("object") == "database":
        return rich_text(result.get("title", []))
    for prop in (result.get("properties") or {}).values():
        if prop.get("type") == "title":
            return rich_text(prop.get("title", []))
    return "(제목 없음)"


def _search(client: NotionClient, query: str) -> str:
    results = client.search(query).get("results", [])
    if not results:
        return "검색 결과 없음. 다른 키워드로 시도하거나, Integration에 연결되지 않은 문서일 수 있다."
    lines = []
    for r in results:
        lines.append(
            f"- [{r.get('object')}] {_title_of(r)}\n"
            f"  id: {r.get('id')} | url: {_page_url(r)} | last_edited: {r.get('last_edited_time')}"
        )
    return "\n".join(lines)


def _read_page(client: NotionClient, page_id: str) -> str:
    meta = client.page_meta(page_id)
    body = blocks_to_markdown(
        client.all_block_children(page_id),
        fetch_children=client.all_block_children,
        max_depth=config.MAX_BLOCK_DEPTH,
    )
    header = (
        f"제목: {_title_of(meta)}\n"
        f"url: {_page_url(meta)}\n"
        f"last_edited_time: {meta.get('last_edited_time')}\n---"
    )
    return f"{header}\n{body}" if body else f"{header}\n(본문 없음)"


def _query_database(client: NotionClient, database_id: str, filter_json: str | None) -> str:
    filter_obj = json.loads(filter_json) if filter_json else None
    rows = client.query_database(database_id, filter=filter_obj).get("results", [])
    if not rows:
        return "조회된 행 없음."
    lines = []
    for row in rows:
        lines.append(
            f"- {properties_to_text(row.get('properties', {}))}\n"
            f"  id: {row.get('id')} | url: {_page_url(row)} | last_edited: {row.get('last_edited_time')}"
        )
    return "\n".join(lines)


def _list_children(client: NotionClient, block_id: str) -> str:
    children = client.all_block_children(block_id)
    if not children:
        return "하위 블록 없음."
    md = blocks_to_markdown(children, fetch_children=lambda _id: [], max_depth=0)
    return md or "표시할 내용 없음."
