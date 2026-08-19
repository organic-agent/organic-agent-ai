"""Notion 블록 트리 → 마크다운 변환 — 조회 시점 변환, 사전 적재 없음.

핵심 블록 타입만 지원: 문단·헤딩·리스트·토글·코드·표·인용·콜아웃.
이미지/임베드는 플레이스홀더로 대체한다 (docs/notion-chatbot-plan.md §2).
"""

from collections.abc import Callable

# fetch_children(block_id) -> list[block dict]
FetchChildren = Callable[[str], list[dict]]


# 노션에서 한 줄의 텍스트는 서식 단위로 쪼개진 rich text 조각의 배열
def rich_text(items: list[dict]) -> str:
    parts = []
    for t in items or []:
        text = t.get("plain_text", "")
        href = t.get("href")
        parts.append(f"[{text}]({href})" if href else text)
    return "".join(parts)


# 메인 변환기
def blocks_to_markdown(
    blocks: list[dict],
    fetch_children: FetchChildren, # 블록 id를 주면 하위 블록 목록을 돌려주는 함수
    max_depth: int,
    depth: int = 0,
    indent: str = "",
) -> str:
    lines: list[str] = []
    numbered_index = 0
    for block in blocks:
        btype = block.get("type", "")
        if btype != "numbered_list_item":
            numbered_index = 0

        data = block.get(btype, {}) or {}
        text = rich_text(data.get("rich_text", []))

        if btype == "paragraph":
            lines.append(f"{indent}{text}")
        elif btype in ("heading_1", "heading_2", "heading_3"):
            level = int(btype[-1])
            lines.append(f"{indent}{'#' * level} {text}")
        elif btype == "bulleted_list_item":
            lines.append(f"{indent}- {text}")
        elif btype == "numbered_list_item":
            numbered_index += 1
            lines.append(f"{indent}{numbered_index}. {text}")
        elif btype == "to_do":
            mark = "x" if data.get("checked") else " "
            lines.append(f"{indent}- [{mark}] {text}")
        elif btype == "toggle":
            lines.append(f"{indent}▸ {text}")
        elif btype == "code":
            lang = data.get("language", "")
            lines.append(f"{indent}```{lang}\n{text}\n{indent}```")
        elif btype == "quote":
            lines.append(f"{indent}> {text}")
        elif btype == "callout":
            lines.append(f"{indent}> 💡 {text}")
        elif btype == "divider":
            lines.append(f"{indent}---")
        elif btype == "table":
            lines.append(_table_to_markdown(block, fetch_children, indent))
            continue  # 하위(table_row)는 이미 처리했다
        elif btype == "child_page":
            lines.append(f"{indent}📄 하위 페이지: {data.get('title', '')} (id: {block.get('id')})")
        elif btype == "child_database":
            lines.append(f"{indent}🗃 하위 데이터베이스: {data.get('title', '')} (id: {block.get('id')})")
        elif btype in ("image", "video", "file", "pdf", "embed", "bookmark"):
            url = (data.get("external") or {}).get("url") or (data.get("file") or {}).get("url") or data.get("url", "")
            lines.append(f"{indent}[{btype}] {url}".rstrip())
        elif text:
            lines.append(f"{indent}{text}")

        if block.get("has_children") and btype not in ("child_page", "child_database"):
            if depth < max_depth:
                children = fetch_children(block["id"])
                child_md = blocks_to_markdown(children, fetch_children, max_depth, depth + 1, indent + "  ")
                if child_md:
                    lines.append(child_md)
            else:
                lines.append(f"{indent}  … (깊이 제한 — list_children(\"{block.get('id')}\")로 확장 가능)")

    return "\n".join(line for line in lines if line != "")


# 표 변환
def _table_to_markdown(table_block: dict, fetch_children: FetchChildren, indent: str) -> str:
    rows = fetch_children(table_block["id"])
    lines = []
    for i, row in enumerate(rows):
        cells = (row.get("table_row") or {}).get("cells", [])
        rendered = [rich_text(cell) for cell in cells]
        lines.append(f"{indent}| " + " | ".join(rendered) + " |")
        if i == 0:
            lines.append(f"{indent}|" + "---|" * len(rendered))
    return "\n".join(lines)


def properties_to_text(properties: dict) -> str:
    """DB 행 속성을 '이름: 값' 문장으로 편다 (속성 문장화)."""
    parts = []
    for name, prop in (properties or {}).items():
        value = _property_value(prop)
        if value:
            parts.append(f"{name}: {value}")
    return " / ".join(parts)


def _property_value(prop: dict) -> str:
    ptype = prop.get("type", "")
    data = prop.get(ptype)
    if data is None:
        return ""
    if ptype in ("title", "rich_text"):
        return rich_text(data)
    if ptype == "select":
        return data.get("name", "")
    if ptype == "multi_select":
        return ", ".join(o.get("name", "") for o in data)
    if ptype == "status":
        return data.get("name", "")
    if ptype == "date":
        start, end = data.get("start", ""), data.get("end")
        return f"{start}~{end}" if end else start
    if ptype == "people":
        return ", ".join(p.get("name", "") for p in data)
    if ptype in ("number", "checkbox", "url", "email", "phone_number"):
        return str(data)
    if ptype == "relation":
        return f"{len(data)}건 연결"
    return ""
