"""ASM 공개 노션(활동비 규정)을 마크다운 스냅샷으로 저장한다.

대상: https://asm-busan.notion.site/ai-sw-17-2026-project-activity-expenses
우리 Integration 권한 밖(외부 워크스페이스)이라 공식 API로 못 읽는다.
공개 페이지 전용 비공식 엔드포인트(loadCachedPageChunkV2/queryCollection)를 쓰므로
Notion이 스펙을 바꾸면 깨질 수 있다 — 그때 이 스크립트만 고치면 된다.

실행: notionchat/ 에서  python scripts/snapshot_asm_expenses.py
출력: api/data/asm_expense_guide.md  (read_expense_guide 도구가 읽는 파일)
규정이 갱신되면 다시 실행해 커밋한다.
"""

from datetime import date
from pathlib import Path

import httpx

SITE = "https://asm-busan.notion.site"
PAGE_URL = f"{SITE}/ai-sw-17-2026-project-activity-expenses"
PAGE_ID = "390a01ba-dc21-8076-9ec0-d77bfeac8745"
OUT = Path(__file__).resolve().parent.parent / "api" / "data" / "asm_expense_guide.md"

_http = httpx.Client(timeout=30.0, headers={"Content-Type": "application/json"})


def _rec(entry: dict) -> dict:
    v = entry.get("value", {})
    return v.get("value", v)


def _rich(segments: list | None) -> str:
    """비공식 포맷의 rich text: [["텍스트", [["b"],["a","url"],...]], ...]"""
    if not segments:
        return ""
    parts = []
    for seg in segments:
        text = seg[0] if seg else ""
        decos = seg[1] if len(seg) > 1 else []
        for d in decos:
            if d and d[0] == "a" and len(d) > 1:
                text = f"[{text}]({d[1]})"
            elif d and d[0] == "d" and len(d) > 1 and isinstance(d[1], dict):
                start = d[1].get("start_date", "")
                end = d[1].get("end_date")
                text = f"{start}~{end}" if end else start
        parts.append(text)
    return "".join(parts)


def load_page(page_id: str) -> dict:
    """페이지 전체 블록 recordMap을 커서 페이지네이션으로 수집한다."""
    blocks: dict = {}
    collections: dict = {}
    cursor = {"stack": []}
    while True:
        resp = _http.post(
            f"{SITE}/api/v3/loadCachedPageChunkV2",
            json={"page": {"id": page_id}, "limit": 100, "cursor": cursor, "verticalColumns": False},
        )
        resp.raise_for_status()
        data = resp.json()
        rm = data.get("recordMap", {})
        blocks.update(rm.get("block", {}))
        collections.update(rm.get("collection", {}))
        cursor = data.get("cursor")
        if not cursor or not cursor.get("stack"):
            return {"block": blocks, "collection": collections}


def query_collection(space_id: str, collection_id: str, view_id: str) -> tuple[list, dict]:
    """공개 collection의 행 목록과 recordMap을 반환한다."""
    resp = _http.post(
        f"{SITE}/api/v3/queryCollection?src=initial_load",
        json={
            "source": {"type": "collection", "id": collection_id, "spaceId": space_id},
            "collectionView": {"id": view_id, "spaceId": space_id},
            "loader": {
                "reducers": {"collection_group_results": {"type": "results", "limit": 200}},
                "searchQuery": "",
                "userTimeZone": "Asia/Seoul",
            },
        },
    )
    resp.raise_for_status()
    data = resp.json()
    row_ids = (
        data.get("result", {}).get("reducerResults", {}).get("collection_group_results", {}).get("blockIds", [])
    )
    return row_ids, data.get("recordMap", {})


def render_collection(record_map: dict, block: dict) -> list[str]:
    cid = block.get("collection_id") or block.get("format", {}).get("collection_pointer", {}).get("id")
    view_ids = block.get("view_ids") or []
    space_id = block.get("space_id")
    if not cid or not view_ids:
        return ["(내장 데이터베이스 — 내용 생략)"]
    coll = _rec(record_map.get("collection", {}).get(cid, {}))
    name = _rich(coll.get("name"))
    schema = coll.get("schema", {})
    row_ids, rm = query_collection(space_id, cid, view_ids[0])

    lines = [f"#### {name}" if name else "#### (무제 DB)"]
    # 각 행은 본문을 가진 내부 페이지 — 제목만이 아니라 페이지 내용까지 내려가 추출한다
    for rid in row_ids:
        row = _rec(rm.get("block", {}).get(rid, {}))
        title = _rich((row.get("properties") or {}).get("title")) or "(무제)"
        lines.append(f"\n##### {title}")
        row_map = load_page(rid)
        row_map["collection"] = {**record_map.get("collection", {}), **row_map.get("collection", {})}
        body = render_block(row_map, rid)
        lines += body if body else ["(본문 없음)"]
    return lines


def render_block(record_map: dict, block_id: str, depth: int = 0, number: list | None = None) -> list[str]:
    entry = record_map["block"].get(block_id)
    if not entry:
        return []
    b = _rec(entry)
    btype = b.get("type", "")
    text = _rich((b.get("properties") or {}).get("title"))
    indent = "  " * depth
    lines: list[str] = []

    def children(extra_depth: int = 0) -> list[str]:
        out: list[str] = []
        num: list = [0]
        for cid in b.get("content", []):
            out += render_block(record_map, cid, depth + extra_depth, num)
        return out

    if btype == "page":
        lines += children()
    elif btype in ("header", "sub_header", "sub_sub_header"):
        level = {"header": "##", "sub_header": "###", "sub_sub_header": "####"}[btype]
        lines.append(f"{level} {text}")
    elif btype == "text":
        if text:
            lines.append(f"{indent}{text}")
    elif btype == "bulleted_list":
        lines.append(f"{indent}- {text}")
        lines += children(1)
    elif btype == "numbered_list":
        if number is not None:
            number[0] += 1
            lines.append(f"{indent}{number[0]}. {text}")
        else:
            lines.append(f"{indent}1. {text}")
        lines += children(1)
    elif btype == "toggle":
        lines.append(f"{indent}▸ {text}")
        lines += children(1)
    elif btype == "callout":
        lines.append(f"{indent}> {text}")
        lines += children(1)
    elif btype == "quote":
        lines.append(f"{indent}> {text}")
    elif btype == "code":
        lines.append(f"```\n{text}\n```")
    elif btype == "divider":
        lines.append("---")
    elif btype == "image":
        caption = _rich((b.get("properties") or {}).get("caption"))
        lines.append(f"(이미지{': ' + caption if caption else ''})")
    elif btype == "file":
        lines.append(f"(첨부파일: {text})")
    elif btype in ("column_list", "column"):
        lines += children()
    elif btype == "collection_view":
        lines += render_collection(record_map, b)
    elif btype == "table":
        order = b.get("format", {}).get("table_block_column_order", [])
        rows = []
        for cid in b.get("content", []):
            r = _rec(record_map["block"].get(cid, {}))
            props = r.get("properties", {})
            rows.append([_rich(props.get(col)).replace("|", "\\|") for col in order])
        if rows:
            lines.append("| " + " | ".join(rows[0]) + " |")
            lines.append("|" + "---|" * len(order))
            for row in rows[1:]:
                lines.append("| " + " | ".join(row) + " |")
    elif btype == "table_of_contents":
        pass
    else:
        if text:
            lines.append(f"{indent}{text}")
    return lines


def main() -> None:
    record_map = load_page(PAGE_ID)
    page = _rec(record_map["block"][PAGE_ID])
    title = _rich((page.get("properties") or {}).get("title")) or "ASM 활동비 규정"
    body = "\n".join(render_block(record_map, PAGE_ID))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        f"# {title}\n\n"
        f"> 출처: {PAGE_URL}\n"
        f"> 스냅샷: {date.today().isoformat()} — 원본이 갱신됐을 수 있으니 최신 여부는 출처 링크로 확인.\n\n"
        f"{body}\n"
    )
    print(f"저장: {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
