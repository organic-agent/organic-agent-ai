"""블록 → 마크다운 변환기 테스트 — 네트워크 없이 순수 함수만 검증한다."""

from notionchat.api.notion.blocks import blocks_to_markdown, properties_to_text, rich_text


def _text(content: str, href: str | None = None) -> dict:
    return {"plain_text": content, "href": href}


def _no_children(_block_id: str) -> list[dict]:
    return []


def test_rich_text_with_link():
    assert rich_text([_text("계획 "), _text("문서", href="https://notion.so/x")]) == "계획 [문서](https://notion.so/x)"


def test_basic_blocks():
    blocks = [
        {"type": "heading_1", "heading_1": {"rich_text": [_text("제목")]}},
        {"type": "paragraph", "paragraph": {"rich_text": [_text("본문")]}},
        {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [_text("항목")]}},
        {"type": "code", "code": {"rich_text": [_text("print(1)")], "language": "python"}},
    ]
    md = blocks_to_markdown(blocks, _no_children, max_depth=3)
    assert md == "# 제목\n본문\n- 항목\n```python\nprint(1)\n```"


def test_numbered_list_resets_after_other_block():
    blocks = [
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [_text("하나")]}},
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [_text("둘")]}},
        {"type": "paragraph", "paragraph": {"rich_text": [_text("사이")]}},
        {"type": "numbered_list_item", "numbered_list_item": {"rich_text": [_text("다시 하나")]}},
    ]
    md = blocks_to_markdown(blocks, _no_children, max_depth=3)
    assert md == "1. 하나\n2. 둘\n사이\n1. 다시 하나"


def test_image_placeholder():
    blocks = [{"type": "image", "image": {"external": {"url": "https://img.example/a.png"}}}]
    md = blocks_to_markdown(blocks, _no_children, max_depth=3)
    assert md == "[image] https://img.example/a.png"


def test_depth_limit_annotates_truncation():
    deep = {
        "id": "child-1",
        "type": "toggle",
        "toggle": {"rich_text": [_text("펼치기")]},
        "has_children": True,
    }
    md = blocks_to_markdown([deep], _no_children, max_depth=0)
    assert "깊이 제한" in md
    assert "child-1" in md


def test_recursion_into_children():
    parent = {
        "id": "p1",
        "type": "toggle",
        "toggle": {"rich_text": [_text("부모")]},
        "has_children": True,
    }
    child = {"type": "paragraph", "paragraph": {"rich_text": [_text("자식")]}}

    def fetch(block_id: str) -> list[dict]:
        return [child] if block_id == "p1" else []

    md = blocks_to_markdown([parent], fetch, max_depth=3)
    assert md == "▸ 부모\n  자식"


def test_table_rendering():
    table = {"id": "t1", "type": "table", "table": {}, "has_children": True}
    rows = [
        {"type": "table_row", "table_row": {"cells": [[_text("이름")], [_text("값")]]}},
        {"type": "table_row", "table_row": {"cells": [[_text("리전")], [_text("ap-northeast-2")]]}},
    ]

    def fetch(block_id: str) -> list[dict]:
        return rows if block_id == "t1" else []

    md = blocks_to_markdown([table], fetch, max_depth=3)
    assert md == "| 이름 | 값 |\n|---|---|\n| 리전 | ap-northeast-2 |"


def test_properties_to_text():
    props = {
        "이름": {"type": "title", "title": [_text("보정 요청 정리")]},
        "상태": {"type": "select", "select": {"name": "확정"}},
        "태그": {"type": "multi_select", "multi_select": [{"name": "P2"}, {"name": "LLM"}]},
        "마감": {"type": "date", "date": {"start": "2026-08-20", "end": None}},
        "빈값": {"type": "rich_text", "rich_text": []},
    }
    text = properties_to_text(props)
    assert "이름: 보정 요청 정리" in text
    assert "상태: 확정" in text
    assert "태그: P2, LLM" in text
    assert "마감: 2026-08-20" in text
    assert "빈값" not in text
