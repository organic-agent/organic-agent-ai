"""Integration에 연결된 Notion 문서의 ID 목록 — workspace_map.md 작성/갱신용.

실행: notionchat/ 에서  python scripts/list_notion_ids.py
DB 행(개별 회의록 등)은 제외하고 데이터베이스와 상위 페이지만 출력한다.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.notion.client import NotionClient
from api.notion.tools import _title_of


def main() -> None:
    client = NotionClient()
    cursor, rows = None, []
    while True:
        body: dict = {"page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        data = client._request("POST", "/search", json=body)
        rows += data.get("results", [])
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")

    dbs = [r for r in rows if r["object"] == "database"]
    pages = [
        r for r in rows
        if r["object"] == "page" and r.get("parent", {}).get("type") != "database_id"
    ]
    print(f"데이터베이스 {len(dbs)}개:")
    for r in dbs:
        print(f"  - {_title_of(r):40s} database_id: {r['id'].replace('-', '')}")
    print(f"\n상위 페이지 {len(pages)}개 (DB 행 제외):")
    for r in pages:
        print(f"  - {_title_of(r):40s} page_id: {r['id'].replace('-', '')}")


if __name__ == "__main__":
    main()
