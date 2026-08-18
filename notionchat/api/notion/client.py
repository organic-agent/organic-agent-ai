"""Notion API 클라이언트 — 사전 적재 없이 조회 시점에 호출한다.

rate limit(429)은 Retry-After 기반 backoff만 기본 탑재 (docs/notion-chatbot-plan.md §4).
"""

import time

import httpx

from api import config

_MAX_RETRIES = 3


class NotionAPIError(Exception):
    pass


class NotionClient:
    def __init__(self, token: str | None = None):
        self._http = httpx.Client(
            base_url=config.NOTION_API_BASE,
            headers={
                "Authorization": f"Bearer {token or config.notion_token()}",
                "Notion-Version": config.NOTION_VERSION,
            },
            timeout=30.0,
        )

    def _request(self, method: str, path: str, json: dict | None = None, params: dict | None = None) -> dict:
        for attempt in range(_MAX_RETRIES + 1):
            resp = self._http.request(method, path, json=json, params=params)
            if resp.status_code == 429 and attempt < _MAX_RETRIES:
                time.sleep(float(resp.headers.get("Retry-After", 1)) + 0.1)
                continue
            if resp.is_error:
                raise NotionAPIError(f"{method} {path} -> {resp.status_code}: {resp.text[:300]}")
            return resp.json()
        raise NotionAPIError(f"{method} {path}: rate limit 재시도 소진")

    # --- 도구 4종이 쓰는 원시 호출 ---

    def search(self, query: str, page_size: int = 10) -> dict:
        return self._request("POST", "/search", json={"query": query, "page_size": page_size})

    def page_meta(self, page_id: str) -> dict:
        return self._request("GET", f"/pages/{page_id}")

    def block_children(self, block_id: str, start_cursor: str | None = None) -> dict:
        params = {"page_size": 100}
        if start_cursor:
            params["start_cursor"] = start_cursor
        return self._request("GET", f"/blocks/{block_id}/children", params=params)

    def all_block_children(self, block_id: str) -> list[dict]:
        """페이지네이션을 따라가며 하위 블록 전체를 반환한다."""
        blocks: list[dict] = []
        cursor = None
        while True:
            data = self.block_children(block_id, start_cursor=cursor)
            blocks.extend(data.get("results", []))
            if not data.get("has_more"):
                return blocks
            cursor = data.get("next_cursor")

    def query_database(self, database_id: str, filter: dict | None = None, page_size: int = 20) -> dict:
        body: dict = {"page_size": page_size}
        if filter:
            body["filter"] = filter
        return self._request("POST", f"/databases/{database_id}/query", json=body)
