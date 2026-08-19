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
        self._http = httpx.Client( # 커넥션을 재사용하는 HTTP 세션, TCP/TLS 을 매번 새로 연결하지 않는다.
            base_url=config.NOTION_API_BASE,
            headers={
                "Authorization": f"Bearer {token or config.notion_token()}",
                "Notion-Version": config.NOTION_VERSION,
            },
            timeout=30.0,
        )

    # 공통 요청 처리 — 조회 전용 가드레일: 쓰기 계열 메서드와 미등록 POST 경로를 차단한다.
    # 이 챗봇은 읽기 전용이며, 토큰에 쓰기 권한이 있어도 코드 레벨에서 막는다.
    _READONLY_POST_PREFIXES = ("/search", "/databases/")  # search, databases/{id}/query 뿐

    def _request(self, method: str, path: str, json: dict | None = None, params: dict | None = None) -> dict:
        if method not in ("GET", "POST"):
            raise NotionAPIError(f"조회 전용 클라이언트 — {method} 요청은 허용되지 않는다")
        if method == "POST" and not path.startswith(self._READONLY_POST_PREFIXES):
            raise NotionAPIError(f"조회 전용 클라이언트 — POST {path}는 허용되지 않는다")
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

    # 노션은 한 번에 최대 100개의 블록만 주고, 더 있으면 has_more: true 와 next_cursor 를 준다.
    def block_children(self, block_id: str, start_cursor: str | None = None) -> dict:
        params = {"page_size": 100}
        if start_cursor:
            params["start_cursor"] = start_cursor
        return self._request("GET", f"/blocks/{block_id}/children", params=params)

    # has_more 가 false 가 될 때까지 next_cursor 를 넘기며 반복해서 전체를 모은다.
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

    # 노션 데이터베이스의 행들을 조회하는 메서드.
    def query_database(self, database_id: str, filter: dict | None = None, page_size: int = 20) -> dict:
        body: dict = {"page_size": page_size}
        if filter:
            body["filter"] = filter
        return self._request("POST", f"/databases/{database_id}/query", json=body)
