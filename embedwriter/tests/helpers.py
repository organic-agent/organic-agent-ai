"""테스트 공용 — 계약 v1 메시지 만들기와 가짜 psycopg 커넥션. 네트워크·DB 를 쓰지 않는다."""

from __future__ import annotations

import base64
import json
import os
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# handler 는 import 시점에 Settings.from_env() 를 읽는다. 값은 아무거나 -- 테스트는 접속하지 않는다.
for name, value in {"DB_HOST": "localhost", "DB_NAME": "wes", "DB_USER": "embedder", "DB_PASSWORD": "x"}.items():
    os.environ.setdefault(name, value)

DIM = 4


def vector_b64(values) -> str:
    return base64.b64encode(struct.pack(f"<{len(values)}f", *values)).decode("ascii")


def photo(photo_id: int, key: str | None = None, values=None, metadata="default") -> dict:
    key = key or f"galleries/7/{photo_id}.jpg"
    if metadata == "default":
        metadata = {"takenAt": "2026-10-01T03:04:05", "cameraMake": "Canon", "cameraModel": "R5",
                    "exposureTime": "1/200", "fNumber": 2.8, "iso": 400, "width": 8192, "height": 5464,
                    "byteSize": 12_345_678}
    return {"photoId": photo_id, "storageKey": key, "previewKey": f"previews/{key}",
            "vector": vector_b64(values or [0.5, -0.25, 1.0, 0.0]), "metadata": metadata}


def body(*photos: dict, v: int = 1, gallery_id: int = 7, model: str = "dinov3") -> str:
    return json.dumps({"v": v, "galleryId": gallery_id, "embeddingModel": model, "photos": list(photos)})


def record(message_id: str, message_body: str) -> dict:
    return {"messageId": message_id, "body": message_body, "eventSource": "aws:sqs"}


class FakeCursor:
    def __init__(self, connection: "FakeConnection") -> None:
        self.connection = connection
        self.rowcount = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def execute(self, sql: str, params: dict) -> None:
        self.connection.executed.append((sql, params))
        if self.connection.fail_when(params):
            self.connection.in_error = True
            raise RuntimeError("DB 오류 흉내")
        self.rowcount = len(params["photo_id"])


class FakeConnection:
    """`fail_when(params)` 이 참이면 execute 가 실패한다. commit·rollback 을 센다."""

    def __init__(self, fail_when=lambda params: False) -> None:
        self.fail_when = fail_when
        self.executed: list[tuple[str, dict]] = []
        self.commits = 0
        self.rollbacks = 0
        self.closed = False
        self.broken = False
        self.in_error = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self)

    def commit(self) -> None:
        assert not self.in_error, "실패한 트랜잭션을 commit 했다"
        self.commits += 1

    def rollback(self) -> None:
        self.in_error = False
        self.rollbacks += 1

    def close(self) -> None:
        self.closed = True
