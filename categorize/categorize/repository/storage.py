"""S3 미리보기 읽기. 이 서버는 원본을 열지 않는다 — embedder가 만든 파생 JPEG만 내려받는다."""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.config import Config


class PreviewStorage:
    def __init__(self, bucket: str, max_concurrency: int = 1) -> None:
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"},
                # 풀이 동시 GET 수보다 작으면 urllib3 가 "Connection pool is full" 을 찍으며 매 요청 연결을 새로 맺는다(#81 과 같은 증상).
                # store.PREVIEW_DOWNLOAD_WORKERS 를 그대로 받는다 -- 상수 둘이 따로 놀지 않게(#109).
                max_pool_connections=max(10, max_concurrency),
            ),
        )

    def download(self, key: str, dest: Path) -> Path:
        """이미 있으면 다시 받지 않는다 — 재실행·--force 때 갤러리를 통째로 다시 내려받지 않게."""
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(dest))
        return dest
