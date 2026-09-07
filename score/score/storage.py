"""S3 미리보기 읽기. 이 서버는 원본을 열지 않는다 — embedder가 만든 파생 JPEG만 내려받는다."""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.config import Config


class PreviewStorage:
    def __init__(self, bucket: str) -> None:
        self.bucket = bucket
        # 커넥션 풀은 다운로드 스레드(SCORE_DOWNLOAD_WORKERS, 워커 16)보다 커야 "pool is full" 경고 없이 병렬이 산다(#81).
        self._client = boto3.client("s3", config=Config(retries={"max_attempts": 5, "mode": "standard"},
                                                        max_pool_connections=32))

    def download(self, key: str, dest: Path) -> Path:
        """이미 있으면 다시 받지 않는다 — 재실행·--force 때 갤러리를 통째로 다시 내려받지 않게."""
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(dest))
        return dest
