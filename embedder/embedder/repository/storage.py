"""S3 원본 읽기 · 파생본 쓰기. 여러 스레드에서 동시에 불러도 된다.

Lambda 는 NAT 없이 S3 게이트웨이 VPC 엔드포인트로 나간다. 엔드포인트가 없으면 자격증명 오류가 아니라
타임아웃으로 멈춘다.
"""

from __future__ import annotations

import boto3
from botocore.config import Config


class PhotoStorage:
    def __init__(self, bucket: str, max_concurrency: int = 1) -> None:
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            config=Config(
                # 한두 번 더 기다리는 편이 잡을 통째로 다시 도는 것보다 싸다.
                retries={"max_attempts": 5, "mode": "standard"},
                # 풀이 동시 GET 수보다 작으면 boto 가 연결을 버리고 다시 맺으며 경고를 찍는다.
                max_pool_connections=max(10, max_concurrency + 2),
            ),
        )

    def read(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def write(self, key: str, data: bytes, content_type: str) -> None:
        # content_type 을 비우면 binary/octet-stream 이 나가 브라우저가 이미지 대신 다운로드를 띄운다.
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
