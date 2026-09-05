"""S3에서 원본 바이트를 가져오고 파생본을 올린다. 여러 스레드에서 동시에 불러도 된다.

Lambda는 DB 서브넷에 붙어 있고 NAT가 없어서, 이 호출은 전부 S3 게이트웨이 VPC 엔드포인트를
통해 나간다(인프라 레포의 `modules/network`). 엔드포인트가 없으면 여기서 타임아웃으로 멈춘다 --
자격증명 오류처럼 보이지 않으니 헷갈리지 말 것.
"""

from __future__ import annotations

import boto3
from botocore.config import Config


class PhotoStorage:
    def __init__(self, bucket: str, max_concurrency: int = 1) -> None:
        self.bucket = bucket
        self._client = boto3.client(
            "s3",
            # 기본 재시도는 짧다. 갤러리 하나를 순차로 도는 잡이라 한두 번 더 기다리는 편이
            # 통째로 다시 도는 것보다 싸다.
            config=Config(
                retries={"max_attempts": 5, "mode": "standard"},
                # 클라이언트는 스레드 안전하다. 풀이 동시 GET 수보다 작으면 boto가 연결을 버리고
                # 다시 맺으며 경고를 찍는다 -- job의 다운로드 스레드 수 + PUT 하나 몫을 확보한다.
                max_pool_connections=max(10, max_concurrency + 2),
            ),
        )

    def read(self, key: str) -> bytes:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        return response["Body"].read()

    def write(self, key: str, data: bytes, content_type: str) -> None:
        """파생본을 올린다.

        content_type을 못박아야 한다. S3가 응답에 그대로 실어 주는 값이고, 비워 두면
        기본값(binary/octet-stream)이 나가 브라우저가 이미지 대신 다운로드를 띄운다.
        """
        self._client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=data,
            ContentType=content_type,
        )
