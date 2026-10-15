"""임베딩 결과를 SQS 결과 큐로 보낸다 — 결과 큐 모드(`RESULT_QUEUE_URL`, wes #286).

DB 에 바로 쓰는 `photos.store_embeddings` 의 자리를 대신한다. 모델 배치 하나 = 메시지 하나(8장 ≈ 35KB)이고,
writer Lambda(`embedwriter/`, VPC 안, 동시 실행 ≤ 2)가 메시지 수백 장을 한 트랜잭션으로 적재한다. 임베더 72개가
각자 8장마다 INSERT·UPDATE 하던 것이 DB 시간의 57% 를 먹어 사용자 요청을 굶겼기 때문이다.

메시지 모양은 계약 v1 이다(writer 의 `embedwriter/message.py` 와 짝):

    {"v": 1, "galleryId": 12, "embeddingModel": "...",
     "photos": [{"photoId", "storageKey", "previewKey", "vector": base64(little-endian float32), "metadata": {...} | null}]}

여러 스레드에서 부르지 않는다(job 메인 스레드만 보낸다).
"""

from __future__ import annotations

import base64
import json
from typing import Iterable

import boto3
import numpy as np
from botocore.config import Config

from embedder.domain.photo import EmbeddingResult, PhotoMetadata

#: 메시지 계약 버전. writer 가 모르는 버전은 실패로 돌려 DLQ 로 보낸다.
MESSAGE_VERSION = 1


class ResultQueue:
    def __init__(self, queue_url: str) -> None:
        self.queue_url = queue_url
        self._client = boto3.client(
            "sqs",
            # 일시 오류에 한두 번 더 기다리는 편이 배치 사진을 통째로 실패시키고 wes 재배정을 기다리는 것보다 싸다.
            config=Config(retries={"max_attempts": 5, "mode": "standard"}),
        )

    def send(self, gallery_id: int, results: Iterable[EmbeddingResult], embedding_model: str) -> int:
        """배치 결과를 메시지 하나로 보낸다. 보낸 장수를 돌려준다. 실패는 예외로 — 부르는 쪽이 그 배치를 실패로 센다."""
        body = encode_message(gallery_id, results, embedding_model)
        if not body["photos"]:
            return 0
        self._client.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(body, ensure_ascii=False, separators=(",", ":")),
        )
        return len(body["photos"])


def encode_message(gallery_id: int, results: Iterable[EmbeddingResult], embedding_model: str) -> dict:
    return {
        "v": MESSAGE_VERSION,
        "galleryId": gallery_id,
        "embeddingModel": embedding_model,
        "photos": [
            {
                "photoId": result.ref.photo_id,
                "storageKey": result.ref.storage_key,
                "previewKey": result.preview_key,
                "vector": encode_vector(result.vector),
                "metadata": encode_metadata(result.metadata),
            }
            for result in results
        ],
    }


def encode_vector(vector) -> str:
    """little-endian float32 바이트의 base64. JSON 숫자 배열보다 ~3배 작고 소수점 왕복 손실이 없다."""
    return base64.b64encode(np.asarray(vector, dtype="<f4").tobytes()).decode("ascii")


def encode_metadata(meta: PhotoMetadata | None) -> dict | None:
    """photos 의 EXIF 컬럼과 1:1. None(추출 실패)이면 writer 가 COALESCE 로 기존 값을 지킨다."""
    if meta is None:
        return None
    return {
        # EXIF 는 타임존 없는 벽시계라 오프셋 없이 나간다. 컬럼도 timestamp without time zone 이다.
        "takenAt": meta.taken_at.isoformat() if meta.taken_at is not None else None,
        "cameraMake": meta.camera_make,
        "cameraModel": meta.camera_model,
        "exposureTime": meta.exposure_time,
        "fNumber": meta.f_number,
        "iso": meta.iso,
        "width": meta.width,
        "height": meta.height,
        "byteSize": meta.byte_size,
    }
