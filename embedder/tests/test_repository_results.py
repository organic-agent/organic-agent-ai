"""결과 큐 메시지 계약 v1(wes #286) — writer(`embedwriter/`)가 이 모양을 읽는다. 네트워크는 쓰지 않는다."""

from __future__ import annotations

import base64
import json
import sys
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))

from embedder.domain.photo import EmbeddingResult, PhotoMetadata, PhotoRef
from embedder.repository import results


def _decode(vector: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(vector), dtype="<f4")


class _Sqs:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def send_message(self, **kwargs) -> dict:
        self.calls.append(kwargs)
        return {"MessageId": "m-1"}


class EncodeMessageTest(unittest.TestCase):
    def test_message_shape_and_vector_roundtrip(self) -> None:
        vector = np.random.default_rng(0).standard_normal(768).astype(np.float32)
        meta = PhotoMetadata(taken_at=datetime(2026, 10, 1, 3, 4, 5), camera_make="Canon", camera_model="R5",
                             exposure_time="1/200", f_number=2.8, iso=400, width=8192, height=5464, byte_size=12_345_678)
        rows = [
            EmbeddingResult(PhotoRef(101, "galleries/12/a.jpg"), vector, "previews/galleries/12/a.jpg", meta),
            EmbeddingResult(PhotoRef(102, "galleries/12/b.jpg"), vector * 2, "previews/galleries/12/b.jpg", None),
        ]

        body = results.encode_message(12, rows, "facebook/dinov3-vitb16-pretrain-lvd1689m")
        # JSON 왕복 뒤에도 같아야 한다(실제로는 문자열로 간다).
        body = json.loads(json.dumps(body))

        self.assertEqual(1, body["v"])
        self.assertEqual(12, body["galleryId"])
        self.assertEqual("facebook/dinov3-vitb16-pretrain-lvd1689m", body["embeddingModel"])
        first, second = body["photos"]
        self.assertEqual({"photoId", "storageKey", "previewKey", "vector", "metadata"}, set(first))
        self.assertEqual((101, "galleries/12/a.jpg", "previews/galleries/12/a.jpg"),
                         (first["photoId"], first["storageKey"], first["previewKey"]))
        np.testing.assert_array_equal(vector, _decode(first["vector"]))
        np.testing.assert_array_equal(vector * 2, _decode(second["vector"]))
        self.assertEqual(768 * 4, len(base64.b64decode(first["vector"])))
        self.assertEqual({
            "takenAt": "2026-10-01T03:04:05", "cameraMake": "Canon", "cameraModel": "R5", "exposureTime": "1/200",
            "fNumber": 2.8, "iso": 400, "width": 8192, "height": 5464, "byteSize": 12_345_678,
        }, first["metadata"])
        self.assertIsNone(second["metadata"])

    def test_float64_and_list_vectors_are_sent_as_float32(self) -> None:
        self.assertEqual([0.5, -1.0], _decode(results.encode_vector([0.5, -1.0])).tolist())
        self.assertEqual(8, len(base64.b64decode(results.encode_vector(np.array([0.5, -1.0], dtype=np.float64)))))

    def test_metadata_with_missing_taken_at(self) -> None:
        self.assertIsNone(results.encode_metadata(PhotoMetadata(width=10))["takenAt"])


class ResultQueueSendTest(unittest.TestCase):
    def setUp(self) -> None:
        self._client = results.boto3.client
        self.sqs = _Sqs()
        results.boto3.client = lambda service, config=None: self.sqs

    def tearDown(self) -> None:
        results.boto3.client = self._client

    def test_send_puts_one_compact_json_message(self) -> None:
        queue = results.ResultQueue("https://sqs.ap-northeast-2.amazonaws.com/1/wes-embed-results")
        rows = [EmbeddingResult(PhotoRef(1, "galleries/7/a.jpg"), np.ones(4, dtype=np.float32), "previews/galleries/7/a.jpg", None)]

        sent = queue.send(7, rows, "m")

        self.assertEqual(1, sent)
        self.assertEqual(1, len(self.sqs.calls))
        call = self.sqs.calls[0]
        self.assertEqual("https://sqs.ap-northeast-2.amazonaws.com/1/wes-embed-results", call["QueueUrl"])
        body = json.loads(call["MessageBody"])
        self.assertEqual([1], [p["photoId"] for p in body["photos"]])
        self.assertNotIn(" ", call["MessageBody"])

    def test_empty_batch_sends_nothing(self) -> None:
        self.assertEqual(0, results.ResultQueue("q").send(7, [], "m"))
        self.assertEqual([], self.sqs.calls)


if __name__ == "__main__":
    unittest.main()
