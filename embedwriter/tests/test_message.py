"""계약 v1 읽기 — 짝은 embedder repository/results.py. 모양이 틀린 메시지는 SQL 전에 InvalidMessage."""

from __future__ import annotations

import json
import struct
import unittest
from datetime import datetime

from helpers import DIM, body, photo, vector_b64

from embedwriter import message


class ParseTest(unittest.TestCase):
    def test_parses_photo_and_metadata(self) -> None:
        rows = message.parse(body(photo(101, "galleries/7/a.jpg")), DIM)

        row = rows[0]
        self.assertEqual((101, "galleries/7/a.jpg", "previews/galleries/7/a.jpg", "dinov3"),
                         (row.photo_id, row.storage_key, row.preview_key, row.embedding_model))
        self.assertEqual("[0.5,-0.25,1,0]", row.embedding)
        self.assertEqual(datetime(2026, 10, 1, 3, 4, 5), row.taken_at)
        self.assertEqual(("Canon", "R5", "1/200", 2.8, 400, 8192, 5464, 12_345_678),
                         (row.camera_make, row.camera_model, row.exposure_time, row.f_number, row.iso,
                          row.width, row.height, row.byte_size))

    def test_null_metadata_leaves_exif_none(self) -> None:
        row = message.parse(body(photo(1, metadata=None)), DIM)[0]
        self.assertIsNone(row.taken_at)
        self.assertIsNone(row.byte_size)

    def test_taken_at_offset_is_dropped_to_wall_clock(self) -> None:
        meta = {"takenAt": "2026-10-01T03:04:05+09:00"}
        self.assertEqual(datetime(2026, 10, 1, 3, 4, 5), message.parse(body(photo(1, metadata=meta)), DIM)[0].taken_at)

    def test_float32_roundtrips_through_text(self) -> None:
        values = [0.1, 1 / 3, -123.456789, 1e-8]
        row = message.parse(body(photo(1, values=values)), DIM)[0]
        parsed = [float(x) for x in row.embedding.strip("[]").split(",")]
        # 텍스트를 float32 로 되읽으면 원래 float32 비트와 같다(pgvector 가 float4 로 읽는 것과 같은 왕복).
        self.assertEqual(struct.pack(f"<{DIM}f", *values), struct.pack(f"<{DIM}f", *parsed))

    def test_empty_photos_is_valid(self) -> None:
        self.assertEqual([], message.parse(body(), DIM))

    def test_malformed_messages_are_rejected(self) -> None:
        good = photo(1)
        cases = {
            "not json": "{nope",
            "array": "[]",
            "wrong version": body(good, v=2),
            "no version": json.dumps({"galleryId": 7, "embeddingModel": "m", "photos": []}),
            "photos not list": json.dumps({"v": 1, "galleryId": 7, "embeddingModel": "m", "photos": {}}),
            "no model": json.dumps({"v": 1, "galleryId": 7, "photos": []}),
            "string id": body({**good, "photoId": "1"}),
            "bool id": body({**good, "photoId": True}),
            "no storage key": body({**good, "storageKey": None}),
            "bad base64": body({**good, "vector": "!!!"}),
            "wrong dim": body({**good, "vector": vector_b64([1.0, 2.0])}),
            "nan": body({**good, "vector": vector_b64([float("nan"), 0, 0, 0])}),
            "bad takenAt": body({**good, "metadata": {"takenAt": "yesterday"}}),
            "long make": body({**good, "metadata": {"cameraMake": "x" * 101}}),
            "metadata list": body({**good, "metadata": []}),
        }
        for name, text in cases.items():
            with self.subTest(name), self.assertRaises(message.InvalidMessage):
                message.parse(text, DIM)
        with self.assertRaises(message.InvalidMessage):
            message.parse(None, DIM)


if __name__ == "__main__":
    unittest.main()
