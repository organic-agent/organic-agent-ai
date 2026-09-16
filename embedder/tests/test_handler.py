"""Lambda handler의 데드라인 전달과 자기 재호출 규칙(#24). job.run과 reinvoke는 가짜다."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))

# handler는 import 시점에 Settings.from_env()를 읽는다. 값은 아무거나 -- 접속하지 않는다.
for name, value in {
    "DB_HOST": "localhost", "DB_NAME": "wes", "DB_USER": "embedder", "DB_PASSWORD": "x",
    "S3_BUCKET": "bucket",
}.items():
    os.environ.setdefault(name, value)

# admin_job은 여기서 쓰지 않는데, 진짜를 import해 두면 test_z_admin_job이 가짜 모듈로 다시 import하는
# 트릭이 깨진다(패키지 속성에 이미 잡혀 있어 재import가 안 된다). 가짜를 끼우고 import한 뒤 지운다.
import types

import embedder.service as _package

_fake_admin_job = types.ModuleType("embedder.service.admin_job")
_fake_admin_job.run = lambda event, settings: {"status": "FAKE"}
sys.modules["embedder.service.admin_job"] = _fake_admin_job
_package.admin_job = _fake_admin_job
try:
    from embedder.controller import handler
finally:
    sys.modules.pop("embedder.service.admin_job", None)
    delattr(_package, "admin_job")


class _Context:
    function_name = "wes-embedder"

    def __init__(self, remaining_ms: int = 600_000) -> None:
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self) -> int:
        return self.remaining_ms


class PhotoIdsHandlerTest(unittest.TestCase):
    """v2(#73·#100): 분석 페이로드는 {galleryId, photoIds} 하나다. 재호출·fan-out·갤러리 전수 스캔이 없다."""

    def setUp(self) -> None:
        self._run = handler.job.run
        self.calls: list[dict] = []

        def run(gallery_id, settings, remaining_seconds=None, photo_ids=None, **kwargs):
            self.calls.append({"gallery_id": gallery_id, "photo_ids": photo_ids, "kwargs": kwargs,
                               "remaining": remaining_seconds() if remaining_seconds else None})
            return {"galleryId": gallery_id, "photoIds": len(photo_ids), "processed": len(photo_ids), "stopped": True}
        handler.job.run = run

    def tearDown(self) -> None:
        handler.job.run = self._run

    def test_photo_ids_go_to_job_without_fan_out_or_reinvoke(self) -> None:
        result = handler.handler({"galleryId": "7", "photoIds": [3, "1"]}, _Context(remaining_ms=120_000))
        self.assertEqual(1, len(self.calls))
        self.assertEqual(7, self.calls[0]["gallery_id"])
        self.assertEqual([3, 1], self.calls[0]["photo_ids"])
        self.assertEqual({}, self.calls[0]["kwargs"])          # fan_out·shard·force 는 사라졌다(#100)
        self.assertAlmostEqual(120.0, self.calls[0]["remaining"], places=0)
        self.assertNotIn("reinvoked", result)           # stopped=True 여도 재호출 없음 — 남은 장은 wes 가 재배정

    def test_legacy_gallery_payload_is_rejected(self) -> None:
        """옛 조정자 페이로드(#56)는 조용히 갤러리를 전수 스캔하지 않고 에러다(#100)."""
        for payload in ({"galleryId": 7}, {"galleryId": 7, "force": True},
                        {"galleryId": 7, "shard": {"index": 0, "total": 2}}):
            with self.assertRaises(ValueError):
                handler.handler(payload, _Context())
        with self.assertRaises(ValueError):
            handler.handler({"photoIds": [1]}, _Context())
        self.assertEqual([], self.calls)


if __name__ == "__main__":
    unittest.main()
