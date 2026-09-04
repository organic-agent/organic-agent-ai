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

import embedder as _package

_fake_admin_job = types.ModuleType("embedder.admin_job")
_fake_admin_job.run = lambda event, settings: {"status": "FAKE"}
sys.modules["embedder.admin_job"] = _fake_admin_job
_package.admin_job = _fake_admin_job
try:
    from embedder import handler
finally:
    sys.modules.pop("embedder.admin_job", None)
    delattr(_package, "admin_job")


class _Context:
    function_name = "wes-embedder"

    def __init__(self, remaining_ms: int = 600_000) -> None:
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self) -> int:
        return self.remaining_ms


class HandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._run = handler.job.run
        self._reinvoke = handler.reinvoke
        self.run_calls: list[dict] = []
        self.reinvoked: list[int] = []
        handler.reinvoke = lambda context, gallery_id: (self.reinvoked.append(gallery_id) or True)

    def tearDown(self) -> None:
        handler.job.run = self._run
        handler.reinvoke = self._reinvoke

    def _fake_run(self, **result):
        def run(gallery_id, force, settings, remaining_seconds):
            self.run_calls.append({
                "gallery_id": gallery_id, "force": force,
                "remaining": remaining_seconds() if remaining_seconds else None,
            })
            return {"galleryId": gallery_id, **result}
        handler.job.run = run

    def test_passes_remaining_seconds_from_context(self) -> None:
        self._fake_run(processed=3, stopped=False)

        handler.handler({"galleryId": "7", "force": True}, _Context(remaining_ms=123_000))

        self.assertEqual([{"gallery_id": 7, "force": True, "remaining": 123.0}], self.run_calls)
        self.assertEqual([], self.reinvoked)

    def test_reinvokes_when_stopped_with_progress(self) -> None:
        self._fake_run(processed=8, stopped=True, remaining=40)

        result = handler.handler({"galleryId": 7}, _Context())

        self.assertEqual([7], self.reinvoked)
        self.assertTrue(result["reinvoked"])

    def test_does_not_reinvoke_when_stopped_without_progress(self) -> None:
        # 같은 사진이 계속 실패하는 갤러리에서 무한히 돌지 않는다.
        self._fake_run(processed=0, stopped=True, remaining=40, failed=["a", "b"])

        result = handler.handler({"galleryId": 7}, _Context())

        self.assertEqual([], self.reinvoked)
        self.assertNotIn("reinvoked", result)

    def test_no_context_means_no_deadline(self) -> None:
        self._fake_run(processed=1, stopped=False)

        handler.handler({"galleryId": 7}, None)

        self.assertIsNone(self.run_calls[0]["remaining"])

    def test_reinvoke_without_function_name_returns_false(self) -> None:
        self.assertFalse(self._reinvoke(SimpleNamespace(), 7))


if __name__ == "__main__":
    unittest.main()
