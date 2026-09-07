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
        handler.reinvoke = lambda context, gallery_id, **kw: (self.reinvoked.append(gallery_id) or True)

    def tearDown(self) -> None:
        handler.job.run = self._run
        handler.reinvoke = self._reinvoke

    def _fake_run(self, **result):
        def run(gallery_id, force, settings, remaining_seconds, **kwargs):
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

    # ── 샤딩(#56) ──
    def _capture_invocations(self) -> list[dict]:
        payloads: list[dict] = []
        saved = handler._invoke_self
        handler._invoke_self = lambda context, gallery_id, payload: (payloads.append(payload) or True)
        self.addCleanup(lambda: setattr(handler, "_invoke_self", saved))
        return payloads

    def test_coordinator_event_gets_fan_out_that_invokes_self_per_shard(self) -> None:
        payloads = self._capture_invocations()
        seen: dict = {}

        def run(gallery_id, force, settings, remaining_seconds, shard, run_started_at, fan_out):
            seen.update(shard=shard, run_started_at=run_started_at)
            self.assertTrue(fan_out(3, run_started_at))
            return {"galleryId": gallery_id, "coordinator": True, "shards": 3, "stopped": False}
        handler.job.run = run

        result = handler.handler({"galleryId": 7, "force": True}, _Context())

        self.assertIsNone(seen["shard"])
        self.assertIsNotNone(seen["run_started_at"])           # force 라 handler 가 시작 시각을 만든다
        self.assertEqual(
            [{"galleryId": 7, "force": True, "shard": {"index": i, "total": 3}, "runStartedAt": seen["run_started_at"]}
             for i in range(3)],
            payloads,
        )
        self.assertTrue(result["coordinator"])
        self.assertEqual([], self.reinvoked)

    def test_shard_event_passes_shard_and_run_started_at_and_no_fan_out(self) -> None:
        seen: dict = {}

        def run(gallery_id, force, settings, remaining_seconds, shard, run_started_at, fan_out):
            seen.update(shard=shard, run_started_at=run_started_at, fan_out=fan_out, force=force)
            return {"galleryId": gallery_id, "processed": 2, "stopped": False}
        handler.job.run = run

        handler.handler({"galleryId": 7, "force": True, "shard": {"index": 2, "total": 4},
                         "runStartedAt": "2026-09-06T00:00:00+00:00"}, _Context())

        self.assertEqual(handler.job.Shard(2, 4), seen["shard"])
        self.assertEqual("2026-09-06T00:00:00+00:00", seen["run_started_at"])   # 조정자의 시각을 그대로
        self.assertIsNone(seen["fan_out"])                                       # 샤드는 다시 나누지 않는다
        self.assertTrue(seen["force"])

    def test_non_force_event_has_no_run_started_at(self) -> None:
        seen: dict = {}

        def run(gallery_id, force, settings, remaining_seconds, shard, run_started_at, fan_out):
            seen["run_started_at"] = run_started_at
            return {"galleryId": gallery_id, "processed": 1, "stopped": False}
        handler.job.run = run

        handler.handler({"galleryId": 7}, _Context())

        self.assertIsNone(seen["run_started_at"])

    def test_reinvoke_payload_keeps_shard_and_run_started_at_but_not_force(self) -> None:
        payloads = self._capture_invocations()

        ok = self._reinvoke(_Context(), 7, shard=handler.job.Shard(1, 4), run_started_at="2026-09-06T00:00:00+00:00")

        self.assertTrue(ok)
        self.assertEqual(
            [{"galleryId": 7, "force": False, "shard": {"index": 1, "total": 4}, "runStartedAt": "2026-09-06T00:00:00+00:00"}],
            payloads,
        )

    def test_stopped_shard_reinvokes_with_its_shard(self) -> None:
        handler.reinvoke = self._reinvoke
        payloads = self._capture_invocations()
        self._fake_run(processed=8, stopped=True, remaining=40)

        result = handler.handler({"galleryId": 7, "shard": {"index": 1, "total": 4},
                                  "runStartedAt": "2026-09-06T00:00:00+00:00"}, _Context())

        self.assertTrue(result["reinvoked"])
        self.assertEqual({"index": 1, "total": 4}, payloads[0]["shard"])
        self.assertEqual("2026-09-06T00:00:00+00:00", payloads[0]["runStartedAt"])
        self.assertFalse(payloads[0]["force"])


if __name__ == "__main__":
    unittest.main()


class PhotoIdsHandlerTest(unittest.TestCase):
    """v2(#73): photoIds 페이로드는 job.run(photo_ids=…) 로 가고, 재호출·fan-out 을 하지 않는다."""

    def setUp(self) -> None:
        self._run = handler.job.run
        self._reinvoke = handler.reinvoke
        self.calls: list[dict] = []
        handler.reinvoke = lambda *a, **kw: self.fail("photoIds 경로는 재호출하지 않는다")

        def run(gallery_id, settings, remaining_seconds=None, photo_ids=None, **kwargs):
            self.calls.append({"gallery_id": gallery_id, "photo_ids": photo_ids, "kwargs": kwargs,
                               "remaining": remaining_seconds() if remaining_seconds else None})
            return {"galleryId": gallery_id, "photoIds": len(photo_ids), "processed": len(photo_ids), "stopped": True}
        handler.job.run = run

    def tearDown(self) -> None:
        handler.job.run = self._run
        handler.reinvoke = self._reinvoke

    def test_photo_ids_go_to_job_without_fan_out_or_reinvoke(self) -> None:
        result = handler.handler({"galleryId": "7", "photoIds": [3, "1"]}, _Context(remaining_ms=120_000))
        self.assertEqual(1, len(self.calls))
        self.assertEqual(7, self.calls[0]["gallery_id"])
        self.assertEqual([3, 1], self.calls[0]["photo_ids"])
        self.assertNotIn("fan_out", self.calls[0]["kwargs"])
        self.assertNotIn("shard", self.calls[0]["kwargs"])
        self.assertAlmostEqual(120.0, self.calls[0]["remaining"], places=0)
        self.assertNotIn("reinvoked", result)           # stopped=True 여도 재호출 없음 — 남은 장은 wes 가 재배정
