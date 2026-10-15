"""장애 주입은 확률 안에서만, 끝내기 전에 표시 줄을 남긴다(wes #286 R-3)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedder.controller import fault


class MaybeExitTest(unittest.TestCase):
    def setUp(self) -> None:
        self.exits: list[int] = []

    def test_rate_zero_never_exits(self) -> None:
        fault.maybe_exit("x", rate=0.0, draw=lambda: 0.0, exit_process=self.exits.append)
        self.assertEqual([], self.exits)

    def test_draw_below_rate_logs_then_exits(self) -> None:
        with self.assertLogs(fault.log, level="WARNING") as logs:
            fault.maybe_exit("gallery=7 photoIds=1,2", rate=0.02, draw=lambda: 0.01, exit_process=self.exits.append)
        self.assertEqual([1], self.exits)
        self.assertEqual(["WARNING:embedder.controller.fault:fault.injected gallery=7 photoIds=1,2"], logs.output)

    def test_draw_at_or_above_rate_passes(self) -> None:
        fault.maybe_exit("x", rate=0.02, draw=lambda: 0.02, exit_process=self.exits.append)
        self.assertEqual([], self.exits)

    def test_unset_env_means_off(self) -> None:
        import os
        saved = os.environ.pop(fault.ENV_NAME, None)
        try:
            fault.maybe_exit("x", draw=lambda: 0.0, exit_process=self.exits.append)
        finally:
            if saved is not None:
                os.environ[fault.ENV_NAME] = saved
        self.assertEqual([], self.exits)
