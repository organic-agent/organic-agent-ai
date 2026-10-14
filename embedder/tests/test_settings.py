"""RESULT_QUEUE_URL 스위치(wes #286): 큐 모드는 DB 환경변수 없이 뜨고, 아니면 지금처럼 DB_* 가 필수다."""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest import mock

EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import db_fakes

db_fakes.install_stubs()

from embedder.config.settings import Settings
from embedder.repository import connection

_DB = {"DB_HOST": "db", "DB_NAME": "wes", "DB_USER": "embedder", "DB_PASSWORD": "x"}


class SettingsSwitchTest(unittest.TestCase):
    def test_without_queue_db_env_is_required(self) -> None:
        with mock.patch.dict(os.environ, {"S3_BUCKET": "b"}, clear=True):
            with self.assertRaises(RuntimeError):
                Settings.from_env()
        with mock.patch.dict(os.environ, {"S3_BUCKET": "b", **_DB}, clear=True):
            settings = Settings.from_env()
        self.assertIsNone(settings.result_queue_url)
        self.assertEqual("db", settings.db_host)

    def test_queue_mode_starts_without_db_env(self) -> None:
        with mock.patch.dict(os.environ, {"S3_BUCKET": "b", "RESULT_QUEUE_URL": "https://sqs/q"}, clear=True):
            settings = Settings.from_env()
        self.assertEqual("https://sqs/q", settings.result_queue_url)
        self.assertIsNone(settings.db_host)
        self.assertIsNone(settings.db_password)
        # 큐 모드에서 DB 를 열려 하면 기본 호스트로 붙지 않고 이유를 말하며 죽는다.
        with self.assertRaises(RuntimeError):
            connection.connect(settings)


if __name__ == "__main__":
    unittest.main()
