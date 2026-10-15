"""DB 연결 상한에 막혔을 때 잠깐 쉬었다 다시 여는지(wes #274 R-1 E-3).

psycopg.connect 를 가짜로 바꿔 끼우고, 쉬는 함수(sleep)는 넘겨받은 시간을 모으기만 한다 — 실제로 기다리지 않는다.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import db_fakes

db_fakes.install_stubs()

import psycopg

from embedder.repository import connection

ROLE_LIMIT = 'connection failed: connection to server at "10.0.10.188", port 5432 failed: FATAL:  too many connections for role "embedder"'
SERVER_LIMIT = "connection failed: FATAL:  sorry, too many clients already"
WRONG_PASSWORD = 'connection failed: FATAL:  password authentication failed for user "embedder"'


def _settings():
    return SimpleNamespace(
        db_host="db", db_port=5432, db_name="wes", db_user="embedder", db_password="pw",
        db_sslmode="require", db_sslrootcert="/ca.pem",
    )


class ConnectRetryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = (connection.psycopg.connect, connection.register_vector)
        self.attempts = 0
        self.registered: list[object] = []
        self.sleeps: list[float] = []
        connection.register_vector = self.registered.append

    def tearDown(self) -> None:
        connection.psycopg.connect, connection.register_vector = self._saved

    def _fail_then_succeed(self, *messages: str) -> object:
        opened = object()

        def fake_connect(**kwargs):
            self.attempts += 1
            if self.attempts <= len(messages):
                raise psycopg.OperationalError(messages[self.attempts - 1])
            return opened

        connection.psycopg.connect = fake_connect
        return opened

    def test_retries_when_role_or_server_limit_is_hit(self) -> None:
        opened = self._fail_then_succeed(ROLE_LIMIT, SERVER_LIMIT)

        result = connection.connect(_settings(), sleep=self.sleeps.append)

        self.assertIs(opened, result)
        self.assertEqual(3, self.attempts)
        self.assertEqual([opened], self.registered)   # 벡터 타입 등록은 열린 연결에 한 번만
        # 쉬는 시간은 0.5·1초에 지터 ×0.5~1.5
        self.assertEqual(2, len(self.sleeps))
        self.assertTrue(0.25 <= self.sleeps[0] <= 0.75, self.sleeps)
        self.assertTrue(0.5 <= self.sleeps[1] <= 1.5, self.sleeps)

    def test_other_connection_errors_are_not_retried(self) -> None:
        self._fail_then_succeed(WRONG_PASSWORD)

        with self.assertRaises(psycopg.OperationalError):
            connection.connect(_settings(), sleep=self.sleeps.append)

        self.assertEqual(1, self.attempts)
        self.assertEqual([], self.sleeps)

    def test_gives_up_after_all_retries_and_raises(self) -> None:
        retries = len(connection.CONNECT_RETRY_DELAYS)
        self._fail_then_succeed(*([ROLE_LIMIT] * (retries + 1)))

        with self.assertRaises(psycopg.OperationalError):
            connection.connect(_settings(), sleep=self.sleeps.append)

        self.assertEqual(retries + 1, self.attempts)
        self.assertEqual(retries, len(self.sleeps))
        # 최대로 쉬어도 Lambda 데드라인 여유(60초)보다 짧다.
        self.assertLess(sum(d * 1.5 for d in connection.CONNECT_RETRY_DELAYS), 60)


if __name__ == "__main__":
    unittest.main()
