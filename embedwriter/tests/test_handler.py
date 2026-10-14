"""SQS 이벤트 → 한 트랜잭션, 실패 시 메시지별 재시도와 batchItemFailures."""

from __future__ import annotations

import unittest

from helpers import FakeConnection, body, photo, record

from embedwriter import connection, handler


class HandlerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._shared = connection.shared
        self.conn = FakeConnection()
        connection.shared = lambda settings: self.conn
        self.settings = handler._SETTINGS.__class__(**{**handler._SETTINGS.__dict__, "embed_dim": 4})

    def tearDown(self) -> None:
        connection.shared = self._shared

    def test_all_records_in_one_transaction(self) -> None:
        event = {"Records": [record("m1", body(photo(1), photo(2))), record("m2", body(photo(3)))]}

        result = handler.handle(event, self.settings)

        self.assertEqual({"batchItemFailures": []}, result)
        self.assertEqual(1, len(self.conn.executed))
        self.assertEqual([1, 2, 3], self.conn.executed[0][1]["photo_id"])
        self.assertEqual(1, self.conn.commits)

    def test_bulk_failure_retries_each_record_and_reports_only_failing_ones(self) -> None:
        # 사진 2 가 든 문장은 실패한다 — 묶음(1,2,3) 실패 → m1(1) 성공 · m2(2) 실패 · m3(3) 성공.
        self.conn.fail_when = lambda params: 2 in params["photo_id"]
        event = {"Records": [record("m1", body(photo(1))), record("m2", body(photo(2))), record("m3", body(photo(3)))]}

        result = handler.handle(event, self.settings)

        self.assertEqual({"batchItemFailures": [{"itemIdentifier": "m2"}]}, result)
        self.assertEqual([[1, 2, 3], [1], [2], [3]], [params["photo_id"] for _, params in self.conn.executed])
        self.assertEqual(2, self.conn.commits)
        self.assertEqual(2, self.conn.rollbacks)

    def test_malformed_body_fails_only_that_message(self) -> None:
        event = {"Records": [record("bad-json", "{nope"), record("bad-v", body(photo(9), v=2)),
                             record("ok", body(photo(1)))]}

        result = handler.handle(event, self.settings)

        self.assertEqual([{"itemIdentifier": "bad-json"}, {"itemIdentifier": "bad-v"}], result["batchItemFailures"])
        self.assertEqual([[1]], [params["photo_id"] for _, params in self.conn.executed])
        self.assertEqual(1, self.conn.commits)

    def test_only_malformed_records_never_touch_db(self) -> None:
        result = handler.handle({"Records": [record("x", "null")]}, self.settings)
        self.assertEqual([{"itemIdentifier": "x"}], result["batchItemFailures"])
        self.assertEqual([], self.conn.executed)

    def test_broken_connection_is_discarded(self) -> None:
        discarded = []
        original = connection.discard
        connection.discard = lambda: discarded.append(True)
        try:
            def fail(params):
                self.conn.broken = True
                return True
            self.conn.fail_when = fail
            result = handler.handle({"Records": [record("m1", body(photo(1)))]}, self.settings)
        finally:
            connection.discard = original
        self.assertEqual([{"itemIdentifier": "m1"}], result["batchItemFailures"])
        self.assertEqual(2, len(discarded))      # 묶음 한 번 + 메시지별 한 번
        self.assertEqual(0, self.conn.rollbacks)

    def test_connect_failure_fails_whole_invocation(self) -> None:
        def refuse(settings):
            raise OSError("connection refused")
        connection.shared = refuse
        # 묶음도 메시지별도 접속에서 실패하면 전부 batchItemFailures — 예외 없이도 전부 다시 온다.
        result = handler.handle({"Records": [record("m1", body(photo(1))), record("m2", body(photo(2)))]},
                                self.settings)
        self.assertEqual(["m1", "m2"], [f["itemIdentifier"] for f in result["batchItemFailures"]])


if __name__ == "__main__":
    unittest.main()
