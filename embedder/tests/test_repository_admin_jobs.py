from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path

EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import db_fakes

db_fakes.install_stubs()
_Connection = db_fakes._Connection
_ManyConnection = db_fakes._ManyConnection

from embedder.domain.admin import AdminPhotoEvent
from embedder.repository import admin_jobs


class AdminPhotoJobDatabaseContractTest(unittest.TestCase):
    def event(self) -> AdminPhotoEvent:
        return AdminPhotoEvent(11, 2, "QUALITY_ANALYSIS", 31, 41, "galleries/41/photo.jpg", 51)

    def test_verification_cas_includes_job_attempt_revision_and_storage_key(self) -> None:
        connection = _Connection(rows=[(1,)])

        self.assertTrue(admin_jobs.verify_admin_photo_event(connection, self.event()))

        sql, params = connection.executed[0]
        self.assertIn("j.attempt_count = %s", sql)
        self.assertIn("j.revision_id = %s", sql)
        self.assertIn("p.storage_key = %s", sql)
        self.assertIn("r.storage_key = %s", sql)
        self.assertEqual(11, params[0])
        self.assertEqual(2, params[1])

    def test_photo_result_and_job_success_are_both_required_before_commit(self) -> None:
        """사진 결과 CAS 는 맞았는데 잡 terminal CAS 가 빗나가면 둘 다 버린다(rowcount 1 → 0)."""
        connection = _Connection(rows=[], rowcounts=[1, 0])

        with self.assertRaises(admin_jobs.AdminJobClaimLost):
            admin_jobs.complete_admin_embedding(connection, self.event(), "VECTOR", "m")

        self.assertEqual(2, len(connection.executed))
        self.assertIn("INSERT INTO photo_analysis", connection.executed[0][0])
        self.assertIn("status = 'SUCCEEDED'", connection.executed[1][0])

    def test_explicit_failure_only_updates_matching_exact_attempt(self) -> None:
        connection = _Connection(rows=[], rowcounts=[1])

        updated = admin_jobs.fail_admin_photo_job(connection, self.event(), "NO_SUCH_KEY")

        self.assertEqual(1, updated)
        sql, params = connection.executed[0]
        self.assertIn("attempt_count = %s", sql)
        self.assertIn("revision_id = %s", sql)
        self.assertIn("status IN ('DISPATCHING', 'DISPATCHED')", sql)
        self.assertEqual("NO_SUCH_KEY", params[0])


class NeverWritesPhotoStatusTest(unittest.TestCase):
    """wes V15(#100): photos.status 는 PENDING·UPLOADED 뿐이고 embedder 역할에 그 컬럼 UPDATE 권한이 없다.
    어드민 사진 결과 CAS 도 status 를 쓰지 않는다."""

    def test_admin_embedding_cas_never_touches_status(self) -> None:
        event = types.SimpleNamespace(job_id=1, attempt_count=1, job_type="EMBEDDING", photo_id=1, revision_id=1,
                                      gallery_id=7, storage_key="galleries/7/a.jpg")
        connection = _Connection(rows=[], rowcounts=[1, 1])
        admin_jobs.complete_admin_embedding(connection, event, "VECTOR", "m")
        photo_sql, _ = connection.executed[0]
        self.assertIn("UPDATE photos p", photo_sql)
        self.assertNotIn("status", photo_sql)
        self.assertIn("SET version = version + 1", photo_sql)


if __name__ == "__main__":
    unittest.main()
