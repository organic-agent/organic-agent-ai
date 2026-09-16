from __future__ import annotations

import sys
import unittest
from pathlib import Path


EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))

from embedder.controller.admin_event import InvalidAdminPhotoEvent, parse_admin_photo_event
from embedder.domain.admin import AdminPhotoEvent


class AdminPhotoEventTest(unittest.TestCase):
    def payload(self, **overrides) -> dict:
        value = {
            "jobId": 11,
            "attemptCount": 2,
            "jobType": "EMBEDDING",
            "photoId": 31,
            "galleryId": 41,
            "storageKey": "galleries/41/revisions/photo.jpg",
            "revisionId": 51,
        }
        value.update(overrides)
        return value

    def test_parses_exact_photo_contract(self) -> None:
        event = parse_admin_photo_event(self.payload())

        self.assertEqual(11, event.job_id)
        self.assertEqual(2, event.attempt_count)
        self.assertEqual("EMBEDDING", event.job_type)
        self.assertEqual("galleries/41/revisions/photo.jpg", event.storage_key)

    def test_only_two_external_job_types_are_allowed(self) -> None:
        with self.assertRaisesRegex(InvalidAdminPhotoEvent, "UNSUPPORTED_JOB_TYPE"):
            parse_admin_photo_event(self.payload(jobType="MOCK_RECALCULATION"))

    def test_positive_ids_and_attempt_are_required(self) -> None:
        for field in ("jobId", "attemptCount", "photoId", "galleryId", "revisionId"):
            with self.subTest(field=field):
                with self.assertRaises(InvalidAdminPhotoEvent):
                    parse_admin_photo_event(self.payload(**{field: 0}))

        with self.assertRaises(InvalidAdminPhotoEvent):
            parse_admin_photo_event(self.payload(attemptCount=1.5))


if __name__ == "__main__":
    unittest.main()
