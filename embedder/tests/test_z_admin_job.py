from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace


EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))

import embedder as embedder_package
from embedder.admin_event import AdminPhotoEvent


class _ClaimLost(RuntimeError):
    pass


class _Connection:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0

    def __enter__(self):
        global _active_connections
        _active_connections += 1
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        global _active_connections
        _active_connections -= 1
        if exc_type is not None:
            self.rollbacks += 1
        return None

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1


class _Storage:
    fail_read = False
    writes: list[tuple] = []
    active_connection_counts: list[int] = []

    def __init__(self, bucket: str) -> None:
        self.bucket = bucket

    def read(self, key: str) -> bytes:
        self.active_connection_counts.append(_active_connections)
        if self.fail_read:
            raise RuntimeError("s3 read failed")
        return b"image"

    def write(self, key: str, data: bytes, content_type: str) -> None:
        self.writes.append((key, data, content_type))


class _Image:
    size = (6000, 4000)


class _Vector(list):
    pass


_connections: list[_Connection] = []
_active_connections = 0
_completed: list[str] = []
_failed: list[str] = []
_model_loads = 0

db_module = types.ModuleType("embedder.db")
db_module.AdminJobClaimLost = _ClaimLost
def _connect(settings):
    connection = _Connection()
    _connections.append(connection)
    return connection


db_module.connect = _connect
db_module.verify_admin_photo_event = lambda connection, event: True
db_module.complete_admin_derivative = lambda connection, event, preview, meta: _completed.append("DERIVATIVE")
db_module.complete_admin_embedding = lambda connection, event, vector, model_id, set_status=None: _completed.append("EMBEDDING")
db_module.complete_admin_quality = lambda connection, event, result: _completed.append("QUALITY_ANALYSIS")
db_module.fail_admin_photo_job = lambda connection, event, code: (_failed.append(code) or 1)

images_module = types.ModuleType("embedder.images")
images_module.open_original = lambda data: _Image()
images_module.prepare = lambda image, long_edge: _Image()
images_module.preview_key_for = lambda key: f"previews/{key}.jpg"
images_module.to_jpeg = lambda image, quality: b"jpeg"

metadata_module = types.ModuleType("embedder.metadata")
metadata_module.extract = lambda original, byte_size: object()

model_module = types.ModuleType("embedder.model")


def _load_model(settings):
    global _model_loads
    _model_loads += 1
    return SimpleNamespace(encode=lambda images: [_Vector([0.1, 0.2, 0.3])])


model_module.load_from = _load_model
quality_module = types.ModuleType("embedder.quality")
quality_module.analyze = lambda image, size: SimpleNamespace(score=88.0, signals={"algorithmVersion": "test"})
storage_module = types.ModuleType("embedder.storage")
storage_module.PhotoStorage = _Storage

_replacements = {
    "db": db_module,
    "images": images_module,
    "metadata": metadata_module,
    "model": model_module,
    "quality": quality_module,
    "storage": storage_module,
}
_saved_modules = {name: sys.modules.get(f"embedder.{name}") for name in _replacements}
_saved_attributes = {name: getattr(embedder_package, name, None) for name in _replacements}
for name, module in _replacements.items():
    sys.modules[f"embedder.{name}"] = module
    setattr(embedder_package, name, module)

from embedder import admin_job

for name, previous in _saved_modules.items():
    if previous is None:
        sys.modules.pop(f"embedder.{name}", None)
    else:
        sys.modules[f"embedder.{name}"] = previous
for name, previous in _saved_attributes.items():
    if previous is None:
        delattr(embedder_package, name)
    else:
        setattr(embedder_package, name, previous)


class AdminPhotoJobTest(unittest.TestCase):
    def setUp(self) -> None:
        global _active_connections, _model_loads
        _connections.clear()
        _active_connections = 0
        _model_loads = 0
        _completed.clear()
        _failed.clear()
        _Storage.fail_read = False
        _Storage.writes.clear()
        _Storage.active_connection_counts.clear()

    def event(self, job_type: str) -> AdminPhotoEvent:
        return AdminPhotoEvent(11, 2, job_type, 31, 41, "galleries/41/photo.jpg", 51)

    def settings(self):
        return SimpleNamespace(s3_bucket="bucket", resize_long_edge=1024, preview_quality=82, model_id="test-model")

    def test_derivative_does_not_load_embedding_model(self) -> None:
        result = admin_job.run(self.event("DERIVATIVE"), self.settings())

        self.assertEqual("SUCCEEDED", result["status"])
        self.assertEqual(["DERIVATIVE"], _completed)
        self.assertEqual(0, _model_loads)
        self.assertEqual(2, len(_connections))
        self.assertEqual(0, _connections[0].commits)
        self.assertEqual(1, _connections[1].commits)
        self.assertEqual([0], _Storage.active_connection_counts)

    def test_embedding_loads_model_only_on_embedding_branch(self) -> None:
        result = admin_job.run(self.event("EMBEDDING"), self.settings())

        self.assertEqual("SUCCEEDED", result["status"])
        self.assertEqual(["EMBEDDING"], _completed)
        self.assertEqual(1, _model_loads)

    def test_explicit_processing_error_is_persisted_as_failed(self) -> None:
        _Storage.fail_read = True

        result = admin_job.run(self.event("QUALITY_ANALYSIS"), self.settings())

        self.assertEqual("FAILED", result["status"])
        self.assertEqual(["RUNTIMEERROR"], _failed)
        self.assertEqual(2, len(_connections))
        self.assertEqual(1, _connections[1].commits)

    def test_result_cas_loss_marks_still_current_job_failed(self) -> None:
        original = admin_job.db.complete_admin_quality
        admin_job.db.complete_admin_quality = lambda connection, event, result: (_ for _ in ()).throw(
            _ClaimLost("PHOTO_REVISION_MISMATCH")
        )
        try:
            result = admin_job.run(self.event("QUALITY_ANALYSIS"), self.settings())
        finally:
            admin_job.db.complete_admin_quality = original

        self.assertEqual("FAILED", result["status"])
        self.assertEqual(["PHOTO_REVISION_MISMATCH"], _failed)
        self.assertEqual(3, len(_connections))
        self.assertEqual(1, _connections[1].rollbacks)
        self.assertEqual(1, _connections[2].commits)

    def test_late_previous_attempt_is_ignored_when_terminal_cas_is_lost(self) -> None:
        original_complete = admin_job.db.complete_admin_quality
        original_fail = admin_job.db.fail_admin_photo_job
        admin_job.db.complete_admin_quality = lambda connection, event, result: (_ for _ in ()).throw(
            _ClaimLost("JOB_ATTEMPT_MISMATCH")
        )
        admin_job.db.fail_admin_photo_job = lambda connection, event, code: 0
        try:
            result = admin_job.run(self.event("QUALITY_ANALYSIS"), self.settings())
        finally:
            admin_job.db.complete_admin_quality = original_complete
            admin_job.db.fail_admin_photo_job = original_fail

        self.assertEqual("IGNORED", result["status"])
        self.assertEqual("JOB_ATTEMPT_MISMATCH", result["failureCode"])
        self.assertEqual(1, _connections[1].rollbacks)
        self.assertEqual(1, _connections[2].commits)

    def test_s3_and_model_work_run_without_an_open_database_connection(self) -> None:
        active_during_encode: list[int] = []
        original = admin_job.model.load_from
        admin_job.model.load_from = lambda settings: SimpleNamespace(
            encode=lambda images: (active_during_encode.append(_active_connections) or [_Vector([0.1])])
        )
        try:
            result = admin_job.run(self.event("EMBEDDING"), self.settings())
        finally:
            admin_job.model.load_from = original

        self.assertEqual("SUCCEEDED", result["status"])
        self.assertEqual([0], _Storage.active_connection_counts)
        self.assertEqual([0], active_during_encode)
        self.assertEqual(2, len(_connections))


if __name__ == "__main__":
    unittest.main()
