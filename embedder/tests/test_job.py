"""갤러리 잡의 순서 계약 -- 미리보기 PUT → 그 파일로 임베딩 → 한 트랜잭션 -- 과 데드라인 재개(#24).

S3·모델·DB는 전부 가짜다. job.py가 부르는 모듈 속성을 setUp에서 바꿔 끼우고 tearDown에서 되돌린다.
"""

from __future__ import annotations

import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))

from embedder import job


class _Ref:
    def __init__(self, photo_id: int, storage_key: str) -> None:
        self.photo_id = photo_id
        self.storage_key = storage_key


class _Connection:
    def __init__(self) -> None:
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None

    def commit(self) -> None:
        self.commits += 1


class _Db:
    def __init__(self, targets: list[_Ref], locked: bool = True) -> None:
        self.targets = targets
        self.locked = locked
        self.connection = _Connection()
        self.stored: list[list[tuple]] = []
        self.lock_keys: list[tuple[int, int]] = []
        self.fetch_calls: list[dict] = []

    def connect(self, settings):
        return self.connection

    def try_lock_gallery(self, connection, gallery_id: int, shard_index: int = 0) -> bool:
        self.lock_keys.append((gallery_id, shard_index))
        return self.locked

    def fetch_targets(self, connection, gallery_id: int, force: bool, since=None):
        self.fetch_calls.append({"gallery_id": gallery_id, "force": force, "since": since})
        return list(self.targets)

    def store_embeddings(self, connection, results, model_id: str, set_status="EMBEDDED") -> int:
        rows = list(results)
        self.stored.append(rows)
        self.set_status = set_status
        return len(rows)

    def fetch_by_ids(self, connection, photo_ids):
        self.fetch_calls.append({"photo_ids": list(photo_ids)})
        by_id = {t.photo_id: t for t in self.targets}
        return [by_id[pid] for pid in photo_ids if pid in by_id]


class _Storage:
    """키에 'bad'가 들어간 사진은 PUT이, 'missing'이 들어간 사진은 GET이 실패한다."""

    instances: list["_Storage"] = []

    def __init__(self, bucket: str, max_concurrency: int = 1) -> None:
        self.max_concurrency = max_concurrency
        self.reads: list[str] = []
        self.read_threads: set[str] = set()
        self.writes: list[tuple[str, bytes, str]] = []
        self._lock = threading.Lock()
        _Storage.instances.append(self)

    def read(self, key: str) -> bytes:
        with self._lock:
            self.reads.append(key)
            self.read_threads.add(threading.current_thread().name)
        if "missing" in key:
            raise RuntimeError("NoSuchKey")
        return f"original:{key}".encode()

    def write(self, key: str, data: bytes, content_type: str) -> None:
        if "bad" in key:
            raise RuntimeError("AccessDenied")
        self.writes.append((key, data, content_type))


class _Model:
    def __init__(self) -> None:
        self.encoded: list[list] = []
        self.loads = 0

    def load_from(self, settings):
        self.loads += 1
        return self

    def encode(self, images_):
        self.encoded.append(list(images_))
        return [f"vec:{img}" for img in images_]


class _Images:
    """이미지 처리는 문자열 태그로 흉내 낸다. open_preview의 입력이 write된 바이트인지 볼 수 있게."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def open_original(self, data: bytes):
        self.calls.append("open_original")
        return f"orig({data.decode()})"

    def prepare(self, data: bytes, long_edge: int):
        self.calls.append("prepare")
        return f"prep({data.decode()})"

    def preview_key_for(self, storage_key: str) -> str:
        return f"previews/{storage_key.rsplit('.', 1)[0]}.jpg"

    def to_jpeg(self, image, quality: int) -> bytes:
        self.calls.append("to_jpeg")
        return f"jpeg({image})".encode()

    def open_preview(self, data: bytes):
        self.calls.append("open_preview")
        return f"preview({data.decode()})"


class _Metadata:
    def __init__(self, fail_for: str | None = None) -> None:
        self.fail_for = fail_for

    def extract(self, original, byte_size: int):
        if self.fail_for and self.fail_for in original:
            raise ValueError("bad exif")
        return f"meta({original})"


def _settings(batch_size: int = 2, stop_margin: int = 60, download_workers: int = 2,
              shard_photos: int = 0, max_shards: int = 8):
    return SimpleNamespace(
        s3_bucket="bucket", batch_size=batch_size, resize_long_edge=1024, preview_quality=82,
        model_id="test-model", stop_margin_seconds=stop_margin, download_workers=download_workers,
        shard_photos=shard_photos, max_shards=max_shards,
    )


class GalleryJobTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {name: getattr(job, name) for name in ("db", "images", "metadata", "model", "PhotoStorage")}
        _Storage.instances.clear()
        self.images = _Images()
        self.model = _Model()
        job.images = self.images
        job.model = self.model
        job.metadata = _Metadata()
        job.PhotoStorage = _Storage

    def tearDown(self) -> None:
        for name, value in self._saved.items():
            setattr(job, name, value)

    def _install_db(self, targets: list[_Ref], locked: bool = True) -> _Db:
        fake = _Db(targets, locked=locked)
        job.db = fake
        return fake

    def test_preview_is_uploaded_before_encode_and_encode_reads_uploaded_bytes(self) -> None:
        fake_db = self._install_db([_Ref(1, "galleries/7/a.heic"), _Ref(2, "galleries/7/b.jpg")])

        result = job.run(7, settings=_settings(batch_size=2))

        storage = _Storage.instances[0]
        # PUT이 encode보다 먼저 -- 호출 순서에 open_preview가 to_jpeg 뒤에 온다.
        self.assertEqual(
            ["open_original", "prepare", "to_jpeg", "open_preview"] * 2, self.images.calls,
        )
        # encode 입력은 S3에 올린 바로 그 JPEG 바이트를 다시 연 것이다.
        written = [data.decode() for _, data, _ in storage.writes]
        self.assertEqual([[f"preview({written[0]})", f"preview({written[1]})"]], self.model.encoded)
        self.assertEqual(
            [("previews/galleries/7/a.jpg", "image/jpeg"), ("previews/galleries/7/b.jpg", "image/jpeg")],
            [(key, ct) for key, _, ct in storage.writes],
        )
        # 적재 행에는 preview_key가 항상 있다.
        rows = fake_db.stored[0]
        self.assertEqual(["previews/galleries/7/a.jpg", "previews/galleries/7/b.jpg"], [r[2] for r in rows])
        self.assertEqual(2, result["processed"])
        self.assertEqual(0, result["remaining"])
        self.assertFalse(result["stopped"])
        self.assertNotIn("previewsFailed", result)
        self.assertEqual(1, fake_db.connection.commits)

    def test_put_failure_keeps_photo_out_of_encode_and_store(self) -> None:
        fake_db = self._install_db([_Ref(1, "galleries/7/good.jpg"), _Ref(2, "galleries/7/bad.jpg")])

        result = job.run(7, settings=_settings(batch_size=2))

        self.assertEqual(["galleries/7/bad.jpg"], result["failed"])
        self.assertEqual(1, result["processed"])
        # 실패한 사진은 encode에도, store에도 들어가지 않는다 -- 벡터만 남는 사진이 없다.
        self.assertEqual(1, len(self.model.encoded[0]))
        self.assertEqual([1], [row[0].photo_id for row in fake_db.stored[0]])
        # 데드라인과 무관하게 전부 시도했으므로 remaining은 0이다 (실패는 다음 호출이 다시 집는다).
        self.assertEqual(0, result["remaining"])

    def test_get_failure_marks_only_that_photo_failed(self) -> None:
        fake_db = self._install_db([_Ref(1, "galleries/7/missing.jpg"), _Ref(2, "galleries/7/ok.jpg")])

        result = job.run(7, settings=_settings(batch_size=2))

        self.assertEqual(["galleries/7/missing.jpg"], result["failed"])
        self.assertEqual(1, result["processed"])
        self.assertEqual([2], [row[0].photo_id for row in fake_db.stored[0]])
        # GET에 실패한 사진은 open_original 이전에 접혔다 -- 이미지 호출은 성공한 한 장 몫뿐이다.
        self.assertEqual(["open_original", "prepare", "to_jpeg", "open_preview"], self.images.calls)

    def test_originals_are_fetched_concurrently_on_worker_threads(self) -> None:
        """GET이 직렬이면 두 스레드가 만나는 barrier가 시간 초과로 깨져 두 장 다 failed가 된다."""
        targets = [_Ref(i, f"galleries/7/{i}.jpg") for i in range(1, 5)]
        fake_db = self._install_db(targets)
        barrier = threading.Barrier(2, timeout=3)
        original_read = _Storage.read

        def read_in_pairs(storage, key):
            barrier.wait()
            return original_read(storage, key)

        _Storage.read = read_in_pairs
        try:
            result = job.run(7, settings=_settings(batch_size=2, download_workers=2))
        finally:
            _Storage.read = original_read

        self.assertEqual([], result["failed"])
        self.assertEqual(4, result["processed"])
        storage = _Storage.instances[0]
        self.assertEqual(2, storage.max_concurrency)
        self.assertEqual({t.storage_key for t in targets}, set(storage.reads))
        # 메인 스레드가 아니라 풀 스레드가 받았다.
        self.assertTrue(all(name.startswith("s3-get") for name in storage.read_threads), storage.read_threads)
        # 순서·짝은 그대로다: 벡터는 대상 순서대로 적재된다.
        self.assertEqual([1, 2], [row[0].photo_id for row in fake_db.stored[0]])
        self.assertEqual([3, 4], [row[0].photo_id for row in fake_db.stored[1]])

    def test_metadata_failure_does_not_drop_the_photo(self) -> None:
        fake_db = self._install_db([_Ref(1, "galleries/7/noexif.jpg")])
        job.metadata = _Metadata(fail_for="noexif")

        result = job.run(7, settings=_settings())

        self.assertEqual(["galleries/7/noexif.jpg"], result["metadataFailed"])
        self.assertEqual([], result["failed"])
        self.assertEqual(1, result["processed"])
        self.assertIsNone(fake_db.stored[0][0][3])

    def test_stops_at_batch_boundary_when_remaining_time_is_short(self) -> None:
        targets = [_Ref(i, f"galleries/7/{i}.jpg") for i in range(1, 6)]
        fake_db = self._install_db(targets)
        # 배치 1 앞: 넉넉. 배치 2 앞: 여유(60s) 미만 → 멈춤. 세 번째는 불리면 안 된다.
        clock = iter([800.0, 30.0])

        result = job.run(7, settings=_settings(batch_size=2), remaining_seconds=lambda: next(clock))

        self.assertTrue(result["stopped"])
        self.assertEqual(2, result["processed"])
        self.assertEqual(3, result["remaining"])
        self.assertEqual(5, result["targets"])
        self.assertEqual(1, fake_db.connection.commits)
        self.assertEqual(1, len(self.model.encoded))

    def test_no_deadline_processes_everything(self) -> None:
        targets = [_Ref(i, f"galleries/7/{i}.jpg") for i in range(1, 6)]
        fake_db = self._install_db(targets)

        result = job.run(7, settings=_settings(batch_size=2), remaining_seconds=None)

        self.assertFalse(result["stopped"])
        self.assertEqual(5, result["processed"])
        self.assertEqual(3, fake_db.connection.commits)

    def test_skips_when_another_run_holds_the_gallery(self) -> None:
        self._install_db([_Ref(1, "galleries/7/a.jpg")], locked=False)

        result = job.run(7, settings=_settings())

        self.assertEqual(job.ALREADY_RUNNING, result["skipped"])
        self.assertEqual(0, result["targets"])
        self.assertEqual([], _Storage.instances[0].reads)
        self.assertEqual(0, self.model.loads)


class ShardTest(unittest.TestCase):
    """갤러리 샤딩(#56) — 분배·샤드 수·조정자·샤드 실행."""

    def test_shard_select_partitions_in_fixed_order_and_validates(self) -> None:
        refs = list("abcdefg")
        parts = [job.Shard(i, 3).select(refs) for i in range(3)]
        self.assertEqual([["a", "d", "g"], ["b", "e"], ["c", "f"]], parts)
        self.assertEqual(sorted(refs), sorted(sum(parts, [])))      # 빠지거나 겹치지 않는다
        self.assertEqual({"index": 1, "total": 3}, job.Shard(1, 3).to_payload())
        self.assertEqual(job.Shard(1, 3), job.Shard.from_payload({"index": "1", "total": "3"}))
        self.assertIsNone(job.Shard.from_payload(None))
        for index, total in ((3, 3), (-1, 3), (0, 0)):
            with self.subTest(index=index, total=total), self.assertRaises(ValueError):
                job.Shard(index, total)

    def test_plan_shards_by_photo_count_with_cap(self) -> None:
        s = _settings(shard_photos=250, max_shards=8)
        self.assertEqual(1, job.plan_shards(0, s))
        self.assertEqual(1, job.plan_shards(250, s))
        self.assertEqual(2, job.plan_shards(251, s))
        self.assertEqual(4, job.plan_shards(822, s))
        self.assertEqual(8, job.plan_shards(7000, s))                    # 상한
        self.assertEqual(1, job.plan_shards(7000, _settings(shard_photos=0)))   # 샤딩 끔


class ShardedJobTest(GalleryJobTest):
    """GalleryJobTest 의 가짜(S3·모델·DB)를 그대로 쓴다. 상속이라 부모 테스트도 한 번 더 돈다 — 값싸다."""

    def test_coordinator_fans_out_without_loading_model(self) -> None:
        targets = [_Ref(i, f"galleries/7/{i}.jpg") for i in range(1, 6)]
        fake_db = self._install_db(targets)
        fan_outs: list[tuple[int, str | None]] = []

        result = job.run(7, force=True, settings=_settings(shard_photos=2, max_shards=8),
                         fan_out=lambda n, started: (fan_outs.append((n, started)) or True))

        self.assertTrue(result["coordinator"])
        self.assertEqual(3, result["shards"])                 # ceil(5 / 2)
        self.assertTrue(result["fannedOut"])
        self.assertEqual(5, result["targets"])
        self.assertEqual(0, result["processed"])
        self.assertEqual([(3, result["runStartedAt"])], fan_outs)
        self.assertIsNotNone(result["runStartedAt"])           # force 라 시작 시각이 생겼고 샤드에 넘어간다
        self.assertEqual(0, self.model.loads)                  # 조정자는 모델을 올리지 않는다
        self.assertEqual([], _Storage.instances[0].reads)
        self.assertEqual([(7, 0)], fake_db.lock_keys)          # 조정자 잠금 = 샤드 0 키
        # force 의 대상 조회는 since 로 — "이번 실행 이후 벡터"만 건너뛴다.
        self.assertEqual([{"gallery_id": 7, "force": True, "since": result["runStartedAt"]}], fake_db.fetch_calls)

    def test_coordinator_failure_to_fan_out_raises(self) -> None:
        self._install_db([_Ref(i, f"galleries/7/{i}.jpg") for i in range(1, 6)])

        with self.assertRaises(RuntimeError):
            job.run(7, settings=_settings(shard_photos=2), fan_out=lambda n, started: False)

    def test_single_shard_plan_runs_inline(self) -> None:
        fake_db = self._install_db([_Ref(1, "galleries/7/a.jpg"), _Ref(2, "galleries/7/b.jpg")])
        fan_outs: list = []

        result = job.run(7, settings=_settings(shard_photos=250), fan_out=lambda n, s: fan_outs.append(n) or True)

        self.assertEqual([], fan_outs)
        self.assertNotIn("coordinator", result)
        self.assertEqual(2, result["processed"])
        self.assertEqual(1, self.model.loads)
        self.assertEqual([{"gallery_id": 7, "force": False, "since": None}], fake_db.fetch_calls)

    def test_shard_processes_only_its_slice_with_its_own_lock(self) -> None:
        targets = [_Ref(i, f"galleries/7/{i}.jpg") for i in range(1, 6)]
        fake_db = self._install_db(targets)

        result = job.run(7, settings=_settings(batch_size=2, shard_photos=2), shard=job.Shard(1, 2),
                         run_started_at="2026-09-06T00:00:00+00:00")

        self.assertEqual({"index": 1, "total": 2}, result["shard"])
        self.assertEqual(2, result["targets"])                 # 위치 1, 3 → id 2, 4
        self.assertEqual(2, result["processed"])
        self.assertEqual([2, 4], [row[0].photo_id for row in fake_db.stored[0]])
        self.assertEqual([(7, 1)], fake_db.lock_keys)
        self.assertEqual("2026-09-06T00:00:00+00:00", result["runStartedAt"])
        self.assertEqual("2026-09-06T00:00:00+00:00", fake_db.fetch_calls[0]["since"])
        self.assertNotIn("coordinator", result)

    def test_shard_never_fans_out_even_if_large(self) -> None:
        self._install_db([_Ref(i, f"galleries/7/{i}.jpg") for i in range(1, 6)])

        result = job.run(7, settings=_settings(shard_photos=1), shard=job.Shard(0, 1), fan_out=None)

        self.assertEqual(5, result["processed"])
        self.assertNotIn("coordinator", result)


if __name__ == "__main__":
    unittest.main()


class PhotoIdsJobTest(GalleryJobTest):
    """v2 스트리밍(#73): 사진 id 목록 호출은 잠금·대상 조회·fan-out 없이 그 목록만 처리한다."""

    def test_photo_ids_path_skips_lock_and_fetch_targets_and_keeps_request_order(self) -> None:
        fake_db = self._install_db([_Ref(1, "galleries/7/a.jpg"), _Ref(2, "galleries/7/b.jpg"), _Ref(3, "galleries/7/c.jpg")],
                                   locked=False)   # 잠금이 막혀 있어도 이 경로는 잠금을 보지 않는다
        fanned: list[int] = []

        result = job.run(7, settings=_settings(batch_size=2, shard_photos=1, max_shards=8),
                         photo_ids=[3, 1, 99], fan_out=lambda n, started: (fanned.append(n) or True))

        self.assertEqual([], fake_db.lock_keys)
        self.assertEqual([{"photo_ids": [3, 1, 99]}], fake_db.fetch_calls)
        self.assertEqual([], fanned)                       # shard_photos=1 이어도 조정자로 가지 않는다
        self.assertEqual(2, result["targets"])             # 없는 id 99 는 조용히 빠진다
        self.assertEqual(2, result["processed"])
        self.assertEqual(3, result["photoIds"])
        self.assertNotIn("coordinator", result)
        rows = fake_db.stored[0]
        self.assertEqual(["previews/galleries/7/c.jpg", "previews/galleries/7/a.jpg"], [r[2] for r in rows])

    def test_set_status_from_settings_reaches_store(self) -> None:
        fake_db = self._install_db([_Ref(1, "galleries/7/a.jpg")])
        settings = _settings(batch_size=1)
        settings.set_status = None
        job.run(7, settings=settings, photo_ids=[1])
        self.assertIsNone(fake_db.set_status)

        fake_db = self._install_db([_Ref(1, "galleries/7/a.jpg")])
        job.run(7, settings=_settings(batch_size=1), photo_ids=[1])   # 설정에 없으면 옛 계약 EMBEDDED
        self.assertEqual("EMBEDDED", fake_db.set_status)

    def test_empty_photo_ids_does_nothing(self) -> None:
        fake_db = self._install_db([_Ref(1, "galleries/7/a.jpg")])
        result = job.run(7, settings=_settings(), photo_ids=[])
        self.assertEqual(0, result["targets"])
        self.assertEqual([], fake_db.stored)
