"""사진마다 미리보기를 만들어 올리고, 그 파일로 임베딩을 계산해 촬영 정보와 함께 적재한다.

순서가 계약이다:
    ① 원본을 열어 EXIF 를 읽고 1024px 로 다듬는다
    ② JPEG 로 인코딩해 S3 에 PUT
    ③ PUT 한 그 JPEG 바이트를 다시 열어 DINOv3 에 넣는다
    ④ 벡터 · preview_key · EXIF 를 한 트랜잭션으로 적재
벡터가 실제 올라간 파일에서 나오므로 "미리보기 없는 벡터"는 생길 수 없고, PUT 실패는 그 사진의 실패가 된다.

운영은 wes 스위퍼가 배정한 `photo_ids` 만 처리한다. 갤러리 전체 경로는 로컬 CLI 전용이다.
배치(8장)마다 commit 하고, Lambda 에서는 `remaining_seconds` 를 보고 하드 킬 전에 배치 경계에서 멈춘다.
원본 GET 은 스레드 풀이 두 배치 앞서 미리 받아 둔다.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, NamedTuple

from PIL import Image

from embedder.config.settings import Settings
from embedder.domain.photo import EmbeddingResult, PhotoMetadata, PhotoRef
from embedder.domain.run import RunResult
from embedder.infrastructure import model
from embedder.repository import connection, photos
from embedder.repository.storage import PhotoStorage
from embedder.service import images, metadata

log = logging.getLogger(__name__)

#: 몇 배치 앞까지 원본 GET 을 미리 걸어 두는가. 2 면 메모리에 원본이 최대 세 배치(~300MB) 올라간다.
PREFETCH_BATCHES = 2


class _Prepared(NamedTuple):
    """벡터가 나오기 전까지 사진 한 장에 대해 알아낸 것. encode 뒤 벡터와 합쳐 EmbeddingResult 가 된다."""

    ref: PhotoRef
    preview_key: str
    metadata: PhotoMetadata | None


def run(
    gallery_id: int,
    settings: Settings | None = None,
    remaining_seconds: Callable[[], float] | None = None,
    photo_ids: list[int] | None = None,
) -> dict:
    """`photo_ids` 가 있으면 그 목록만(운영), 없으면 갤러리에서 벡터 없는 사진 전체(로컬 CLI).

    `remaining_seconds` 는 남은 실행 시간을 알려 주는 함수. Lambda 가 넘기고 로컬은 None(멈추지 않는다).
    """
    started = time.monotonic()
    settings = settings or Settings.from_env()
    result = RunResult(gallery_id=gallery_id)
    if photo_ids is not None:
        result.photo_ids = len(photo_ids)
    tag = f"갤러리 {gallery_id}" + (f" 사진 {len(photo_ids)}장" if photo_ids is not None else "")

    storage = PhotoStorage(settings.s3_bucket, max_concurrency=settings.download_workers)

    with connection.connect(settings) as conn:
        if photo_ids is not None:
            targets = photos.fetch_by_ids(conn, photo_ids)
        else:
            targets = photos.fetch_targets(conn, gallery_id)

        result.targets = len(targets)
        log.info("%s: 대상 %s장", tag, len(targets))

        embedder = model.load_from(settings)

        # 데드라인 판단은 평균이 아니라 최댓값으로 한다. 배치 시간은 원본 크기에 따라 크게 흔들린다.
        longest_batch = 0.0

        batches = _chunked(targets, settings.batch_size)
        # GET 만 풀에서 미리 받는다. 처리·PUT·commit 은 이 스레드가 한 장씩 한다.
        pool = ThreadPoolExecutor(
            max_workers=settings.download_workers, thread_name_prefix="s3-get",
        )
        prefetched: dict[int, list[Future]] = {}
        try:
            for index, batch in enumerate(batches):
                if remaining_seconds is not None:
                    budget = longest_batch + settings.stop_margin_seconds
                    left = remaining_seconds()
                    if left < budget:
                        result.stopped = True
                        log.warning(
                            "%s: 남은 시간 %.0fs < 배치 예산 %.0fs -- 배치 경계에서 멈춤 (남은 사진 %s장)",
                            tag, left, budget, result.remaining,
                        )
                        break

                for ahead in range(index, min(index + PREFETCH_BATCHES + 1, len(batches))):
                    if ahead not in prefetched:
                        prefetched[ahead] = [
                            pool.submit(storage.read, ref.storage_key) for ref in batches[ahead]
                        ]
                downloads = prefetched.pop(index)

                batch_started = time.monotonic()
                # 벡터는 배치 단위로 나오므로 (ref · 미리보기 키 · EXIF) 와 모델 입력을 같은 순서로 모아 둔다.
                pending: list[_Prepared] = []
                loaded_images: list[Image.Image] = []

                for ref, download in zip(batch, downloads):
                    result.attempted += 1
                    try:
                        data = download.result()  # GET 실패는 여기서 터진다
                        original = images.open_original(data)
                        prepared = images.prepare(data, settings.resize_long_edge)
                        # ① EXIF 는 회전·축소 전 원본에서. 실패해도 사진을 버리지 않는다.
                        photo_metadata = _read_metadata(ref, original, len(data), result)

                        # ② 미리보기를 먼저 올린다. 실패하면 encode 에 들어가지 않아 벡터만 남는 사진이 없다.
                        key = images.preview_key_for(ref.storage_key)
                        jpeg = images.to_jpeg(prepared, settings.preview_quality)
                        storage.write(key, jpeg, "image/jpeg")

                        # ③ 모델 입력은 S3 에 올린 바로 그 바이트다.
                        model_input = images.open_preview(jpeg)

                        # 두 리스트는 함께 늘린다. 위에서 실패하면 어느 쪽에도 안 들어가 벡터와 짝이 어긋나지 않는다.
                        pending.append(_Prepared(ref, key, photo_metadata))
                        loaded_images.append(model_input)
                    except Exception:
                        # 한 장이 잡 전체를 죽이지 않는다. 벡터가 없는 채로 남아 다음 호출이 다시 집는다.
                        log.exception("사진을 처리하지 못했습니다: %s", ref.storage_key)
                        result.failed.append(ref.storage_key)

                if loaded_images:
                    vectors = embedder.encode(loaded_images)

                    stored = photos.store_embeddings(
                        conn,
                        [
                            EmbeddingResult(item.ref, vector, item.preview_key, item.metadata)
                            for item, vector in zip(pending, vectors)
                        ],
                        model_id=settings.model_id,
                    )

                    # ④ 배치 단위 커밋. 중간에 죽어도 그때까지는 남는다.
                    conn.commit()
                    result.processed += stored

                batch_seconds = time.monotonic() - batch_started
                longest_batch = max(longest_batch, batch_seconds)
                if loaded_images:
                    log.info(
                        "진행 %s/%s (장당 %.2fs)",
                        result.processed, result.targets, batch_seconds / len(loaded_images),
                    )
        finally:
            # 데드라인으로 멈췄으면 시작 안 한 GET 은 취소하고, 진행 중인 GET 은 기다리지 않는다.
            pool.shutdown(wait=False, cancel_futures=True)

    result.elapsed_seconds = time.monotonic() - started
    log.info("완료: %s", result.to_dict())
    return result.to_dict()


def _read_metadata(
    ref: PhotoRef,
    original: Image.Image,
    byte_size: int,
    result: RunResult,
) -> PhotoMetadata | None:
    """원본에서 촬영 정보를 읽는다. 실패하면 None — EXIF 파싱 문제가 임베딩 실패로 둔갑하면 안 된다."""
    try:
        return metadata.extract(original, byte_size)
    except Exception:
        log.exception("촬영 정보를 읽지 못했습니다: %s", ref.storage_key)
        result.metadata_failed.append(ref.storage_key)
        return None


def _chunked(items: list, size: int) -> list[list]:
    return [items[start:start + size] for start in range(0, len(items), size)]
