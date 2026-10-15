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

DB 연결은 대상 조회와 배치 적재 때만 짧게 연다. 모델 로드·다운로드·디코드·PUT·추론 동안에는 연결을 쥐지 않는다 —
동시 실행 수가 곧 DB 연결 수가 되지 않게 하려는 것이다(wes #274, 동시 실행 32 → 48).

결과 큐 스위치(wes #286):
    - 페이로드에 `storage_keys` 가 있으면 대상을 DB 에서 읽지 않고 페이로드로 만든다.
    - `settings.result_queue_url` 이 있으면 ④ 를 DB 대신 SQS 결과 큐로 보낸다(배치 하나 = 메시지 하나).
      writer Lambda 가 같은 CAS 규칙으로 적재한다. 전송 실패는 그 배치 사진의 실패다(벡터 없음 → wes 재배정).
    - 둘 다면 DB 접속을 아예 열지 않는다(배치 임베더는 VPC 밖, DB 환경변수 없음).
    - 둘 다 없으면(로컬 CLI · 큐 없는 배포) 지금 그대로 DB 를 읽고 쓴다.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Callable, Iterator, NamedTuple

from PIL import Image

from embedder.config.settings import Settings
from embedder.domain.photo import EmbeddingResult, PhotoMetadata, PhotoRef
from embedder.domain.run import RunResult
from embedder.infrastructure import model
from embedder.repository import connection, photos
from embedder.repository.results import ResultQueue
from embedder.repository.storage import PhotoStorage
from embedder.service import images, metadata

log = logging.getLogger(__name__)

#: 몇 배치 앞까지 원본 GET 을 미리 걸어 두는가. 2 면 메모리에 원본이 최대 세 배치(~300MB) 올라간다.
PREFETCH_BATCHES = 2


#: 단계 시간 로그의 순서. 받기 대기 · 원본 디코드(EXIF 포함) · JPEG 인코딩 · PUT · 미리보기 다시 열기 · 추론 · 적재 · 모델 로드.
STAGES = ("get", "decode", "jpeg", "put", "reopen", "infer", "store", "model")


class _StageClock:
    """단계별 누적 시간(wes #274 T0). 장당 0.5초가 어디에 쓰이는지 보고 처리량 처방을 고르려는 계측이다."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = defaultdict(float)

    @contextmanager
    def measure(self, stage: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[stage] += time.perf_counter() - started

    def summary(self) -> str:
        return " ".join(f"{stage}={self.seconds[stage]:.2f}" for stage in STAGES)


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
    storage_keys: list[str] | None = None,
) -> dict:
    """`photo_ids` 가 있으면 그 목록만(운영), 없으면 갤러리에서 벡터 없는 사진 전체(로컬 CLI).

    `storage_keys` 는 `photo_ids` 와 같은 순서·길이의 원본 키. 있으면 대상을 DB 에서 읽지 않는다.
    `remaining_seconds` 는 남은 실행 시간을 알려 주는 함수. Lambda 가 넘기고 로컬은 None(멈추지 않는다).
    """
    if storage_keys is not None and (photo_ids is None or len(storage_keys) != len(photo_ids)):
        raise ValueError("storage_keys 는 photo_ids 와 같은 길이여야 합니다")

    started = time.monotonic()
    settings = settings or Settings.from_env()
    result = RunResult(gallery_id=gallery_id)
    if photo_ids is not None:
        result.photo_ids = len(photo_ids)
    tag = f"갤러리 {gallery_id}" + (f" 사진 {len(photo_ids)}장" if photo_ids is not None else "")

    storage = PhotoStorage(settings.s3_bucket, max_concurrency=settings.download_workers)
    clock = _StageClock()
    queue = ResultQueue(settings.result_queue_url) if settings.result_queue_url else None

    # 대상이 페이로드에 있으면 조회 연결을 열지 않는다. 결과까지 큐로 가면 이 호출은 DB 를 한 번도 열지 않는다.
    if storage_keys is not None:
        targets = [PhotoRef(photo_id=pid, storage_key=key) for pid, key in zip(photo_ids, storage_keys)]
    else:
        with connection.connect(settings) as conn:
            if photo_ids is not None:
                targets = photos.fetch_by_ids(conn, photo_ids)
            else:
                targets = photos.fetch_targets(conn, gallery_id)

    result.targets = len(targets)
    log.info("%s: 대상 %s장", tag, len(targets))

    with clock.measure("model"):
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
                    with clock.measure("get"):
                        data = download.result()  # GET 실패는 여기서 터진다
                    with clock.measure("decode"):
                        original = images.open_original(data)
                        prepared = images.prepare(data, settings.resize_long_edge)
                        # ① EXIF 는 회전·축소 전 원본에서. 실패해도 사진을 버리지 않는다.
                        photo_metadata = _read_metadata(ref, original, len(data), result)

                    # ② 미리보기를 먼저 올린다. 실패하면 encode 에 들어가지 않아 벡터만 남는 사진이 없다.
                    key = images.preview_key_for(ref.storage_key)
                    with clock.measure("jpeg"):
                        jpeg = images.to_jpeg(prepared, settings.preview_quality)
                    with clock.measure("put"):
                        storage.write(key, jpeg, "image/jpeg")

                    # ③ 모델 입력은 S3 에 올린 바로 그 바이트다.
                    with clock.measure("reopen"):
                        model_input = images.open_preview(jpeg)

                    # 두 리스트는 함께 늘린다. 위에서 실패하면 어느 쪽에도 안 들어가 벡터와 짝이 어긋나지 않는다.
                    pending.append(_Prepared(ref, key, photo_metadata))
                    loaded_images.append(model_input)
                except Exception:
                    # 한 장이 잡 전체를 죽이지 않는다. 벡터가 없는 채로 남아 다음 호출이 다시 집는다.
                    log.exception("사진을 처리하지 못했습니다: %s", ref.storage_key)
                    result.failed.append(ref.storage_key)

            if loaded_images:
                with clock.measure("infer"):
                    vectors = embedder.encode(loaded_images)
                embedded = [
                    EmbeddingResult(item.ref, vector, item.preview_key, item.metadata)
                    for item, vector in zip(pending, vectors)
                ]

                if queue is not None:
                    with clock.measure("store"):
                        result.processed += _send(queue, gallery_id, embedded, settings, result)
                else:
                    # ④ 계산이 끝난 뒤에만 연결을 열어 적재하고 배치 단위로 커밋한다. 중간에 죽어도 그때까지는 남는다.
                    with clock.measure("store"), connection.connect(settings) as conn:
                        stored = photos.store_embeddings(conn, embedded, embedding_model=settings.embedding_model)
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
    log.info("단계 시간(초, %s장): %s", result.processed, clock.summary())
    log.info("완료: %s", result.to_dict())
    return result.to_dict()


def _send(
    queue: ResultQueue,
    gallery_id: int,
    embedded: list[EmbeddingResult],
    settings: Settings,
    result: RunResult,
) -> int:
    """④ 를 결과 큐로. 보낸 장수를 돌려준다. 전송 실패는 그 배치 사진 전부의 실패다 — 다른 사진별 실패처럼
    기록하고 다음 배치로 간다. 벡터가 적재되지 않으므로 wes 가 다시 배정한다(미리보기는 이미 올라가 있어도 무해)."""
    try:
        return queue.send(gallery_id, embedded, settings.embedding_model)
    except Exception:
        log.exception("결과 큐 전송 실패 — 배치 %s장을 실패로 셉니다", len(embedded))
        result.failed.extend(item.ref.storage_key for item in embedded)
        return 0


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
