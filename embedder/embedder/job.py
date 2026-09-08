"""갤러리 하나의 미리보기 파생본을 만들어 올리고, 그 파일로 임베딩을 계산해 적재한다. 촬영 정보도 같이.

순서가 계약이다: **① 원본을 열어 EXIF를 읽고 1024px로 다듬는다 → ② JPEG로 인코딩해 S3에 PUT →
③ PUT이 성공한 그 JPEG 바이트를 다시 열어 DINOv3에 넣는다 → ④ 벡터·preview_key·EXIF·status를 한
트랜잭션으로 적재.** 벡터가 S3에 실제로 올라간 파일에서 나오므로 "미리보기 없는 벡터"는 구조적으로
생길 수 없고, PUT 실패는 그 사진의 실패가 되어 다음 호출에서 fetch_targets가 자연히 다시 집어 온다.
photoselect가 읽는 픽셀과 벡터가 같은 파일이라는 점도 따라온다. (#24)

파생본과 EXIF를 여기서 만드는 이유는 이 잡이 어차피 원본을 받아 HEIC를 디코딩하고 EXIF 회전과
축소를 해야 하기 때문이다. 비싼 부분은 그것이고, JPEG 인코딩·PUT·1024px 재디코드는 그 옆에 얹히는
비용이다. 별도 잡으로 빼면 같은 이미지를 두 번 받아 두 번 디코딩하게 된다. 앱 서버는 이미지
바이트를 만지지 않으므로 애초에 그쪽에는 선택지가 없다.

진입점(`handler.py` / `__main__.py`)이 둘이고 본체는 이 함수 하나다. Lambda로 감싸기 전에
로컬에서 실제 S3·RDS를 상대로 같은 코드를 검증할 수 있어야 해서 이렇게 갈라 두었다.
나중에 Fargate로 옮겨도 바뀌는 것은 진입점뿐이다.

원본 GET은 스레드 풀이 두 배치 앞서 미리 받아 둔다(`download_workers`). 5~13MB 원본을 한 장씩
받고 다듬기를 번갈아 하면 네트워크와 CPU가 서로를 기다린다 -- 2026-09-05 로컬 E2E의 장당 1.7초는
대부분 이 대기였다. 처리·PUT·commit 순서는 그대로 메인 스레드가 한 장씩 밟는다.

갤러리 샤딩(#56): wes 가 부른 실행(shard 없음)은 **조정자**다 — 잠금 → 대상 조회까지만 하고 대상이 `shard_photos`
를 넘으면 모델을 올리지 않은 채 자기 함수를 N번(`shard:{index,total}`) EVENT 하고 끝난다(`fan_out`). 샤드는 같은
순서의 목록에서 위치 % total == index 인 사진만 맡고, 잠금 키도 (갤러리, 샤드)다. score 와 달리 카운터·체인이 없다 —
embedder 는 잡을 모르고 wes 가 `photo_analysis` 를 세어 EMBED 단계를 닫으므로 샤드는 각자 끝나면 그만이다.
force 는 시작 시각(`run_started_at`)을 샤드·재호출에 넘겨 그 이후 벡터만 "있음"으로 본다 — 재호출이 force 를 잃어도
옛 벡터가 남지 않는다.

끊김에 대한 태도: 배치(8장)마다 commit하므로 어디서 죽어도 그때까지는 남는다. Lambda에서는 남은
시간(`remaining_seconds`)을 보고 하드 킬 전에 배치 경계에서 스스로 멈춘다 -- 결과에 `stopped`와
`remaining`이 실리고, 재호출은 handler의 몫이다.
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from PIL import Image

from embedder import db, images, metadata, model
from embedder.config import Settings
from embedder.storage import PhotoStorage

log = logging.getLogger(__name__)

#: 다른 실행이 같은 갤러리를 잡고 있을 때 결과의 skipped 값.
ALREADY_RUNNING = "already running"

#: 지금 처리하는 배치보다 몇 배치 앞까지 원본 GET을 미리 걸어 두는가. 2면 메모리에 원본이 최대
#: 세 배치(24장, 13MB 원본이면 ~300MB) 올라간다. Lambda 3GB에서 모델과 함께 두어도 남는다.
PREFETCH_BATCHES = 2


@dataclass(frozen=True)
class Shard:
    """N 개 중 index 번째. 사진 목록(고정 순서 `ORDER BY p.id`)에서 위치 % total == index 인 것만 맡는다."""

    index: int
    total: int

    def __post_init__(self) -> None:
        if not (self.total >= 1 and 0 <= self.index < self.total):
            raise ValueError(f"잘못된 샤드 {self.index}/{self.total}")

    def select(self, refs: list) -> list:
        return [r for i, r in enumerate(refs) if i % self.total == self.index]

    def to_payload(self) -> dict:
        return {"index": self.index, "total": self.total}

    @classmethod
    def from_payload(cls, value) -> "Shard | None":
        if not value:
            return None
        return cls(index=int(value["index"]), total=int(value["total"]))


def plan_shards(n_photos: int, settings: Settings) -> int:
    """사진 수로 샤드 수. ceil(n / shard_photos) 를 [1, max_shards] 로 자른다. shard_photos 0 이면 샤딩 없음."""
    shard_photos = getattr(settings, "shard_photos", 0)
    if shard_photos <= 0 or n_photos <= 0:
        return 1
    return max(1, min(getattr(settings, "max_shards", 1), math.ceil(n_photos / shard_photos)))


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RunResult:
    gallery_id: int
    targets: int = 0
    #: 이번 실행이 손댄 사진 수 (성공 + 실패). targets - attempted 가 아직 시도하지 않은 수다.
    attempted: int = 0
    processed: int = 0
    #: 원본을 못 읽었거나, 미리보기를 못 올렸거나, 올린 파일을 못 열었거나 -- 어느 쪽이든 벡터가
    #: 없으므로 다음 호출의 fetch_targets가 다시 집어 온다. 미리보기 실패를 따로 세지 않는
    #: 이유가 이것이다: 미리보기가 곧 임베딩 입력이라 둘은 같은 실패다.
    failed: list[str] = field(default_factory=list)
    #: 벡터·미리보기는 나왔지만 촬영 정보만 읽지 못한 사진. 상세 화면에 정보가 덜 나올 뿐
    #: 사진은 멀쩡히 보이고 임베딩도 끝나 있다.
    metadata_failed: list[str] = field(default_factory=list)
    #: 데드라인 때문에 배치 경계에서 멈췄다. 남은 사진은 remaining.
    stopped: bool = False
    #: 다른 실행이 갤러리를 잡고 있어 아무것도 하지 않았다.
    skipped: str | None = None
    elapsed_seconds: float = 0.0
    #: 이 실행이 맡은 샤드. 없으면 갤러리 전체.
    shard: Shard | None = None
    #: v2(#73) 사진 목록 호출이면 요청 장수. 갤러리 호출이면 None.
    photo_ids: int | None = None
    #: 조정자로 끝났다 — 샤드를 띄우기만 하고 사진은 처리하지 않았다. targets 는 갤러리 전체 대상 수.
    coordinator: bool = False
    shards: int = 0
    fanned_out: bool = False
    run_started_at: str | None = None

    @property
    def remaining(self) -> int:
        return self.targets - self.attempted

    def to_dict(self) -> dict:
        out = {
            "galleryId": self.gallery_id,
            "targets": self.targets,
            "processed": self.processed,
            "failed": self.failed,
            "metadataFailed": self.metadata_failed,
            "stopped": self.stopped,
            "remaining": self.remaining,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
        }
        if self.skipped is not None:
            out["skipped"] = self.skipped
        if self.shard is not None:
            out["shard"] = self.shard.to_payload()
        if self.photo_ids is not None:
            out["photoIds"] = self.photo_ids
        if self.coordinator:
            out.update({"coordinator": True, "shards": self.shards, "fannedOut": self.fanned_out})
        if self.run_started_at:
            out["runStartedAt"] = self.run_started_at
        return out


def run(
    gallery_id: int,
    force: bool = False,
    settings: Settings | None = None,
    remaining_seconds: Callable[[], float] | None = None,
    shard: Shard | None = None,
    run_started_at: str | None = None,
    fan_out: Callable[[int, str | None], bool] | None = None,
    photo_ids: list[int] | None = None,
) -> dict:
    """갤러리 하나(또는 그 샤드 하나), 또는 사진 id 목록(v2, #73)을 처리한다.

    `photo_ids` 가 있으면 **그 목록만** 임베딩한다 — wes 스위퍼가 UPLOADED 사진을 50장씩 배정해 부르는 스트리밍 경로.
    갤러리 잠금·대상 조회·조정자·fan-out 을 전부 건너뛴다(배정 자체가 wes 의 `dispatched_at` 로 원자적이라 잠금이 필요 없다).
    force 는 뜻이 없다(목록에 있으면 계산한다). 데드라인 정지는 그대로 — 50장이면 발동하지 않지만 코드 경로는 같다.

    `remaining_seconds`는 실행 환경이 남은 시간을 알려 주는 함수다. Lambda handler가
    `context.get_remaining_time_in_millis`를 감싸 넘기고, 로컬 CLI는 None이다(멈추지 않는다).

    `shard` 가 없고 `fan_out` 이 있으면 조정자다: 대상이 샤드 2개 이상이면 fan_out(N, run_started_at) 을 부르고
    `coordinator: true` 로 끝난다(모델을 올리지 않는다). `run_started_at`(ISO) 은 force 실행의 시작 시각 —
    그 이후 적재된 벡터만 "있음"으로 친다. force 인데 없으면 지금 시각으로 만든다.
    """
    started = time.monotonic()
    settings = settings or Settings.from_env()
    if force and run_started_at is None:
        run_started_at = now_iso()
    result = RunResult(gallery_id=gallery_id, shard=shard, run_started_at=run_started_at)
    if photo_ids is not None:
        result.photo_ids = len(photo_ids)
    tag = f"갤러리 {gallery_id}" + (f" 샤드 {shard.index}/{shard.total}" if shard else "") + (
        f" 사진 {len(photo_ids)}장" if photo_ids is not None else "")

    storage = PhotoStorage(settings.s3_bucket, max_concurrency=settings.download_workers)

    with db.connect(settings) as connection:
        if photo_ids is not None:
            # v2 스트리밍 경로 — 잠금·조정자 없음. 배정한 쪽이 겹치지 않게 했다.
            all_targets = db.fetch_by_ids(connection, photo_ids)
            result.targets = len(all_targets)
            fan_out = None
            shard = None
        else:
            # 모델을 올리기 전에 잠금부터 본다. 겹친 실행이 수 초짜리 모델 로드를 치르고 나서야
            # 물러나는 것보다 낫다. 잠금은 연결이 닫힐 때(with 블록 끝, 또는 프로세스 종료) 풀린다.
            if not db.try_lock_gallery(connection, gallery_id, shard.index if shard else 0):
                result.skipped = ALREADY_RUNNING
                result.elapsed_seconds = time.monotonic() - started
                log.info("%s: 다른 실행이 진행 중 -- 건너뜀", tag)
                return result.to_dict()

            # 대상은 갤러리 전체를 같은 순서로 읽는다 — 샤드는 그 목록에서 자기 몫만 고른다. run_started_at 이 있으면
            # (force 이거나 그 재호출) 그 이후 벡터만 건너뛴다.
            all_targets = db.fetch_targets(connection, gallery_id, force, run_started_at)
            result.targets = len(all_targets)

        if shard is None and fan_out is not None:
            n = plan_shards(len(all_targets), settings)
            if n > 1:
                result.coordinator = True
                result.shards = n
                result.fanned_out = fan_out(n, run_started_at)
                result.elapsed_seconds = time.monotonic() - started
                if not result.fanned_out:
                    raise RuntimeError(f"{tag}: 샤드 {n}개 호출 실패")
                log.info("%s: 조정자 -- %s장을 샤드 %s개로 (runStartedAt=%s)", tag, len(all_targets), n, run_started_at)
                return result.to_dict()

        targets = shard.select(all_targets) if shard else all_targets
        result.targets = len(targets)
        log.info("%s: 대상 %s장 (force=%s, runStartedAt=%s)", tag, len(targets), force, run_started_at)

        embedder = model.load_from(settings)

        # 지금까지 가장 오래 걸린 배치. 데드라인 판단은 이 값 + 여유로 한다 -- 배치 시간은
        # 원본 크기(HEIC·4천만 화소)에 따라 크게 흔들려서 평균보다 최댓값이 안전하다.
        longest_batch = 0.0

        batches = _chunked(targets, settings.batch_size)
        # 원본 GET을 미리 걸어 둔다. 배치 index를 처리하기 시작할 때 index+1·index+2의 GET이
        # 이미 풀에 들어가 있다. 처리 순서·PUT·commit은 여전히 이 스레드가 한 장씩 한다.
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
                loaded_refs = []
                loaded_images = []
                loaded_keys = []
                loaded_metadata = []

                for ref, download in zip(batch, downloads):
                    result.attempted += 1
                    try:
                        # GET 실패는 여기서 터진다 -- 아래 except가 그 사진 하나를 failed로 접는다.
                        data = download.result()
                        original = images.open_original(data)
                        # prepare는 바이트를 다시 연다. 축소 디코드가 이미지 객체의 크기를 바꾸므로
                        # metadata가 볼 original과 같은 객체를 쓰면 안 된다.
                        prepared = images.prepare(data, settings.resize_long_edge)
                        # EXIF는 회전·축소 전 원본에서. 실패해도 이 사진을 버리지 않는다(best-effort).
                        photo_metadata = _read_metadata(ref, original, len(data), result)

                        # ② 미리보기를 먼저 올린다. 여기서 실패하면 아래 encode에 들어가지 않는다 --
                        # 벡터만 남는 사진을 만들지 않기 위해서다. IAM에 s3:PutObject가 없으면
                        # 사진마다 여기서 실패하고, 그 사진들은 failed로 드러난다.
                        key = images.preview_key_for(ref.storage_key)
                        jpeg = images.to_jpeg(prepared, settings.preview_quality)
                        storage.write(key, jpeg, "image/jpeg")

                        # ③ 모델 입력은 S3에 올린 바로 그 바이트다. prepared를 그대로 쓰면 JPEG 압축
                        # 전 픽셀을 임베딩하게 되어 photoselect가 보는 파일과 어긋난다.
                        model_input = images.open_preview(jpeg)

                        # 네 리스트를 여기서 함께 늘린다. 위 어느 줄에서 실패해도 이 사진은 어느
                        # 리스트에도 들어가지 않아, 아래에서 벡터와 짝이 어긋날 일이 없다.
                        loaded_refs.append(ref)
                        loaded_images.append(model_input)
                        loaded_keys.append(key)
                        loaded_metadata.append(photo_metadata)
                    except Exception:
                        # 한 장이 잡 전체를 죽이지 않게 한다. 실패한 사진은 photo_analysis에 벡터가
                        # 없는 채로 남으므로, 다시 호출하면 fetch_targets가 자연히 다시 집어 온다.
                        log.exception("사진을 처리하지 못했습니다: %s", ref.storage_key)
                        result.failed.append(ref.storage_key)

                if loaded_images:
                    vectors = embedder.encode(loaded_images)

                    stored = db.store_embeddings(
                        connection,
                        zip(loaded_refs, vectors, loaded_keys, loaded_metadata),
                        model_id=settings.model_id,
                        set_status=getattr(settings, "set_status", None),
                    )

                    # ④ 배치 단위로 커밋한다. 중간에 죽어도 그때까지의 벡터·미리보기는 남고,
                    # 다시 부르면 fetch_targets가 나머지만 집어 온다.
                    connection.commit()
                    result.processed += stored

                batch_seconds = time.monotonic() - batch_started
                longest_batch = max(longest_batch, batch_seconds)
                if loaded_images:
                    log.info(
                        "진행 %s/%s (장당 %.2fs)",
                        result.processed, result.targets, batch_seconds / len(loaded_images),
                    )
        finally:
            # 데드라인으로 멈췄으면 아직 시작하지 않은 GET은 취소한다. 진행 중인 GET은 끝까지
            # 받지만 결과는 버려진다 -- 기다리지 않는다(wait=False). 남은 사진은 다음 호출이 집는다.
            pool.shutdown(wait=False, cancel_futures=True)

    result.elapsed_seconds = time.monotonic() - started
    log.info("완료: %s", result.to_dict())
    return result.to_dict()


def _read_metadata(
    ref: db.PhotoRef,
    original: Image.Image,
    byte_size: int,
    result: RunResult,
) -> metadata.PhotoMetadata | None:
    """원본에서 촬영 정보를 읽는다. 실패하면 None.

    바깥 try와 분리된 것이 핵심이다. 여기서 예외를 그대로 올려보내면 사진이 '처리하지
    못했다'로 분류되어 미리보기도 벡터도 적재되지 않는다 -- EXIF 파싱 문제 하나가 임베딩
    실패로 둔갑한다. 이 값이 없어도 사진은 멀쩡히 보이므로 best-effort다.

    회전·축소를 거치기 전의 이미지를 넘겨야 한다. 그쪽은 Orientation 태그가 지워지고 크기도
    원본이 아니다.
    """
    try:
        return metadata.extract(original, byte_size)
    except Exception:
        log.exception("촬영 정보를 읽지 못했습니다: %s", ref.storage_key)
        result.metadata_failed.append(ref.storage_key)
        return None


def _chunked(items: list, size: int) -> list[list]:
    return [items[start:start + size] for start in range(0, len(items), size)]
