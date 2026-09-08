"""갤러리 하나의 미리보기 파생본을 만들어 올리고, 그 파일로 임베딩을 계산해 적재한다. 촬영 정보도 같이.

순서가 계약이다: **① 원본을 열어 EXIF를 읽고 1024px로 다듬는다 → ② JPEG로 인코딩해 S3에 PUT →
③ PUT이 성공한 그 JPEG 바이트를 다시 열어 DINOv3에 넣는다 → ④ 벡터·preview_key·EXIF를 한
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

**운영은 `photo_ids` 하나다**(#100): wes 스위퍼가 UPLOADED·벡터 없음인 사진을 50장씩 배정해(`photos.dispatched_at`)
`{galleryId, photoIds}` 로 부른다. 배정 자체가 원자적이라 갤러리 잠금이 필요 없고, 샤딩·조정자·자기 재호출도 없다 —
남거나 실패한 장은 wes 가 `dispatched_at` 을 되돌려 다시 배정한다. `photos.status` 는 쓰지 않는다(V15 부터 권한도 없다).

갤러리 전체 경로는 **로컬 전용**으로 남는다: wes `scripts/local-ai.sh` 가 사진 목록 없이 CLI 를 부른다.
재계산은 플래그가 아니라 `photo_analysis` 행 삭제(관리자 재처리)다.

끊김에 대한 태도: 배치(8장)마다 commit하므로 어디서 죽어도 그때까지는 남는다. Lambda에서는 남은
시간(`remaining_seconds`)을 보고 하드 킬 전에 배치 경계에서 스스로 멈춘다 -- 결과에 `stopped`와
`remaining`이 실리고, 재호출은 handler의 몫이다.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from embedder import db, images, metadata, model
from embedder.config import Settings
from embedder.storage import PhotoStorage

log = logging.getLogger(__name__)

#: 지금 처리하는 배치보다 몇 배치 앞까지 원본 GET을 미리 걸어 두는가. 2면 메모리에 원본이 최대
#: 세 배치(24장, 13MB 원본이면 ~300MB) 올라간다. Lambda 3GB에서 모델과 함께 두어도 남는다.
PREFETCH_BATCHES = 2


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
    #: 데드라인 때문에 배치 경계에서 멈췄다. 남은 사진은 remaining — wes 가 다시 배정한다.
    stopped: bool = False
    elapsed_seconds: float = 0.0
    #: 사진 목록 호출이면 요청 장수(#73). 갤러리 호출(로컬 CLI)이면 None.
    photo_ids: int | None = None

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
        if self.photo_ids is not None:
            out["photoIds"] = self.photo_ids
        return out


def run(
    gallery_id: int,
    settings: Settings | None = None,
    remaining_seconds: Callable[[], float] | None = None,
    photo_ids: list[int] | None = None,
) -> dict:
    """사진 id 목록(운영), 또는 갤러리에서 아직 벡터가 없는 사진 전체(로컬 CLI).

    `photo_ids` 가 있으면 **그 목록만** 임베딩한다 — wes 스위퍼가 배정해 부르는 스트리밍 경로(#73). 대상 조회·잠금이 없다.
    없으면 `db.fetch_targets` 로 갤러리를 훑는다(로컬 `local-ai.sh`).

    `remaining_seconds`는 실행 환경이 남은 시간을 알려 주는 함수다. Lambda handler가
    `context.get_remaining_time_in_millis`를 감싸 넘기고, 로컬 CLI는 None이다(멈추지 않는다).
    """
    started = time.monotonic()
    settings = settings or Settings.from_env()
    result = RunResult(gallery_id=gallery_id)
    if photo_ids is not None:
        result.photo_ids = len(photo_ids)
    tag = f"갤러리 {gallery_id}" + (f" 사진 {len(photo_ids)}장" if photo_ids is not None else "")

    storage = PhotoStorage(settings.s3_bucket, max_concurrency=settings.download_workers)

    with db.connect(settings) as connection:
        if photo_ids is not None:
            # v2 스트리밍 경로 — 잠금·조정자 없음. 배정한 쪽(wes)이 겹치지 않게 했다.
            targets = db.fetch_by_ids(connection, photo_ids)
        else:
            targets = db.fetch_targets(connection, gallery_id)

        result.targets = len(targets)
        log.info("%s: 대상 %s장", tag, len(targets))

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
                    )

                    # ④ 배치 단위로 커밋한다. 중간에 죽어도 그때까지의 벡터·미리보기는 남고,
                    # 남은 사진은 wes 가 다시 배정한다(로컬 CLI 면 fetch_targets 가 나머지만 집어 온다).
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
