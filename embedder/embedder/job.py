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

끊김에 대한 태도: 배치(8장)마다 commit하므로 어디서 죽어도 그때까지는 남는다. Lambda에서는 남은
시간(`remaining_seconds`)을 보고 하드 킬 전에 배치 경계에서 스스로 멈춘다 -- 결과에 `stopped`와
`remaining`이 실리고, 재호출은 handler의 몫이다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

from PIL import Image

from embedder import db, images, metadata, model
from embedder.config import Settings
from embedder.storage import PhotoStorage

log = logging.getLogger(__name__)

#: 다른 실행이 같은 갤러리를 잡고 있을 때 결과의 skipped 값.
ALREADY_RUNNING = "already running"


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
        return out


def run(
    gallery_id: int,
    force: bool = False,
    settings: Settings | None = None,
    remaining_seconds: Callable[[], float] | None = None,
) -> dict:
    """갤러리 하나를 처리한다.

    `remaining_seconds`는 실행 환경이 남은 시간을 알려 주는 함수다. Lambda handler가
    `context.get_remaining_time_in_millis`를 감싸 넘기고, 로컬 CLI는 None이다(멈추지 않는다).
    """
    started = time.monotonic()
    settings = settings or Settings.from_env()
    result = RunResult(gallery_id=gallery_id)

    storage = PhotoStorage(settings.s3_bucket)

    with db.connect(settings) as connection:
        # 모델을 올리기 전에 잠금부터 본다. 겹친 실행이 수 초짜리 모델 로드를 치르고 나서야
        # 물러나는 것보다 낫다. 잠금은 연결이 닫힐 때(with 블록 끝, 또는 프로세스 종료) 풀린다.
        if not db.try_lock_gallery(connection, gallery_id):
            result.skipped = ALREADY_RUNNING
            result.elapsed_seconds = time.monotonic() - started
            log.info("갤러리 %s: 다른 실행이 진행 중 -- 건너뜀", gallery_id)
            return result.to_dict()

        embedder = model.load_from(settings)

        targets = db.fetch_targets(connection, gallery_id, force)
        result.targets = len(targets)
        log.info("갤러리 %s: 대상 %s장 (force=%s)", gallery_id, len(targets), force)

        # 지금까지 가장 오래 걸린 배치. 데드라인 판단은 이 값 + 여유로 한다 -- 배치 시간은
        # 원본 크기(HEIC·4천만 화소)에 따라 크게 흔들려서 평균보다 최댓값이 안전하다.
        longest_batch = 0.0

        for batch in _chunked(targets, settings.batch_size):
            if remaining_seconds is not None:
                budget = longest_batch + settings.stop_margin_seconds
                left = remaining_seconds()
                if left < budget:
                    result.stopped = True
                    log.warning(
                        "갤러리 %s: 남은 시간 %.0fs < 배치 예산 %.0fs -- 배치 경계에서 멈춤 (남은 사진 %s장)",
                        gallery_id, left, budget, result.remaining,
                    )
                    break

            batch_started = time.monotonic()
            loaded_refs = []
            loaded_images = []
            loaded_keys = []
            loaded_metadata = []

            for ref in batch:
                result.attempted += 1
                try:
                    data = storage.read(ref.storage_key)
                    original = images.open_original(data)
                    prepared = images.prepare(original, settings.resize_long_edge)
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


def _chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]
