"""갤러리 하나의 임베딩을 계산해 적재하고, 같은 김에 미리보기 파생본과 촬영 정보를 만든다.

파생본과 EXIF를 여기서 만드는 이유는 이 잡이 이미 원본을 받아 HEIC를 디코딩하고 EXIF 회전과
축소까지 마친 이미지를 들고 있기 때문이다. 비싼 부분은 이미 지불했고 남은 것은 인코딩과
PUT 하나, 태그를 훑는 일뿐이다. 별도 잡으로 빼면 같은 이미지를 두 번 받아 두 번 디코딩하게
된다. 앱 서버는 이미지 바이트를 만지지 않으므로 애초에 그쪽에는 선택지가 없다.

진입점(`handler.py` / `__main__.py`)이 둘이고 본체는 이 함수 하나다. Lambda로 감싸기 전에
로컬에서 실제 S3·RDS를 상대로 같은 코드를 검증할 수 있어야 해서 이렇게 갈라 두었다.
나중에 Fargate로 옮겨도 바뀌는 것은 진입점뿐이다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from PIL import Image

from embedder import db, images, metadata, model
from embedder.config import Settings
from embedder.storage import PhotoStorage

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    gallery_id: int
    targets: int = 0
    processed: int = 0
    failed: list[str] = field(default_factory=list)
    #: 벡터는 나왔지만 파생본만 올리지 못한 사진. failed와 섞으면 안 된다 -- 이쪽은
    #: 임베딩이 성공했으므로 다시 불러도 fetch_targets가 집어 오지 않는다.
    previews_failed: list[str] = field(default_factory=list)
    #: 벡터는 나왔지만 촬영 정보만 읽지 못한 사진. previews_failed와 같은 취급이다 --
    #: 상세 화면에 정보가 덜 나올 뿐 사진은 멀쩡히 보이고 임베딩도 끝나 있다.
    metadata_failed: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "galleryId": self.gallery_id,
            "targets": self.targets,
            "processed": self.processed,
            "failed": self.failed,
            "previewsFailed": self.previews_failed,
            "metadataFailed": self.metadata_failed,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
        }


def run(gallery_id: int, force: bool = False, settings: Settings | None = None) -> dict:
    started = time.monotonic()
    settings = settings or Settings.from_env()
    result = RunResult(gallery_id=gallery_id)

    storage = PhotoStorage(settings.s3_bucket)
    embedder = model.load_from(settings)

    with db.connect(settings) as connection:
        targets = db.fetch_targets(connection, gallery_id, force)
        result.targets = len(targets)
        log.info("갤러리 %s: 대상 %s장 (force=%s)", gallery_id, len(targets), force)

        for batch in _chunked(targets, settings.batch_size):
            loaded_refs = []
            loaded_images = []
            loaded_metadata = []

            for ref in batch:
                try:
                    data = storage.read(ref.storage_key)
                    original = images.open_original(data)
                    prepared = images.prepare(original, settings.resize_long_edge)

                    # 세 리스트를 여기서 함께 늘린다. 위 두 줄 중 하나라도 실패하면 이 사진은
                    # 어느 리스트에도 들어가지 않아, 아래에서 벡터와 짝이 어긋날 일이 없다.
                    loaded_refs.append(ref)
                    loaded_images.append(prepared)
                    loaded_metadata.append(_read_metadata(ref, original, len(data), result))
                except Exception:
                    # 한 장이 잡 전체를 죽이지 않게 한다. 실패한 사진은 photo_analysis에 벡터가
                    # 없는 채로 남으므로, 다시 호출하면 fetch_targets가 자연히 다시 집어 온다.
                    log.exception("사진을 읽지 못했습니다: %s", ref.storage_key)
                    result.failed.append(ref.storage_key)

            if not loaded_images:
                continue

            batch_started = time.monotonic()
            vectors = embedder.encode(loaded_images)

            preview_keys = [
                _upload_preview(storage, ref, image, settings, result)
                for ref, image in zip(loaded_refs, loaded_images)
            ]

            stored = db.store_embeddings(
                connection,
                zip(loaded_refs, vectors, preview_keys, loaded_metadata),
                model_id=settings.model_id,
            )

            # 배치 단위로 커밋한다. 중간에 죽어도 그때까지의 벡터는 남고, 다시 부르면
            # fetch_targets가 나머지만 집어 온다.
            connection.commit()
            result.processed += stored

            per_photo = (time.monotonic() - batch_started) / len(loaded_images)
            log.info("진행 %s/%s (장당 %.2fs)", result.processed, result.targets, per_photo)

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

    바깥 try와 분리된 것이 핵심이다. 여기서 예외를 그대로 올려보내면 사진이 '읽지 못했다'로
    분류되어 벡터까지 적재되지 않는다 -- EXIF 파싱 문제 하나가 임베딩 실패로 둔갑한다.
    파생본 업로드와 같은 취급이고, 이유도 같다: 이 값이 없어도 사진은 멀쩡히 보인다.

    회전·축소를 거치기 전의 이미지를 넘겨야 한다. 그쪽은 Orientation 태그가 지워지고 크기도
    원본이 아니다.
    """
    try:
        return metadata.extract(original, byte_size)
    except Exception:
        log.exception("촬영 정보를 읽지 못했습니다: %s", ref.storage_key)
        result.metadata_failed.append(ref.storage_key)
        return None


def _upload_preview(
    storage: PhotoStorage,
    ref: db.PhotoRef,
    image: Image.Image,
    settings: Settings,
    result: RunResult,
) -> str | None:
    """브라우저가 그릴 수 있는 파생본을 올리고 그 키를 돌려준다. 실패하면 None.

    임베딩과 분리된 try인 것이 핵심이다. IAM에 s3:PutObject가 없으면 이 호출이 사진마다
    실패하는데, 바깥 try가 이를 삼키면 사진이 '읽지 못했다'로 분류되어 벡터까지 적재되지
    않는다 -- 미리보기 권한 문제가 임베딩 실패로 둔갑한다.

    파생본이 없어도 사진은 보인다(원본을 그대로 서명해 준다). 그래서 여기서 잡을 멈추지
    않고, 대신 결과의 previewsFailed로 드러낸다.
    """
    key = images.preview_key_for(ref.storage_key)
    try:
        data = images.to_jpeg(image, settings.preview_quality)
        storage.write(key, data, "image/jpeg")
        return key
    except Exception:
        log.exception("미리보기를 올리지 못했습니다: %s", key)
        result.previews_failed.append(ref.storage_key)
        return None


def _chunked(items: list, size: int):
    for start in range(0, len(items), size):
        yield items[start:start + size]
