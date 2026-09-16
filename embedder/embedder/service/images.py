"""바이트 → 모델에 넣을 수 있는 RGB 이미지."""

from __future__ import annotations

import io

import pillow_heif
from PIL import Image, ImageOps

# 아이폰 HEIC. 이게 없으면 그 사진들만 조용히 전부 실패한다.
pillow_heif.register_heif_opener()


def open_original(data: bytes) -> Image.Image:
    """손대지 않은 원본을 연다. `metadata.extract()` 용 — 지연 디코딩이라 픽셀은 풀지 않는다.

    `prepare` 와 나뉜 이유: 회전·축소를 거치면 Orientation 태그와 원본 크기가 사라진다.
    """
    return Image.open(io.BytesIO(data))


#: 축소 디코드가 남길 여유 배수. 긴 변이 long_edge*1.5 이상으로 남는 가장 큰 축소(1/2·1/4·1/8)를 고른다.
#: Pillow 기본 2.0 은 8MP 에서 축소가 안 걸린다. 1.5 에서 전체 디코드와의 픽셀 차이는 평균 0.4/255.
DRAFT_GAP = 1.5


def prepare(data: bytes, long_edge: int) -> Image.Image:
    """모델·미리보기용으로 다듬는다. 축소 → 긴 변 맞춤 → EXIF 회전 → RGB.

    `open_original` 결과를 받지 않고 바이트를 다시 여는 이유: 축소 디코드가 이미지 객체 크기를 바꿔
    metadata 가 볼 가로·세로가 틀어진다. 회전·RGB 변환은 픽셀 전체를 복사하므로 축소 뒤에 한다.
    """
    image = Image.open(io.BytesIO(data))

    # JPEG 는 1/2·1/4·1/8 로 바로 풀 수 있다. HEIC·PNG 는 아무 일도 안 한다.
    request_reduced_decode(image, long_edge)

    # thumbnail 도 draft 를 부르지만 정사각 기준이라 3:2 사진에서 축소가 안 걸린다. 그래서 위에서 직접 요청했다.
    image.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)

    # 세로 사진은 파일 안에서 누워 있다. 회전을 안 구우면 같은 장면을 90도 돌려 임베딩하는 셈이다.
    prepared = ImageOps.exif_transpose(image)

    # 팔레트·알파 채널이 섞이면 모델 프로세서가 채널 수에서 깨진다.
    return prepared.convert("RGB")


def request_reduced_decode(image: Image.Image, long_edge: int, gap: float = DRAFT_GAP) -> None:
    """아직 픽셀을 풀지 않은 이미지에 "긴 변이 long_edge*gap 이상이면 된다"고 알린다.

    Pillow `draft` 는 요청 크기보다 작아지지 않는 가장 큰 축소 배율을 고른다. 이미 작거나 JPEG 가 아니면 아무 일도 없다.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= long_edge * gap:
        return
    scale = longest / (long_edge * gap)
    image.draft(None, (int(width / scale), int(height / scale)))


def to_jpeg(image: Image.Image, quality: int) -> bytes:
    """`prepare` 결과를 JPEG 바이트로. 디코딩·축소는 이미 끝났고 여기는 인코딩뿐이다.

    progressive 는 목록에서 수십 장이 동시에 뜰 때 위에서부터 차오르게 한다.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buffer.getvalue()


def open_preview(data: bytes) -> Image.Image:
    """S3 에 올린 미리보기 JPEG 를 모델 입력으로 연다.

    임베딩은 메모리의 `prepare` 결과가 아니라 실제 올라간 파일에서 계산한다 — "미리보기 없는 벡터"가
    구조적으로 생기지 않고, 화면이 보는 픽셀과 벡터가 같은 파일이다.
    `load()` 를 지금 불러 디코드 실패가 배치 전체가 아니라 이 한 장의 실패로 떨어지게 한다.
    """
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def preview_key_for(storage_key: str) -> str:
    """`galleries/1/{uuid}.heic` → `previews/galleries/1/{uuid}.jpg`. 확장자는 원본이 무엇이든 jpg."""
    stem = storage_key.rsplit(".", 1)[0]
    return f"previews/{stem}.jpg"
