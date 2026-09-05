"""바이트 -> 모델에 넣을 수 있는 RGB 이미지."""

from __future__ import annotations

import io

import pillow_heif
from PIL import Image, ImageOps

# 아이폰 사진이 HEIC로 올라온다. 서버가 image/heic와 image/heif를 허용하고 있으므로
# (PhotoService.ALLOWED_CONTENT_TYPES) 이게 없으면 그 사진들만 조용히 전부 실패한다.
pillow_heif.register_heif_opener()


def open_original(data: bytes) -> Image.Image:
    """손대지 않은 원본을 연다.

    [prepare]와 나뉘어 있는 이유는 촬영 정보 때문이다. EXIF 회전을 굽고 크기를 줄인 뒤에는
    Orientation 태그가 지워지고 크기도 원본이 아니라서, `metadata.extract()`가 읽을 것이
    남지 않는다. 여는 것과 다듬는 것을 갈라 두면 둘 다 같은 원본을 본다.

    Pillow는 지연 디코딩이라 이 호출만으로는 픽셀을 풀지 않는다. 비싼 일은 [prepare]에서 한다.
    """
    return Image.open(io.BytesIO(data))


#: 축소 디코드가 남겨 둘 여유. 디코더가 목표 긴 변의 이 배수 이상으로 풀고, 나머지는 LANCZOS가
#: 줄인다. 1.5면 긴 변 1536px 이상을 남기는 가장 큰 축소(1/2·1/4·1/8)를 고른다. Pillow thumbnail의
#: 기본값 2.0은 8MP(3504px)에서 1/2(1752px)를 거부해 축소가 안 걸린다. 1.5로 내렸을 때 전체 디코드
#: 결과와의 픽셀 차이는 평균 0.4/255, 99퍼센타일 2/255였다(2026-09-05, 3504x2336 · 4608x3072).
DRAFT_GAP = 1.5


def prepare(data: bytes, long_edge: int) -> Image.Image:
    """모델(과 미리보기)에 넣을 수 있게 다듬는다. 원본 바이트에서 새로 연다.

    [open_original]이 돌려준 이미지를 받지 않고 바이트를 다시 여는 이유는 [request_reduced_decode]
    때문이다. 축소 디코드는 이미지 객체의 크기를 바꾸므로, 같은 객체를 `metadata.extract`와
    나눠 쓰면 촬영 정보의 가로·세로가 축소된 값으로 나간다. 헤더만 다시 읽는 비용은 무시할 만하다.

    순서가 비용을 정한다. 회전·RGB 변환은 픽셀 전체를 복사하므로 **축소 뒤**에 한다.
    앞에 두면 원본을 통째로 풀고 두 번 복사한 다음에야 1024로 줄이는 셈이다. M 시리즈 맥에서
    8~14MP JPEG 기준 68~94ms → 39~70ms(2026-09-05). Lambda의 느린 vCPU에서 배수는 같고 절대값은 커진다.
    """
    image = Image.open(io.BytesIO(data))

    # JPEG는 디코더가 1/2·1/4·1/8 크기로 바로 풀 수 있다(DCT 계수 일부만 씀). 비용이 픽셀 수에
    # 비례해 줄어든다. HEIC·PNG는 이 호출이 아무것도 하지 않고 지나간다.
    request_reduced_decode(image, long_edge)

    # 비율을 유지한 채 줄인다. 제자리 연산이라 반환값이 없다. thumbnail도 draft를 부르지만
    # 정사각 (long_edge, long_edge) 기준이라 3:2 사진에서는 짧은 변이 걸려 축소가 안 걸린다 --
    # 그래서 위에서 비율을 맞춰 직접 요청했다.
    image.thumbnail((long_edge, long_edge), Image.Resampling.LANCZOS)

    # EXIF 회전을 픽셀에 굽는다. 세로로 찍은 사진은 파일 안에서는 가로로 누워 있고 방향만
    # 메타데이터에 적혀 있다. 반영하지 않으면 같은 장면을 90도 돌려서 임베딩하는 셈이라
    # 유사도가 실제보다 낮게 나온다. Orientation 태그는 thumbnail을 지나도 info에 남아 있다.
    prepared = ImageOps.exif_transpose(image)

    # 팔레트 이미지나 알파 채널이 섞여 들어오면 모델 프로세서가 채널 수에서 깨진다.
    return prepared.convert("RGB")


def request_reduced_decode(image: Image.Image, long_edge: int, gap: float = DRAFT_GAP) -> None:
    """아직 픽셀을 풀지 않은 이미지에 "긴 변이 long_edge*gap 이상이면 된다"고 알린다.

    Pillow `draft`는 요청 크기보다 작아지지 않는 가장 큰 축소 배율을 고른다. 1024·gap 1.5면
    (1536, ·)을 요청하므로 3504x2336은 1/2(1752), 6000x4000도 1/2(3000), 8192x5464는 1/4(2048)로
    푼다. 이미 충분히 작은 이미지나 JPEG가 아닌 포맷에는 아무 일도 없다.
    """
    width, height = image.size
    longest = max(width, height)
    if longest <= long_edge * gap:
        return
    scale = longest / (long_edge * gap)
    image.draft(None, (int(width / scale), int(height / scale)))


def to_jpeg(image: Image.Image, quality: int) -> bytes:
    """[prepare]가 돌려준 이미지를 브라우저가 그릴 수 있는 JPEG 바이트로 만든다.

    별도의 디코딩이나 축소가 없다는 점이 중요하다. HEIC 디코딩도, EXIF 회전 굽기도,
    긴 변 축소도 [prepare]에서 이미 끝났다 -- 여기 남은 것은 인코딩뿐이다. 미리보기를
    임베딩 잡 안에서 만드는 이유가 이것이다.

    progressive는 목록에서 수십 장이 동시에 뜰 때 위에서부터 차오르게 한다.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True, progressive=True)
    return buffer.getvalue()


def open_preview(data: bytes) -> Image.Image:
    """S3에 올린 미리보기 JPEG 바이트를 모델 입력으로 연다.

    임베딩은 메모리의 [prepare] 결과가 아니라 **S3에 실제로 올라간 파일**에서 계산한다.
    그래야 "미리보기 없는 벡터"가 구조적으로 생기지 않고, photoselect가 읽는 픽셀과 벡터가
    같은 파일에서 나온다. 미리보기는 이미 EXIF 회전이 구워지고 RGB·긴 변 1024라
    [prepare]를 다시 거치지 않는다 -- 여기서 하는 일은 JPEG 디코드뿐이다(1024px, 장당 수십 ms).

    `load()`를 지금 부른다. Pillow는 지연 디코딩이라 그냥 두면 모델 프로세서가 부를 때
    풀리는데, 그러면 디코드 실패가 배치 전체의 encode 안에서 터진다. 여기서 풀면 실패가
    그 한 장의 `failed`로 떨어진다.
    """
    image = Image.open(io.BytesIO(data))
    image.load()
    return image.convert("RGB")


def preview_key_for(storage_key: str) -> str:
    """원본 키에서 파생본 키를 만든다.

    `galleries/1/{uuid}.heic` -> `previews/galleries/1/{uuid}.jpg`

    원본 키를 그대로 접두사 아래에 붙이므로 둘의 대응이 눈으로 보이고, 접두사 하나로
    수명 주기 규칙이나 일괄 삭제를 걸 수 있다. 확장자는 원본이 무엇이었든 jpg다.
    """
    stem = storage_key.rsplit(".", 1)[0]
    return f"previews/{stem}.jpg"
