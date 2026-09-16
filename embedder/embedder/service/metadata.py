"""원본에서 촬영 정보(EXIF)를 읽는다.

추출 실패가 임베딩을 죽이면 안 된다. 예외는 밖으로 내보내고 부르는 쪽(job.py)이 best-effort 로 감싼다.
"""

from __future__ import annotations

from datetime import datetime

from PIL import Image
from PIL.ExifTags import IFD, Base as ExifTag

from embedder.domain.photo import PhotoMetadata

#: 90도 회전이 걸린 Orientation. 파일의 가로·세로가 사람이 보는 방향과 반대다.
_ROTATED_ORIENTATIONS = frozenset({5, 6, 7, 8})

#: photos.camera_make / camera_model 은 VARCHAR(100). 이상한 파일 하나가 UPDATE 를 실패시키지 않게 자른다.
_MAX_TEXT_LENGTH = 100

#: photos.exposure_time 은 VARCHAR(30).
_MAX_EXPOSURE_LENGTH = 30


def extract(image: Image.Image, byte_size: int) -> PhotoMetadata:
    """회전·축소 전 원본에서 읽는다. `images.prepare()` 를 지난 이미지는 크기와 Orientation 이 이미 바뀌어 있다.

    EXIF 가 없는 파일도 정상이다. 그때는 크기와 바이트 수만 채워진다.
    """
    exif = image.getexif()
    # 셔터·조리개·ISO 는 하위 Exif IFD 에, 제조사·모델·Orientation 은 최상위에 있다.
    photo_ifd = exif.get_ifd(IFD.Exif) if exif else {}

    width, height = _oriented_size(image, exif)

    return PhotoMetadata(
        # DateTimeOriginal 이 찍은 순간. DateTime 은 파일 수정 시각이라 차선책으로만 쓴다.
        taken_at=_parse_datetime(
            photo_ifd.get(ExifTag.DateTimeOriginal) or exif.get(ExifTag.DateTime)
        ),
        camera_make=_text(exif.get(ExifTag.Make)),
        camera_model=_text(exif.get(ExifTag.Model)),
        exposure_time=_exposure(photo_ifd.get(ExifTag.ExposureTime)),
        f_number=_number(photo_ifd.get(ExifTag.FNumber)),
        iso=_iso(photo_ifd.get(ExifTag.ISOSpeedRatings)),
        width=width,
        height=height,
        byte_size=byte_size,
    )


def _oriented_size(image: Image.Image, exif) -> tuple[int, int]:
    """사람이 보는 방향의 가로·세로. 세로 사진은 파일 안에서 누워 있고 방향만 Orientation 에 적혀 있다."""
    width, height = image.size
    if exif and exif.get(ExifTag.Orientation) in _ROTATED_ORIENTATIONS:
        return height, width
    return width, height


def _parse_datetime(value) -> datetime | None:
    """EXIF 의 `2026:05:16 14:32:10` 표기. 타임존 없는 벽시계 그대로다."""
    text = _text(value, limit=None)
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        # 시계가 비어 있던 카메라의 "0000:00:00 00:00:00" 이 여기 걸린다.
        return None


def _exposure(value) -> str | None:
    """셔터 속도를 `1/200`, `2.5` 같은 문자열로. 실수로 담으면 화면에서 되돌릴 때 반올림이 낀다."""
    seconds = _number(value)
    if seconds is None or seconds <= 0:
        return None

    if seconds >= 1:
        return f"{seconds:g}"[:_MAX_EXPOSURE_LENGTH]
    return f"1/{round(1 / seconds)}"[:_MAX_EXPOSURE_LENGTH]


def _number(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _iso(value) -> int | None:
    # 값이 튜플로 오는 카메라가 있다.
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None

    number = _number(value)
    return int(number) if number is not None else None


def _text(value, limit: int | None = _MAX_TEXT_LENGTH) -> str | None:
    """문자열 태그 정리. 바이트로 오거나 고정 길이 필드를 NUL 로 채워 보내는 카메라가 있다."""
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    text = str(value).replace("\x00", "").strip()
    if not text:
        return None
    return text if limit is None else text[:limit]
