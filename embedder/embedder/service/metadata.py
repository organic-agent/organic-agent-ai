"""원본에서 촬영 정보(EXIF)를 읽는다.

읽는 쪽이 여기인 이유는 미리보기 파생본을 여기서 만드는 이유와 같다 -- 앱 서버는 이미지
바이트를 만지지 않으므로 EXIF를 읽을 방법이 아예 없고, 이 잡은 이미 원본을 받아 열어 두었다.
값을 얻는 데 드는 추가 비용은 태그를 훑는 것뿐이다.

**추출 실패가 임베딩을 죽이면 안 된다.** 여기서 나오는 값은 화면에 곁들이는 정보이고, 벡터는
제품의 본체다. 그래서 이 모듈은 예외를 밖으로 내보내되 부르는 쪽(job.py)이 파생본 업로드와
같은 취급으로 감싼다.
"""

from __future__ import annotations

from datetime import datetime

from PIL import Image
from PIL.ExifTags import IFD, Base as ExifTag

from embedder.domain.photo import PhotoMetadata

#: 90도 회전이 걸린 Orientation. 이 값이면 파일에 적힌 가로·세로가 사람이 보는 방향과 반대다.
_ROTATED_ORIENTATIONS = frozenset({5, 6, 7, 8})

#: photos.camera_make / camera_model 이 VARCHAR(100)이다. 스펙에 길이 제한이 없는 필드라
#: 이상한 파일 하나가 UPDATE 전체를 실패시키지 않도록 여기서 자른다.
_MAX_TEXT_LENGTH = 100

#: photos.exposure_time 이 VARCHAR(30).
_MAX_EXPOSURE_LENGTH = 30


def extract(image: Image.Image, byte_size: int) -> PhotoMetadata:
    """아직 회전·축소를 거치지 않은 원본 이미지에서 촬영 정보를 읽는다.

    `images.prepare()`를 지난 이미지를 넣으면 안 된다. 그쪽은 EXIF 회전을 픽셀에 굽고 긴 변을
    줄인 결과라, 크기는 원본이 아니고 Orientation 태그도 이미 지워져 있다.

    EXIF가 없는 파일(스크린샷, 메타데이터를 떼고 저장한 편집본)도 정상이다. 그때는 크기와
    바이트 수만 채워져 돌아온다.
    """
    exif = image.getexif()
    # 촬영 조건(셔터·조리개·ISO)은 최상위가 아니라 Exif IFD라는 하위 블록에 들어 있다.
    # 제조사·모델·Orientation은 최상위다. 한쪽만 보면 절반이 조용히 비어 나온다.
    photo_ifd = exif.get_ifd(IFD.Exif) if exif else {}

    width, height = _oriented_size(image, exif)

    return PhotoMetadata(
        # DateTimeOriginal이 "찍은 순간"이고 DateTime은 파일이 마지막으로 쓰인 시각이다.
        # 편집을 거치면 뒤쪽이 오늘로 바뀌므로, 없을 때의 차선책으로만 쓴다.
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
    """사람이 보는 방향의 가로·세로.

    세로로 찍은 사진은 파일 안에서 가로로 누워 있고 방향만 Orientation 태그에 적혀 있다.
    그대로 담으면 상세 화면이 세로 사진을 4032x3024로 소개하게 된다 -- 화면에 그려진 그림과
    옆에 적힌 숫자가 서로 다른 말을 한다.
    """
    width, height = image.size
    if exif and exif.get(ExifTag.Orientation) in _ROTATED_ORIENTATIONS:
        return height, width
    return width, height


def _parse_datetime(value) -> datetime | None:
    """EXIF의 `2026:05:16 14:32:10` 표기를 읽는다. 타임존은 없다 -- 벽시계 그대로다."""
    text = _text(value, limit=None)
    if not text:
        return None
    try:
        return datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
    except ValueError:
        # 시각을 못 읽었다고 나머지 값을 버릴 이유는 없다. 비어 있던 카메라가 남기는
        # "0000:00:00 00:00:00"이 실제로 여기 걸린다.
        return None


def _exposure(value) -> str | None:
    """셔터 속도를 사람이 읽는 표기로. `1/200`, `2.5`.

    EXIF에서는 유리수(IFDRational)로 온다. 초 단위 실수로 바꿔 담으면 0.005가 되어 화면에서
    `1/200`로 되돌릴 때 반올림이 끼어들므로, 여기서 문자열로 굳혀 그대로 저장한다.
    """
    seconds = _number(value)
    if seconds is None or seconds <= 0:
        return None

    if seconds >= 1:
        # 1초 이상은 분수로 쓰지 않는다. 2초를 2/1로 적는 카메라는 없다.
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
    # 태그 이름이 ISOSpeedRatings(복수형)인 데서 보이듯 값이 튜플로 오는 카메라가 있다.
    if isinstance(value, (tuple, list)):
        value = value[0] if value else None

    number = _number(value)
    return int(number) if number is not None else None


def _text(value, limit: int | None = _MAX_TEXT_LENGTH) -> str | None:
    """문자열 태그를 다듬는다.

    바이트로 오는 경우가 있고(인코딩이 적혀 있지 않다), 고정 길이 필드를 쓰는 카메라는 남는
    자리를 NUL로 채워 보낸다. 그대로 두면 `Canon\\x00\\x00`이 화면에 나간다.
    """
    if value is None:
        return None

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")

    text = str(value).replace("\x00", "").strip()
    if not text:
        return None
    return text if limit is None else text[:limit]
