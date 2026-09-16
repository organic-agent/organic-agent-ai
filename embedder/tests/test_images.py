"""prepare의 축소 디코드 -- 원본 객체를 건드리지 않고, JPEG만 1/2·1/4로 풀고, 회전·크기는 그대로."""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path

from PIL import Image

EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))

from embedder.service import images

ORIENTATION_TAG = 274
ROTATE_270 = 6  # 세로로 찍은 사진의 흔한 값 -- 파일은 가로, 보는 방향은 세로


def _jpeg(width: int, height: int, orientation: int | None = None) -> bytes:
    image = Image.new("RGB", (width, height), (120, 80, 40))
    exif = Image.Exif()
    if orientation is not None:
        exif[ORIENTATION_TAG] = orientation
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85, exif=exif.tobytes())
    return buffer.getvalue()


class ReducedDecodeTest(unittest.TestCase):
    def test_picks_half_scale_for_8mp_and_quarter_for_45mp(self) -> None:
        for size, expected in (((3504, 2336), (1752, 1168)), ((8192, 5464), (2048, 1366))):
            image = Image.open(io.BytesIO(_jpeg(*size)))
            images.request_reduced_decode(image, 1024)
            self.assertEqual(expected, image.size, size)
            self.assertEqual(expected, image.size)

    def test_leaves_small_images_and_non_jpeg_alone(self) -> None:
        small = Image.open(io.BytesIO(_jpeg(1400, 900)))
        images.request_reduced_decode(small, 1024)
        self.assertEqual((1400, 900), small.size)

        buffer = io.BytesIO()
        Image.new("RGB", (4000, 3000)).save(buffer, format="PNG")
        png = Image.open(io.BytesIO(buffer.getvalue()))
        images.request_reduced_decode(png, 1024)
        self.assertEqual((4000, 3000), png.size)


class PrepareTest(unittest.TestCase):
    def test_output_is_1024_long_edge_rotated_rgb(self) -> None:
        data = _jpeg(3504, 2336, orientation=ROTATE_270)

        prepared = images.prepare(data, 1024)

        self.assertEqual((683, 1024), prepared.size)
        self.assertEqual("RGB", prepared.mode)
        # 회전을 픽셀에 구웠으니 Orientation 태그는 더 이상 없어야 한다.
        self.assertIsNone(prepared.getexif().get(ORIENTATION_TAG))

    def test_original_for_metadata_keeps_its_size(self) -> None:
        data = _jpeg(3504, 2336, orientation=ROTATE_270)
        original = images.open_original(data)

        images.prepare(data, 1024)

        # 축소 디코드는 prepare가 따로 연 객체에만 걸린다. metadata.extract가 볼 원본은 크기도
        # Orientation 태그도 그대로다 -- 같은 객체를 넘겼다면 (1752, 1168)이 됐을 것이다.
        self.assertEqual((3504, 2336), original.size)
        self.assertEqual(ROTATE_270, original.getexif().get(ORIENTATION_TAG))

    def test_matches_full_decode_on_smooth_content(self) -> None:
        """사진에 가까운 부드러운 내용에서 축소 디코드와 전체 디코드의 차이가 눈에 안 띄는 수준인지.

        백색 잡음은 여기 넣지 않는다 -- 8x8 블록의 고주파를 버리는 축소 디코드와 LANCZOS는 잡음을
        다르게 평균 내서 평균 4/255까지 벌어지지만, 실제 사진(2026-09-05 실측)에서는 0.4/255였다.
        """
        channels = [
            Image.radial_gradient("L").resize((3504, 2336), Image.Resampling.BICUBIC),
            Image.linear_gradient("L").resize((3504, 2336), Image.Resampling.BICUBIC),
            Image.linear_gradient("L").rotate(90, expand=True).resize((3504, 2336), Image.Resampling.BICUBIC),
        ]
        image = Image.merge("RGB", channels)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=85)
        data = buffer.getvalue()

        full = Image.open(io.BytesIO(data)).convert("RGB")
        full.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        reduced = images.prepare(data, 1024)

        self.assertEqual(full.size, reduced.size)
        diff = sum(abs(a - b) for a, b in zip(full.tobytes(), reduced.tobytes())) / len(full.tobytes())
        self.assertLess(diff, 2.0, f"mean abs diff {diff}")


if __name__ == "__main__":
    unittest.main()
