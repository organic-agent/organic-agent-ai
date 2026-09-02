from __future__ import annotations

import sys
import unittest
from pathlib import Path


EMBEDDER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EMBEDDER_ROOT))

from embedder import quality


class _GrayImage:
    def __init__(self, rows: list[list[int]]):
        self.rows = rows
        self.size = (len(rows[0]), len(rows))

    def convert(self, mode: str):
        self.mode = mode
        return self

    def copy(self):
        return _GrayImage([row[:] for row in self.rows])

    def thumbnail(self, size: tuple[int, int]) -> None:
        return None

    def getdata(self):
        return [value for row in self.rows for value in row]


class TechnicalQualityTest(unittest.TestCase):
    def test_score_is_bounded_and_signals_are_json_primitives(self) -> None:
        image = _GrayImage([
            [0, 255, 0, 255],
            [255, 0, 255, 0],
            [32, 128, 224, 128],
            [64, 96, 160, 192],
        ])

        result = quality.analyze(image, (6000, 4000))

        self.assertGreaterEqual(result.score, 0.0)
        self.assertLessEqual(result.score, 100.0)
        self.assertEqual("technical-v1", result.signals["algorithmVersion"])
        self.assertEqual(24.0, result.signals["megapixels"])
        self.assertIn("underexposedRatio", result.signals)
        self.assertIn("sharpnessScore", result.signals)


if __name__ == "__main__":
    unittest.main()
