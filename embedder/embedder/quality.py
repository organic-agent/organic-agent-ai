"""가벼운 기술 품질 신호. 결과는 운영자 판단 보조이며 사진을 자동 폐기하지 않는다."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TechnicalQuality:
    score: float
    signals: dict


def analyze(image, original_size: tuple[int, int]) -> TechnicalQuality:
    """노출·에지·해상도를 0..100으로 정규화한다.

    계산량을 제한하려고 회색조 256px 표본을 쓴다. 얼굴·미학·내용을 판단하지 않으며
    어떤 점수에서도 삭제나 숨김 같은 제품 동작을 일으키지 않는다.
    """
    sample = image.convert("L").copy()
    sample.thumbnail((256, 256))
    width, height = sample.size
    pixels = [float(value) / 255.0 for value in sample.getdata()]
    if not pixels or width <= 0 or height <= 0:
        raise ValueError("EMPTY_IMAGE")

    mean_luminance = sum(pixels) / len(pixels)
    underexposed_ratio = sum(value <= 0.02 for value in pixels) / len(pixels)
    overexposed_ratio = sum(value >= 0.98 for value in pixels) / len(pixels)

    differences: list[float] = []
    for y in range(height):
        row = y * width
        for x in range(width - 1):
            differences.append(abs(pixels[row + x + 1] - pixels[row + x]))
    for y in range(height - 1):
        row = y * width
        next_row = row + width
        for x in range(width):
            differences.append(abs(pixels[next_row + x] - pixels[row + x]))
    edge_strength = sum(differences) / max(1, len(differences))

    exposure_penalty = min(1.0, underexposed_ratio + overexposed_ratio)
    midtone_penalty = min(1.0, abs(mean_luminance - 0.5) / 0.5)
    exposure_score = 100.0 * max(0.0, 1.0 - 0.75 * exposure_penalty - 0.25 * midtone_penalty)
    sharpness_score = min(100.0, edge_strength * 1000.0)
    original_width, original_height = original_size
    megapixels = max(0, original_width) * max(0, original_height) / 1_000_000.0
    resolution_score = min(100.0, megapixels / 12.0 * 100.0)
    score = round(0.50 * sharpness_score + 0.35 * exposure_score + 0.15 * resolution_score, 2)

    signals = {
        "algorithmVersion": "technical-v1",
        "originalWidth": original_width,
        "originalHeight": original_height,
        "megapixels": round(megapixels, 3),
        "meanLuminance": round(mean_luminance, 4),
        "underexposedRatio": round(underexposed_ratio, 4),
        "overexposedRatio": round(overexposed_ratio, 4),
        "edgeStrength": round(edge_strength, 6),
        "exposureScore": round(exposure_score, 2),
        "sharpnessScore": round(sharpness_score, 2),
        "resolutionScore": round(resolution_score, 2),
    }
    return TechnicalQuality(score=max(0.0, min(100.0, score)), signals=signals)
