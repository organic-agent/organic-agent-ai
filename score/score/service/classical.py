"""고전 기술 지표 — 모델 없이 사진 한 장에서 바로 재는 값. 사진당 수십 ms.

ARNIQA 점수는 스칼라 하나라 "왜 높은가"를 못 말한다. 여기서 재는 값은 분해돼 있어서
근거 문장이 "초점이 또렷하다", "노출이 안정적이다"처럼 **사진별로** 말할 수 있게 한다.
절대값은 카메라·장면마다 다르므로 갤러리/연사 안의 상대 비교로만 쓴다.

    sharpness       그레이스케일 Laplacian 분산. 클수록 초점·디테일이 살아 있다. 흔들림·아웃포커스에 민감
    highlight_clip  밝기 ≥ 250 픽셀 비율 — 날아간 하이라이트
    shadow_clip     밝기 ≤ 5 픽셀 비율 — 뭉개진 그림자
    mean_luma       평균 밝기 0~255 (참고용)
    bg_luma         테두리 링의 median 밝기 0~255 — 배경이 검은가 흰가. categorize 가 배정 검증에 쓴다(#117·#118)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace

from PIL import Image

from score.infrastructure.images import as_image

#: 선명도 계산 해상도. 리사이즈가 달라지면 값 스케일이 달라지므로 고정한다.
LONG_EDGE = 1024

#: 배경 밝기를 대신 재는 테두리 링의 두께(짧은 변 기준 비율). 인물은 대개 가운데, 배경은 가장자리다.
BORDER_RATIO = 0.10


def background_luma(gray: np.ndarray) -> float:
    """바깥 [BORDER_RATIO] 링의 median 밝기 — 배경색의 대리값.

    평균이 아니라 median 인 이유: 링에 팔·드레스 자락·조명이 걸려도 값이 끌려가지 않는다.
    [mean_luma](전체 평균)로는 배경을 못 가른다 — 검은 배경에 흰 드레스가 크게 잡히면
    평균이 올라가 흰 배경 컷과 겹친다(운영 갤러리 25 실측: 검은 세트 전체 median 5~38,
    링 median 4~22 / 화이트 벽 199 · 흰 배경 240).
    네 변을 이어 붙이므로 모서리가 두 번 세어지지만, median 이라 쏠림이 없다.
    """
    h, w = gray.shape
    my, mx = max(1, int(h * BORDER_RATIO)), max(1, int(w * BORDER_RATIO))
    ring = np.concatenate([gray[:my].ravel(), gray[-my:].ravel(),
                           gray[:, :mx].ravel(), gray[:, -mx:].ravel()])
    return float(np.median(ring))


def measure(source: str | Image.Image) -> dict[str, float]:
    """경로 또는 이미 디코드한 PIL 이미지(1600px 미리보기) — 여기서 LONG_EDGE 로 줄인다."""
    img = as_image(source, LONG_EDGE)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lap = laplace(gray)
    n = gray.size
    return {
        "sharpness": float(lap.var()),
        "highlight_clip": float((gray >= 250).sum() / n),
        "shadow_clip": float((gray <= 5).sum() / n),
        "mean_luma": float(gray.mean()),
        "bg_luma": background_luma(gray),
    }
