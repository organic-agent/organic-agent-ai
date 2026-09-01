"""고전 기술 지표 — 모델 없이 사진 한 장에서 바로 재는 값. 사진당 수십 ms.

ARNIQA 점수는 스칼라 하나라 "왜 높은가"를 못 말한다. 여기서 재는 값은 분해돼 있어서
근거 문장이 "초점이 또렷하다", "노출이 안정적이다"처럼 **사진별로** 말할 수 있게 한다.
절대값은 카메라·장면마다 다르므로 갤러리/연사 안의 상대 비교로만 쓴다.

    sharpness       그레이스케일 Laplacian 분산. 클수록 초점·디테일이 살아 있다. 흔들림·아웃포커스에 민감
    highlight_clip  밝기 ≥ 250 픽셀 비율 — 날아간 하이라이트
    shadow_clip     밝기 ≤ 5 픽셀 비율 — 뭉개진 그림자
    mean_luma       평균 밝기 0~255 (참고용)
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from scipy.ndimage import laplace

from photoselect.v3.runners.common import load_image

#: 선명도 계산 해상도. 리사이즈가 달라지면 값 스케일이 달라지므로 고정한다.
LONG_EDGE = 1024


def measure(path: str) -> dict[str, float]:
    img = load_image(path, long_edge=LONG_EDGE)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lap = laplace(gray)
    n = gray.size
    return {
        "sharpness": float(lap.var()),
        "highlight_clip": float((gray >= 250).sum() / n),
        "shadow_clip": float((gray <= 5).sum() / n),
        "mean_luma": float(gray.mean()),
    }


def measure_image(img: Image.Image) -> dict[str, float]:
    """이미 열린 이미지용(테스트·스파이크)."""
    img = img.convert("RGB")
    img.thumbnail((LONG_EDGE, LONG_EDGE), Image.LANCZOS)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    lap = laplace(gray)
    n = gray.size
    return {
        "sharpness": float(lap.var()),
        "highlight_clip": float((gray >= 250).sum() / n),
        "shadow_clip": float((gray <= 5).sum() / n),
        "mean_luma": float(gray.mean()),
    }
