"""고전 기술 지표 — 모델 없이 사진 한 장에서 바로 재는 값. 사진당 수십 ms.

ARNIQA 점수는 스칼라 하나라 "왜 높은가"를 못 말한다. 여기서 재는 값은 분해돼 있어서
근거 문장이 "초점이 또렷하다", "노출이 안정적이다"처럼 **사진별로** 말할 수 있게 한다.
절대값은 카메라·장면마다 다르므로 갤러리/연사 안의 상대 비교로만 쓴다.

    sharpness       그레이스케일 Laplacian 분산. 클수록 초점·디테일이 살아 있다. 흔들림·아웃포커스에 민감
    highlight_clip  밝기 ≥ 250 픽셀 비율 — 날아간 하이라이트
    shadow_clip     밝기 ≤ 5 픽셀 비율 — 뭉개진 그림자
    mean_luma       평균 밝기 0~255 (참고용)
    is_bw           흑백 사진이면 1(#164). 색 채널 사이 차이가 거의 없는 사진 — 세피아처럼 색조를 입힌 흑백은 0 이다
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import laplace

from PIL import Image

from score.infrastructure.images import as_image

#: 선명도 계산 해상도. 리사이즈가 달라지면 값 스케일이 달라지므로 고정한다.
LONG_EDGE = 1024
#: 흑백 판정 — 96×96 으로 줄인 사진의 채널 차이(|R−G| · |G−B| 평균)가 이 값 미만이면 흑백. JPEG 색 잡음은 1 안쪽이다.
BW_CHROMA = 2.0
BW_SIDE = 96


def chroma(img: Image.Image) -> float:
    """색이 얼마나 있는가 — 채널 사이 차이의 평균(0~255)."""
    small = np.asarray(img.convert("RGB").resize((BW_SIDE, BW_SIDE)), dtype=np.float32)
    return float((np.abs(small[..., 0] - small[..., 1]).mean() + np.abs(small[..., 1] - small[..., 2]).mean()) / 2)


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
        "is_bw": int(chroma(img) < BW_CHROMA),
    }
