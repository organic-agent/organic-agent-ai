"""러너 공통 — 이미지 로드와 가중치 다운로드.

미리보기 파생본을 흉내내기 위해 긴 변 기준으로 리사이즈해서 넣는다
(서비스에서는 embedder가 만든 preview JPEG을 읽는다 — tech-stack.md).
"""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from PIL import Image, ImageOps

try:  # dataset1의 HEIC 스냅 대응 (설치 안 돼 있으면 JPG만)
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass

WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "weights"
PREVIEW_LONG_EDGE = 1600  # embedder 미리보기 가정 — 스파이크에서 해상도 축도 실험 대상


def load_image(path: str, long_edge: int = PREVIEW_LONG_EDGE) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale < 1:
        img = img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img


def fetch_weight(url: str, filename: str, sha256: str | None = None) -> Path:
    """고정 URL에서 가중치를 받아 weights/에 캐시. 스파이크 전용 — 서비스는 빌드 시 번들."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = WEIGHTS_DIR / filename
    if not dest.exists():
        print(f"다운로드: {url} → {dest}")
        urllib.request.urlretrieve(url, dest)
    if sha256 is not None:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        if digest != sha256:
            dest.unlink()
            raise RuntimeError(f"체크섬 불일치: {filename} ({digest})")
    return dest
