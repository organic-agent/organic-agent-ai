"""러너 공통 — 가중치 캐시(weights/). 이미지 로드는 같은 층의 `images.py`(torch 없음)."""

from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path

from score.config.settings import MODULE_ROOT

WEIGHTS_DIR = MODULE_ROOT / "weights"


def fetch_weight(url: str, filename: str, sha256: str | None = None) -> Path:
    """고정 URL에서 가중치를 받아 weights/에 캐시. 프로덕션 이미지는 빌드 시 번들한다."""
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
