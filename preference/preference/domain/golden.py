"""golden 항목 — 부부(또는 작가)가 고른 사진 한 장. 읽는 쪽은 `repository/golden.py`.

두 컷 모두 학습에서는 양성이다 — 부부가 고른 사진이라는 사실이 라벨이고, 보정 종류는 선호가 아니라 후처리 요청이다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GoldenItem:
    cut: str          # 'A' | 'B'
    file_name: str
