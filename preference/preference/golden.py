"""golden 로더 — `golden/파일명_정리.xlsx` (구분 · 번호 · 파일명 세 열, A컷=전체 보정 · B컷=배경 보정).

openpyxl 없이 zipfile + 정규식으로 읽는다(인라인 문자열 xlsx). 두 컷 모두 학습에서는 양성이다 — 부부가 고른 사진이라는
사실이 라벨이고, 보정 종류는 선호가 아니라 후처리 요청이다.
"""

from __future__ import annotations

import html
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

_CELL = re.compile(r'<c r="([A-Z]+)(\d+)"[^>]*?(?:/>|>(.*?)</c>)', re.S)
_TEXT = re.compile(r"<t[^>]*>(.*?)</t>|<v>(.*?)</v>", re.S)
_FILE = re.compile(r"^[\w.-]+\.(jpe?g|png|heic)$", re.I)


@dataclass(frozen=True)
class GoldenItem:
    cut: str          # 'A' | 'B'
    file_name: str


def load_golden(path: Path) -> list[GoldenItem]:
    with zipfile.ZipFile(path) as z:
        sheet = next(n for n in sorted(z.namelist()) if n.startswith("xl/worksheets/sheet"))
        xml = z.read(sheet).decode("utf-8")
    rows: dict[int, dict[str, str]] = {}
    for col, row, body in _CELL.findall(xml):
        if not body:
            continue
        m = _TEXT.search(body)
        if not m:
            continue
        rows.setdefault(int(row), {})[col] = html.unescape(m.group(1) or m.group(2) or "").strip()
    out: list[GoldenItem] = []
    for _, cells in sorted(rows.items()):
        values = [cells[c] for c in sorted(cells)]
        cut = next((v[0].upper() for v in values if re.fullmatch(r"[AaBb]컷", v)), None)
        name = next((v for v in values if _FILE.match(v)), None)
        if cut and name:
            out.append(GoldenItem(cut=cut, file_name=name))
    if not out:
        raise ValueError(f"{path}: 컷·파일명 행을 하나도 못 읽었다")
    return out
