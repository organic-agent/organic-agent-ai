"""golden 로더 — 두 형식.

- `.xlsx` (`golden/파일명_정리.xlsx`): 구분 · 번호 · 파일명 세 열, A컷=전체 보정 · B컷=배경 보정.
  openpyxl 없이 zipfile + 정규식으로 읽는다(인라인 문자열 xlsx).
- `.json` (`golden/dataset1-golden.json`): 파일명 목록. `{"files": [...]}` 또는 `[...]`,
  컷을 적고 싶으면 `[{"cut": "A", "file": "..."}]`. 컷이 없으면 A컷으로 본다.
"""

from __future__ import annotations

import html
import json
import re
import zipfile
from pathlib import Path

from preference.domain.golden import GoldenItem

_CELL = re.compile(r'<c r="([A-Z]+)(\d+)"[^>]*?(?:/>|>(.*?)</c>)', re.S)
_TEXT = re.compile(r"<t[^>]*>(.*?)</t>|<v>(.*?)</v>", re.S)
_FILE = re.compile(r"^[\w.-]+\.(jpe?g|png|heic)$", re.I)


def load_golden(path: Path) -> list[GoldenItem]:
    path = Path(path)
    if path.suffix.lower() == ".json":
        return _load_json(path)
    return _load_xlsx(path)


def _load_json(path: Path) -> list[GoldenItem]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    rows = doc.get("files", []) if isinstance(doc, dict) else doc
    out = [
        GoldenItem(cut=str(r.get("cut", "A")).upper()[:1], file_name=str(r["file"]))
        if isinstance(r, dict) else GoldenItem(cut="A", file_name=str(r))
        for r in rows
    ]
    if not out:
        raise ValueError(f"{path}: 파일명을 하나도 못 읽었다")
    return out


def _load_xlsx(path: Path) -> list[GoldenItem]:
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
