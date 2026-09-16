"""로컬 데이터셋 폴더 — DB 없이 돌리는 `--local` · `--list` 의 사진 소스.

갤러리 이름 = 스파이크 매니페스트의 group 과 같은 규칙(`dataset1/류지혜고객님 (2)` 처럼 상위 2단계 경로).
사진 id = 루트 기준 상대 경로. `PhotoRef` 만 같으면 나머지 코드는 DB 모드와 같다.
"""

from __future__ import annotations

from pathlib import Path

from score.domain.photo import PhotoRef

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}


def list_galleries(dataset_root: Path) -> list[tuple[str, int]]:
    """(갤러리 이름, 장수) 목록. `score --list`용."""
    counts: dict[str, int] = {}
    for p in sorted(dataset_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = p.relative_to(dataset_root)
        if any(part.endswith(".livephoto") for part in rel.parent.parts):
            continue
        g = "/".join(rel.parts[:-1][:2]) or "root"
        counts[g] = counts.get(g, 0) + 1
    return sorted(counts.items(), key=lambda kv: -kv[1])


def load_local(dataset_root: Path, gallery: str, limit: int | None = None,
               jpg_only: bool = True) -> list[PhotoRef]:
    """갤러리 하나의 사진 목록. 파일명 순 — 연사 클러스터링이 이 순서를 쓴다."""
    base = dataset_root / gallery
    if not base.is_dir():
        raise SystemExit(f"갤러리 폴더가 없다: {base}\n  --list 로 이름을 확인할 것")
    refs: list[PhotoRef] = []
    for p in sorted(base.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        if jpg_only and p.suffix.lower() not in (".jpg", ".jpeg"):
            continue
        rel = p.relative_to(dataset_root)
        if any(part.endswith(".livephoto") for part in rel.parent.parts):
            continue
        # 갤러리가 2단계보다 깊은 하위 폴더를 가지면 그 사진은 상위 갤러리에 속하지 않는다
        if "/".join(rel.parts[:-1][:2]) != gallery:
            continue
        refs.append(PhotoRef(photo_id=str(rel), path=str(p.resolve())))
    if limit:
        refs = refs[:limit]
    return refs
