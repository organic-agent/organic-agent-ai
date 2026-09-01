"""갤러리 소스 — "이 갤러리에 어떤 사진이 있고 파일은 어디 있나".

로컬 모드: 데이터셋 폴더. 갤러리 이름 = 스파이크 매니페스트의 group과 같은 규칙
(`dataset1/류지혜고객님 (2)` 처럼 상위 2단계 경로). 사진 id = 루트 기준 상대 경로.

DB 모드: `load_db` — `photos` 중 임베딩이 끝난(EMBEDDED) 사진의 preview_key를 S3에서 내려받는다.
원본이 아니라 미리보기(EXIF 회전·리사이즈 JPEG)다: HEIC 디코드를 피하고 용량이 1/10이며,
점수·태그는 긴 변 1024면 충분하다. 이 모듈의 `PhotoRef`만 같으면 나머지 코드는 안 바뀐다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}


@dataclass(frozen=True)
class PhotoRef:
    photo_id: str     # 갤러리 안에서 유일. 로컬은 상대 경로, DB는 photos.id
    path: str         # 로컬 파일 경로 (DB 모드에서는 임시 다운로드 경로)


def list_galleries(dataset_root: Path) -> list[tuple[str, int]]:
    """(갤러리 이름, 장수) 목록. `analyze --list`용."""
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


def load_db(conn, storage, gallery_id: int, work_dir: Path, limit: int | None = None) -> list[PhotoRef]:
    """DB 모드의 사진 목록. 순서는 wes 화면 순서(display_order, id) — 연사 클러스터링이 이 순서를 쓴다.

    EMBEDDED만 고른다: 임베더가 아직 안 지난 사진은 preview_key가 없고, DINOv3 벡터도 없어
    클러스터에 넣을 수 없다. 그런 사진은 다음 분석 잡에서 잡힌다.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, preview_key FROM photos WHERE gallery_id = %s AND status = 'EMBEDDED' "
            "AND deleted_at IS NULL AND preview_key IS NOT NULL ORDER BY display_order, id",
            (gallery_id,),
        )
        rows = cur.fetchall()
    if limit:
        rows = rows[:limit]
    dest_dir = work_dir / str(gallery_id)
    refs: list[PhotoRef] = []
    for photo_id, key in rows:
        dest = storage.download(key, dest_dir / f"{photo_id}.jpg")
        refs.append(PhotoRef(photo_id=str(photo_id), path=str(dest)))
    return refs
