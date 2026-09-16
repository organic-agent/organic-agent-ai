"""갤러리 소스 — "이 갤러리에 어떤 사진이 있고 파일은 어디 있나". `photos` 테이블(DB) 과 데이터셋 폴더(로컬).

로컬 모드: 데이터셋 폴더. 갤러리 이름 = 스파이크 매니페스트의 group과 같은 규칙
(`dataset1/류지혜고객님 (2)` 처럼 상위 2단계 경로). 사진 id = 루트 기준 상대 경로.

DB 모드: `load_db` — `photos` 중 미리보기가 있는(임베더가 지난) 사진의 preview_key를 S3에서 내려받는다.
원본이 아니라 미리보기(EXIF 회전·리사이즈 JPEG)다: HEIC 디코드를 피하고 용량이 1/10이며,
점수·태그는 긴 변 1024면 충분하다. 돌려주는 `PhotoRef`(domain/photo.py)만 같으면 나머지 코드는 안 바뀐다.
"""

from __future__ import annotations

from pathlib import Path

from categorize.domain.photo import PhotoRef

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


def load_db(conn, storage, gallery_id: int, work_dir: Path, limit: int | None = None,
            download: bool = True) -> list[PhotoRef]:
    """DB 모드의 사진 목록. 기본 순서는 wes 화면 순서(display_order, id)이고, 연사 클러스터링은
    taken_at·camera로 파티션·재정렬한다(cluster.partition_order — 멀티 카메라 대응).

    미리보기(preview_key)가 있는 사진만 고른다 — 임베더가 지난 사진이다(#93). 옛 계약에서는 `status='EMBEDDED'` 를 봤지만
    wes V15(2026-09-08)가 그 값을 없앴다(status 는 "S3 에 있나"만 답한다) — 조건을 그대로 두면 대상이 0장이 된다.
    EMBEDDED ⇔ preview_key 있음이었으므로 결과는 같다. score 의 `gallery.load_db` 도 #75 에서 같은 규칙으로 바뀌었다.
    점수·벡터가 아직 없는 사진이 섞여도 `pipeline.run` 이 걸러 낸다(경고 로그 + 제외).

    download=False 면 미리보기를 내려받지 않는다(path 는 None, storage 도 None 가능) — categorize 는 벡터와
    taken_at·camera만 쓰고, 대표 사진은 naming 이 store.preview_paths 로 한꺼번에 내려받는다(#26).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, preview_key, taken_at, camera_make, camera_model "
            "FROM photos WHERE gallery_id = %s "
            "AND deleted_at IS NULL AND preview_key IS NOT NULL ORDER BY display_order, id",
            (gallery_id,),
        )
        rows = cur.fetchall()
    if limit:
        rows = rows[:limit]
    dest_dir = work_dir / str(gallery_id)
    refs: list[PhotoRef] = []
    for photo_id, key, taken_at, make, model in rows:
        path = str(storage.download(key, dest_dir / f"{photo_id}.jpg")) if download else None
        camera = " ".join(s.strip() for s in (make, model) if s and s.strip()) or None
        refs.append(PhotoRef(photo_id=str(photo_id), path=path, taken_at=taken_at, camera=camera))
    return refs
