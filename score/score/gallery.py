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
    path: str | None  # 로컬 파일 경로 (DB 모드에서는 임시 다운로드 경로, download=False 면 None)
    #: EXIF 촬영 시각(datetime)·카메라 바디("make model"). 연사 클러스터의 순서·파티션 키 —
    #: 임베더가 채운 photos.taken_at/camera_make/camera_model. 로컬 모드는 None(파일명 순 폴백).
    taken_at: object | None = None
    camera: str | None = None
    #: DB 모드의 미리보기 S3 키 — download=False 로 목록만 읽은 뒤 `download_previews` 로 자기 몫만 내려받는다(#54).
    preview_key: str | None = None


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

    미리보기(preview_key)가 있는 사진만 고른다 — 임베더가 지난 사진이다(#75: v2 에서 `status` 는 "S3 에 있나"만 답하고
    EMBEDDED 값이 사라지므로 status 를 보지 않는다. 옛 계약에서도 EMBEDDED ⇔ preview_key 있음이라 결과는 같다).

    download=False 면 미리보기를 내려받지 않는다(path 는 None, storage 도 None 가능) — categorize 는 벡터와
    taken_at·camera만 쓰고, 대표 사진은 store.preview_path 가 필요할 때 지연 다운로드한다(#26).
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
        refs.append(PhotoRef(photo_id=str(photo_id), path=path, taken_at=taken_at, camera=camera, preview_key=key))
    return refs


def load_by_ids(conn, photo_ids: list[int]) -> list[PhotoRef]:
    """사진 id 목록 → PhotoRef(path 없음, preview_key 있음)(#75). v2 Lambda 폴백 페이로드 `photoIds` 용.
    휴지통·미리보기 없는 행은 빠진다. 요청 순서 유지."""
    if not photo_ids:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.id, p.preview_key, p.taken_at, p.camera_make, p.camera_model FROM photos p "
            "JOIN galleries g ON g.id = p.gallery_id WHERE p.id = ANY(%s) AND p.deleted_at IS NULL "
            "AND g.deleted_at IS NULL AND p.preview_key IS NOT NULL",
            (list(photo_ids),),
        )
        rows = {row[0]: row for row in cur.fetchall()}
    refs = []
    for pid in photo_ids:
        if pid not in rows:
            continue
        _, key, taken_at, make, model = rows[pid]
        camera = " ".join(s.strip() for s in (make, model) if s and s.strip()) or None
        refs.append(PhotoRef(photo_id=str(pid), path=None, taken_at=taken_at, camera=camera, preview_key=key))
    return refs


def download_previews(storage, refs: list[PhotoRef], dest_dir: Path, workers: int = 8) -> list[PhotoRef]:
    """path 가 없는 ref 의 미리보기를 내려받아 path 를 채운 새 목록. 샤드가 자기 몫만 받을 때 쓴다(#54).

    `workers` 스레드로 동시에 받는다(#68): 장당 0.2MB 라 시간은 전송량이 아니라 S3 왕복(~80ms)이 정한다 — 한 프로세스가
    7,000장을 한 장씩 받으면 9분이지만 8개 동시면 1분 남짓. boto3 클라이언트는 스레드에서 같이 써도 된다."""
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    def fetch(ref: PhotoRef) -> PhotoRef:
        if ref.path is None and ref.preview_key:
            return replace(ref, path=str(storage.download(ref.preview_key, dest_dir / f"{ref.photo_id}.jpg")))
        return ref

    if workers <= 1:
        return [fetch(ref) for ref in refs]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fetch, refs))
