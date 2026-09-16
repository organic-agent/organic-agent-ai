"""`photos` 조회 — "이 갤러리(또는 이 id 들)에 어떤 사진이 있고 미리보기는 어디 있나".

미리보기(`photos.preview_key`)가 있는 사진만 고른다 — 임베더가 지난 사진이다. 원본이 아니라 미리보기(EXIF 회전·리사이즈 JPEG)를
읽는다: HEIC 디코드를 피하고 용량이 1/10이며, 점수·태그는 긴 변 1024면 충분하다. 내려받기는 `storage.download_previews`.
"""

from __future__ import annotations

from pathlib import Path

from score.domain.photo import PhotoRef


def _camera(make, model) -> str | None:
    return " ".join(s.strip() for s in (make, model) if s and s.strip()) or None


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
        refs.append(PhotoRef(photo_id=str(photo_id), path=path, taken_at=taken_at, camera=_camera(make, model),
                             preview_key=key))
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
        refs.append(PhotoRef(photo_id=str(pid), path=None, taken_at=taken_at, camera=_camera(make, model), preview_key=key))
    return refs
