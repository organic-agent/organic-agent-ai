"""갤러리 잡 — Lambda handler 와 로컬 CLI 가 같은 `run()` 을 부른다.

    사진 목록(`photo_ids`) 또는 갤러리 전체 → 미리보기 다운로드 → pipeline.run

**운영은 `photo_ids` 하나다**(#98): wes 스위퍼가 미점수 사진을 50장씩 `{galleryId, photoIds}` 로 부른다(GPU 워커가 없거나
못 따라갈 때의 폴백). 잡 테이블·체인·샤딩·잠금은 wes 가 소유하므로 여기 없다 — 잡을 열고 닫는 것도, categorize 를 부르는 것도 wes 다.

갤러리 전체 경로는 **로컬·벤치마크 전용**으로 남는다: wes `scripts/local-ai.sh` 가 사진 목록 없이 CLI 를 부르고,
`controller/sagemaker.py`(GPU 벤치마크)도 갤러리 하나를 통째로 돌린다. 그 경로도 점수만 쓰고 끝난다.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from score.config.settings import Settings
from score.domain.errors import PREVIEW_MISSING, SCORE_FAILED
from score.repository import connection
from score.repository.photos import load_by_ids, load_db
from score.repository.storage import PreviewStorage, download_previews
from score.repository.store import DbStore
from score.service import pipeline

log = logging.getLogger(__name__)


def _run_photo_ids(gallery_id: int, photo_ids: list[int], settings: Settings,
                   remaining_seconds: Callable[[], float] | None, started: float) -> dict:
    conn = connection.connect(settings)
    try:
        storage = PreviewStorage(settings.s3_bucket)
        refs = load_by_ids(conn, photo_ids)
        missing: list[str] = []
        refs = download_previews(storage, refs, settings.work_dir / str(gallery_id), workers=settings.download_workers,
                                 missing=missing)
        store = DbStore(settings, conn)
        result = pipeline.run(store, str(gallery_id), refs, settings, force=True, remaining_seconds=remaining_seconds)
        # 결정적 실패는 워커와 같은 표시(#85) — wes 가 그 장을 기대 장수에서 뺀다.
        failed = [str(pid) for pid in result.get("failed", [])]
        if missing:
            store.write_errors(missing, PREVIEW_MISSING)
        if failed:
            store.write_errors(failed, SCORE_FAILED)
        store.commit()
        result["failed"] = failed + missing
        result["photoIds"] = len(photo_ids)
        result["elapsedSeconds"] = round(time.monotonic() - started, 1)
        log.info("갤러리 %s 사진 %d장: 완료 %s", gallery_id, len(photo_ids), result)
        log.info("score lambda gallery=%s photos=%d failed=%d seconds=%.1f", gallery_id, result.get("processed", 0),
                 len(result["failed"]), result["elapsedSeconds"])
        return result
    finally:
        conn.close()


def run(gallery_id: int, force: bool = False, settings: Settings | None = None,
        limit: int | None = None, remaining_seconds: Callable[[], float] | None = None,
        photo_ids: list[int] | None = None) -> dict:
    """`photo_ids` 가 있으면 그 목록만(운영 폴백). 없으면 갤러리 전체(로컬 CLI·벤치마크).

    둘 다 점수만 쓰고 끝난다 — 잡·체인·샤딩·잠금 없음(#98). 갤러리 경로의 `force` 는 이미 점수가 있는 사진도 다시 계산한다."""
    started = time.monotonic()
    settings = settings or Settings.from_env()
    if not settings.s3_bucket:
        raise RuntimeError("S3_BUCKET 이 없다 — 미리보기를 내려받을 버킷")
    if photo_ids is not None:
        return _run_photo_ids(gallery_id, photo_ids, settings, remaining_seconds, started)

    conn = connection.connect(settings)
    try:
        storage = PreviewStorage(settings.s3_bucket)
        refs = load_db(conn, storage, gallery_id, settings.work_dir, limit=limit, download=False)
        if not refs:
            raise RuntimeError(f"갤러리 {gallery_id} 에 미리보기 있는 사진이 없다 — embedder 가 먼저다")
        t_dl = time.monotonic()
        refs = download_previews(storage, refs, settings.work_dir / str(gallery_id), workers=settings.download_workers)
        log.info("갤러리 %s: 미리보기 %d장 다운로드 %.1fs (workers=%d)", gallery_id, len(refs), time.monotonic() - t_dl,
                 settings.download_workers)
        store = DbStore(settings, conn)
        result = pipeline.run(store, str(gallery_id), refs, settings, force=force, remaining_seconds=remaining_seconds)
        result["elapsedSeconds"] = round(time.monotonic() - started, 1)
        log.info("완료: %s", result)
        log.info("score gallery=%s photos=%d failed=%d seconds=%.1f", gallery_id, result.get("processed", 0),
                 len(result.get("failed", [])), result["elapsedSeconds"])
        return result
    finally:
        conn.close()
