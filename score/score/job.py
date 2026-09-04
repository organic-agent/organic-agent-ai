"""갤러리 잡 — Lambda handler 와 로컬 CLI 가 같은 `run()` 을 부른다.

    잠금 → (잡이면 RUNNING) → EMBEDDED 사진 목록 + 미리보기 다운로드 → pipeline.run → (잡이면 result 기록)

`ai_analysis_jobs` 는 이 모듈이 **열기만** 한다. DONE 은 체인의 끝인 categorize 가 찍는다. 그래서 잡이 있을 때
categorize 실행기(chain)가 설정돼 있지 않으면 시작 전에 실패시킨다 — 30분 점수를 낸 뒤 매달리는 것보다 낫다.
결과 dict 의 `stopped`·`remaining`·`processed` 를 handler 가 보고 자기 재호출 / 체인 호출을 결정한다.
"""

from __future__ import annotations

import logging
import time
from typing import Callable

from score import db, jobs, pipeline
from score.config import Settings
from score.gallery import load_db
from score.storage import PreviewStorage
from score.store import DbStore

log = logging.getLogger(__name__)

ALREADY_RUNNING = "already running"

#: 갤러리 advisory lock 의 앞쪽 키. embedder(0x454D42 'EMB')·다른 프로세스와 키 공간이 겹치지 않게 이 모듈만의 상수.
GALLERY_LOCK_NAMESPACE = 0x53434F  # 'SCO'


def try_lock_gallery(connection, gallery_id: int) -> bool:
    """같은 갤러리의 SCORE 가 이미 돌고 있으면 False. 세션 잠금이라 연결이 닫히면 풀린다."""
    with connection.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s, %s)", (GALLERY_LOCK_NAMESPACE, gallery_id))
        row = cur.fetchone()
    return bool(row and row[0])


def run(gallery_id: int, force: bool = False, settings: Settings | None = None,
        job_id: int | None = None, limit: int | None = None,
        remaining_seconds: Callable[[], float] | None = None) -> dict:
    started = time.monotonic()
    settings = settings or Settings.from_env()
    if not settings.s3_bucket:
        raise RuntimeError("S3_BUCKET 이 없다 — 미리보기를 내려받을 버킷")

    connection = db.connect(settings)
    try:
        if not try_lock_gallery(connection, gallery_id):
            log.info("갤러리 %s: 다른 실행이 진행 중 — 건너뜀", gallery_id)
            return {"gallery": str(gallery_id), "pipeline": "v3", "mode": "score", "skipped": ALREADY_RUNNING,
                    "processed": 0, "stopped": False, "remaining": 0,
                    "elapsedSeconds": round(time.monotonic() - started, 1)}

        if job_id is not None:
            status = jobs.start(connection, job_id)
            if status not in ("RUNNING",):
                raise RuntimeError(f"잡 {job_id} 은 실행할 수 없는 상태다: {status}")
            if not settings.chain_configured:
                jobs.fail(connection, job_id, "categorize 실행기 미설정 (CATEGORIZE_FUNCTION_NAME | CATEGORIZE_COMMAND)")
                raise RuntimeError("잡은 categorize 까지가 산출물이다 — categorize 실행기를 설정하라")

        try:
            refs = load_db(connection, PreviewStorage(settings.s3_bucket), gallery_id, settings.work_dir, limit=limit)
            if not refs:
                raise RuntimeError(f"갤러리 {gallery_id} 에 EMBEDDED 사진이 없다 — embedder 가 먼저다")
            store = DbStore(settings, connection)
            result = pipeline.run(store, str(gallery_id), refs, settings, force=force,
                                  remaining_seconds=remaining_seconds)
        except BaseException as exc:  # noqa: BLE001 — SystemExit 포함, 잡에 실패를 남긴다
            if job_id is not None:
                jobs.fail(connection, job_id, f"{type(exc).__name__}: {exc}")
            raise

        if job_id is not None:
            jobs.record(connection, job_id, {"score": result})
        log.info("완료: %s", result)
        return result
    finally:
        connection.close()
