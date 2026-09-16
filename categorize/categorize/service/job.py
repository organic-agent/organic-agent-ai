"""갤러리 잡 — Lambda handler 와 로컬 CLI 가 같은 `run()` 을 부른다.

    미리보기 있는 사진 목록(다운로드 없음) → pipeline.run(그룹 → naming) → (실패면 잡에 error)

잡 상태는 건드리지 않는다(#95, wes V16): wes 가 CATEGORIZING 으로 옮기며 부르고, 배정 행·백분위를 보고 DONE 을 찍는다.
이 함수가 잡 테이블에 쓰는 것은 실패했을 때의 `error` 하나다. 잡(job_id)은
naming(Bedrock)까지가 산출물이라 llm 없이는 시작하지 않는다 — 잡 없는 CLI 실행만 그룹화만으로 끝낼 수 있다.
갤러리 전체를 매번 다시 계산하므로(수 초 + Bedrock 몇 번) 데드라인·잠금이 없다.
"""

from __future__ import annotations

import logging
import time

from categorize.config.settings import Settings
from categorize.repository import connection, jobs
from categorize.repository.analysis import DbStore
from categorize.repository.photos import load_db
from categorize.service import pipeline

log = logging.getLogger(__name__)


def run(gallery_id: int, settings: Settings | None = None, job_id: int | None = None,
        llm=None, limit: int | None = None) -> dict:
    started = time.monotonic()
    settings = settings or Settings.from_env()
    if llm is None and job_id is not None:
        raise RuntimeError("잡은 naming(Bedrock)까지가 산출물이다 — llm 없이 시작하지 않는다")

    conn = connection.connect(settings)
    try:
        try:
            refs = load_db(conn, None, gallery_id, settings.work_dir, limit=limit, download=False)
            if not refs:
                raise RuntimeError(f"갤러리 {gallery_id} 에 미리보기 있는 사진이 없다 — embedder 가 먼저다")
            store = DbStore(settings, conn)
            result = pipeline.run(store, str(gallery_id), refs, settings, llm, job_id=job_id)
        except BaseException as exc:  # noqa: BLE001 — SystemExit 포함, 잡에 실패를 남긴다
            if job_id is not None:
                jobs.fail(conn, job_id, f"{type(exc).__name__}: {exc}")
            raise
        result["elapsedSeconds"] = round(time.monotonic() - started, 1)
        log.info("완료: %s", result)
        return result
    finally:
        conn.close()
