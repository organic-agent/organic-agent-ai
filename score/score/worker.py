"""로컬 폴링 워커 — 운영에는 없다.

운영은 wes 가 `ai_analysis_jobs` 행을 만들고 score Lambda 를 EVENT 로 부른다. 로컬 wes 에는 Lambda 가 없어
이 루프가 그 자리를 대신한다(wes scripts/local-worker.sh 가 `python -m score worker` 를 띄운다).
wes 에 분석 invoker(Lambda / 로컬 서브프로세스)가 생기면 이 파일은 지운다.

    FULL   = job.run(점수) → chain.invoke_categorize (서브프로세스, CATEGORIZE_COMMAND)
    NAMING = chain.invoke_categorize 만
"""

from __future__ import annotations

import logging
import sys
import time

from score import chain, db, job, jobs
from score.config import Settings

log = logging.getLogger(__name__)


def _default_command(settings: Settings) -> Settings:
    """CATEGORIZE_COMMAND 가 없으면 같은 인터프리터의 `-m categorize` 로 — 한 venv 에 둘 다 깔린 로컬 전제."""
    if settings.chain_configured:
        return settings
    from dataclasses import replace
    return replace(settings, categorize_command=f"{sys.executable} -m categorize")


def _one(settings: Settings) -> bool:
    connection = db.connect(settings)
    try:
        job_id = jobs.claim_next(connection)
        if job_id is None:
            return False
        found = jobs.info(connection, job_id)
    finally:
        connection.close()
    if found is None:
        return True
    gallery_id, mode = found
    log.info("[worker] job=%s gallery=%s mode=%s 시작", job_id, gallery_id, mode)
    try:
        if mode.upper() == "FULL":
            result = job.run(gallery_id, settings=settings, job_id=job_id)
            if result.get("skipped") or result.get("stopped"):
                log.warning("[worker] job=%s 점수가 끝나지 않았다: %s", job_id, result)
                return True
        chained = chain.invoke_categorize(settings, gallery_id, job_id)
        if not chained:
            connection = db.connect(settings)
            try:
                jobs.fail(connection, job_id, "categorize 서브프로세스 시작 실패")
            finally:
                connection.close()
        log.info("[worker] job=%s 점수 완료, categorize 체인=%s", job_id, chained)
    except Exception:  # noqa: BLE001 — 잡에는 이미 FAILED 가 적혔다. 워커는 계속 산다
        log.exception("[worker] job=%s 실패", job_id)
    return True


def loop(settings: Settings, *, poll_seconds: float = 2.0, once: bool = False) -> None:
    settings = _default_command(settings)
    log.info("워커 시작 (poll=%ss, categorize=%s)", poll_seconds,
             settings.categorize_function_name or settings.categorize_command)
    while True:
        try:
            did = _one(settings)
        except Exception as exc:  # noqa: BLE001 — 스키마 미적용·접속 끊김 등. 죽지 말고 기다린다
            log.warning("잡 테이블에 접근할 수 없다 (%s: %s) — wes 앱이 떠서 Flyway 가 돌았는지 확인. %ss 뒤 재시도",
                        type(exc).__name__, str(exc).splitlines()[0], max(poll_seconds, 5))
            if once:
                raise
            time.sleep(max(poll_seconds, 5))
            continue
        if did:
            continue
        if once:
            return
        time.sleep(poll_seconds)
