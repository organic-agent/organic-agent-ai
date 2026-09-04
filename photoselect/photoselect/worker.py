"""잡 워커 — `ai_analysis_jobs`의 PENDING을 집어 폴더화 배치를 돌린다.

wes는 "AI 분석" 버튼이 눌리면 PENDING 행만 만든다. 이 모듈이 그 폴링이다. 로컬에서는
`python -m photoselect worker --llm`을 띄워 두면 웹의 "AI 분석" 버튼이 끝까지 간다.
운영 워커도 같은 코드다.

    run_analysis_job    잡 하나. wes mode → 두 잡 매핑(#26):
                          FULL   = score(사진별 점수, torch) → categorize(그룹 + naming)
                          NAMING = categorize (그룹 + naming — 점수는 저장된 것을 쓴다)
    loop                claim_next 로 PENDING을 비운다

`__main__ --db --job-id` 경로도 같은 함수를 쓴다 — CLI 와 워커가 잡을 다르게 닫는 일이 없게.
추천(`ai_selection_jobs`)과 비교샷은 wes가 자기 안에서 처리한다(#25) — 여기서 폴링하지 않는다.
"""

from __future__ import annotations

import logging
import time

from photoselect import jobs
from photoselect.config import Settings

log = logging.getLogger(__name__)

#: wes `ai_analysis_jobs.mode` → 이 패키지의 잡 순서.
MODE_STEPS = {"FULL": ("score", "categorize"), "NAMING": ("categorize",)}


def run_analysis_job(st, job_id: int | None, gallery_id: int, mode: str, settings: Settings, *,
                     llm=None, force: bool = False, limit: int | None = None) -> dict:
    """폴더화 잡 하나. 결과는 {"mode", "score"?, "categorize"} 로 단계별 결과를 담는다.

    잡(job_id 있음)은 naming 산출물(ai_concept_assignments)이 있어야 폴더 생성이 가능하므로
    llm 없이는 시작하지 않는다 — 30분 점수 계산 뒤에 실패하는 것보다 여기서 죽는 게 낫다.
    job_id 없는 CLI 실행은 llm 없이 돌 수 있다(naming 은 skipped 로 표시).
    """
    from photoselect import gallery
    from photoselect import categorize, score
    from photoselect.storage import PreviewStorage

    conn = st.conn
    try:
        mode = (mode or "FULL").upper()
        steps = MODE_STEPS.get(mode)
        if steps is None:
            raise RuntimeError(f"모르는 분석 잡 mode: {mode} ({'|'.join(MODE_STEPS)})")
        if llm is None and job_id is not None:
            raise RuntimeError("폴더화 잡은 naming(Bedrock)까지가 산출물이다 — 워커를 --llm 으로 띄워라")
        if not settings.s3_bucket:
            raise RuntimeError("S3_BUCKET 이 없다 — 미리보기를 내려받을 버킷")
        storage = PreviewStorage(settings.s3_bucket)
        result: dict = {"gallery": str(gallery_id), "pipeline": "v3", "mode": mode}
        if "score" in steps:
            refs = gallery.load_db(conn, storage, gallery_id, settings.work_dir, limit=limit, download=True)
            if not refs:
                raise RuntimeError(f"갤러리 {gallery_id} 에 EMBEDDED 사진이 없다 — 임베더(embeddings/run)가 먼저다")
            result["score"] = score.run(st, str(gallery_id), refs, settings, force=force)
        refs = gallery.load_db(conn, storage, gallery_id, settings.work_dir, limit=limit, download=False)
        result["categorize"] = categorize.run(st, str(gallery_id), refs, settings, llm, job_id=job_id)
    except BaseException as exc:  # noqa: BLE001 — SystemExit 포함, 잡에 실패를 남긴다
        if job_id is not None:
            jobs.fail(conn, jobs.ANALYSIS, job_id, f"{type(exc).__name__}: {exc}")
        raise
    if job_id is not None:
        jobs.finish(conn, jobs.ANALYSIS, job_id, result)
    return result


def _one_analysis(settings: Settings, llm=None) -> bool:
    from photoselect import store

    st = store.DbStore(settings)
    try:
        job_id = jobs.claim_next(st.conn, jobs.ANALYSIS)
        if job_id is None:
            return False
        gallery_id, mode = jobs.analysis_job_info(st.conn, job_id)
        log.info("[analysis] job=%s gallery=%s mode=%s 시작", job_id, gallery_id, mode)
        try:
            result = run_analysis_job(st, job_id, gallery_id, mode, settings, llm=llm)
            log.info("[analysis] job=%s 완료 %s", job_id, result)
        except Exception:  # noqa: BLE001 — 잡에는 이미 FAILED 가 적혔다. 워커는 계속 산다
            log.exception("[analysis] job=%s 실패", job_id)
        return True
    finally:
        st.conn.close()


def loop(settings: Settings, *, llm=None, poll_seconds: float = 2.0, once: bool = False) -> None:
    """PENDING 분석 잡을 비운다. `once` 면 지금 쌓인 것만 처리하고 돌아온다.

    잡마다 새 접속을 연다 — 분석은 수십 분이라 접속 하나를 오래 붙들면 끊긴 뒤 finish 가 실패한다.
    """
    log.info("워커 시작 (poll=%ss, llm=%s)", poll_seconds, llm is not None)
    while True:
        try:
            did = _one_analysis(settings, llm)
        except Exception as exc:  # noqa: BLE001 — 스키마 미적용(wes Flyway 전)·접속 끊김 등. 죽지 말고 기다린다
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
