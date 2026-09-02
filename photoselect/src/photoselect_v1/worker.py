"""잡 워커 — `ai_analysis_jobs`·`ai_selection_jobs`의 PENDING을 집어 배치를 돌린다.

wes는 버튼이 눌리면 PENDING 행만 만든다. 이 모듈이 그 폴링이다. 로컬에서는
`python -m photoselect_v1 worker --llm`을 띄워 두면 웹의 "AI 분석"·"AI 추천" 버튼이 끝까지 간다.
운영 EC2 워커도 같은 코드다.

    run_analysis_job    잡 하나(폴더화): FULL은 분석 뒤 naming까지, NAMING은 배정만 다시
    run_selection_job   잡 하나(폴더별 추천)
    loop                claim_next 로 두 테이블을 번갈아 비운다

`__main__ --db --job-id` 경로도 같은 두 함수를 쓴다 — CLI 와 워커가 잡을 다르게 닫는 일이 없게.
compare(비교샷)는 워커를 거치지 않는다 — handler 의 동기 경로다.
"""

from __future__ import annotations

import logging
import time

import psycopg

from photoselect_v1 import jobs
from photoselect_v1.config import Settings

log = logging.getLogger(__name__)


def run_analysis_job(st, job_id: int | None, gallery_id: int, mode: str, settings: Settings, *,
                     llm=None, force: bool = False, limit: int | None = None) -> dict:
    """폴더화 잡 — FULL 은 사진별 분석 뒤 naming 까지 이어 돌고, NAMING 은 배정만 다시 만든다.

    FULL 잡(job_id 있음)은 naming 산출물(ai_concept_assignments)이 있어야 폴더 생성이 가능하므로
    llm 없이는 시작하지 않는다 — 30분 분석 뒤에 실패하는 것보다 여기서 죽는 게 낫다.
    job_id 없는 CLI 실행은 llm 없이 분석만 돌 수 있다(naming 은 skipped 로 표시).
    """
    from photoselect_v1 import gallery
    from photoselect_v1.foldering import analyze, naming
    from photoselect_v1.storage import PreviewStorage

    conn = st.conn
    try:
        mode = (mode or "FULL").upper()
        if mode not in ("FULL", "NAMING"):
            raise RuntimeError(f"모르는 분석 잡 mode: {mode} (FULL|NAMING)")
        if llm is None and (mode == "NAMING" or job_id is not None):
            raise RuntimeError("폴더화 잡은 naming(Bedrock)까지가 산출물이다 — 워커를 --llm 으로 띄워라")
        if mode == "NAMING":
            result = naming.run(st, str(gallery_id), settings, llm, job_id=job_id)
        else:
            if not settings.s3_bucket:
                raise RuntimeError("S3_BUCKET 이 없다 — 미리보기를 내려받을 버킷")
            refs = gallery.load_db(conn, PreviewStorage(settings.s3_bucket), gallery_id,
                                   settings.work_dir, limit=limit)
            if not refs:
                raise RuntimeError(f"갤러리 {gallery_id} 에 EMBEDDED 사진이 없다 — 임베더(embeddings/run)가 먼저다")
            result = analyze.run(st, str(gallery_id), refs, settings, force=force)
            result["naming"] = (naming.run(st, str(gallery_id), settings, llm, job_id=job_id)
                                if llm is not None else "skipped (no --llm)")
    except BaseException as exc:  # noqa: BLE001 — SystemExit 포함, 잡에 실패를 남긴다
        if job_id is not None:
            jobs.fail(conn, jobs.ANALYSIS, job_id, f"{type(exc).__name__}: {exc}")
        raise
    if job_id is not None:
        jobs.finish(conn, jobs.ANALYSIS, job_id, result)
    return result


def run_selection_job(st, job_id: int | None, selection_id: int | str, settings: Settings, *,
                      llm=None, gallery: str | None = None, round_no: int | None = None,
                      top_k: int | None = None, target: int | None = None) -> dict:
    """추천 한 라운드. `mode='draft'` 는 round_no=1 로, `refine` 은 None(마지막+1)으로 넘긴다."""
    from photoselect_v1.recommend import draft

    conn = st.conn
    st.selection_id = int(selection_id)
    gallery = gallery or st.gallery_of_selection(selection_id)
    try:
        target = target or st.target_count(gallery)
        result = draft.run(st, gallery, settings, selection_id=str(selection_id),
                           round_no=round_no, top_k=top_k, target=target, llm=llm)
    except BaseException as exc:  # noqa: BLE001
        if job_id is not None:
            jobs.fail(conn, jobs.SELECTION, job_id, f"{type(exc).__name__}: {exc}")
        raise
    if job_id is not None:
        jobs.finish(conn, jobs.SELECTION, job_id, result, round_no=result.get("round"))
    return result


def _one_analysis(settings: Settings, llm=None) -> bool:
    from photoselect_v1 import store

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


_warned_no_selection_table = False


def _one_selection(settings: Settings, llm) -> bool:
    global _warned_no_selection_table
    from photoselect_v1 import store

    st = store.DbStore(settings)
    try:
        try:
            job_id = jobs.claim_next(st.conn, jobs.SELECTION)
        except psycopg.errors.UndefinedTable:
            # V45 세대 스키마에는 추천 잡 테이블이 아직 없다(V46 예정) — 분석 잡만 처리한다
            if not _warned_no_selection_table:
                log.info("ai_selection_jobs 가 없다 — 추천 잡 폴링은 끈다 (V45 세대 스키마, 분석 잡만 처리)")
                _warned_no_selection_table = True
            return False
        if job_id is None:
            return False
        selection_id, mode = jobs.selection_job_info(st.conn, job_id)
        log.info("[selection] job=%s selection=%s mode=%s 시작", job_id, selection_id, mode)
        try:
            result = run_selection_job(st, job_id, selection_id, settings, llm=llm,
                                       round_no=1 if mode == "draft" else None)
            log.info("[selection] job=%s 완료 round=%s k=%s", job_id, result.get("round"), result.get("k"))
        except Exception:  # noqa: BLE001
            log.exception("[selection] job=%s 실패", job_id)
        return True
    finally:
        st.conn.close()


def loop(settings: Settings, *, llm=None, poll_seconds: float = 2.0, once: bool = False) -> None:
    """두 테이블을 번갈아 비운다. `once` 면 지금 쌓인 것만 처리하고 돌아온다.

    잡마다 새 접속을 연다 — 분석은 수십 분이라 접속 하나를 오래 붙들면 끊긴 뒤 finish 가 실패한다.
    """
    log.info("워커 시작 (poll=%ss, llm=%s)", poll_seconds, llm is not None)
    while True:
        try:
            did = _one_analysis(settings, llm) or _one_selection(settings, llm)
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
