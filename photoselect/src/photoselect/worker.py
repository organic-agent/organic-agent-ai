"""잡 워커 — `ai_analysis_jobs`·`ai_selection_jobs`의 PENDING을 집어 A·B를 돌린다.

wes는 버튼이 눌리면 PENDING 행만 만든다. 이 모듈이 그 폴링이다. 로컬에서는 `python -m photoselect worker`를
띄워 두면 웹의 "AI 분석"·"AI 추천" 버튼이 끝까지 간다. 운영 EC2 워커도 같은 코드다.

    pipeline_of        settings.pipeline 으로 v1/v2 파사드를 고른다. 버전 코드는 여기서만 갈린다
    run_analysis_job   잡 하나(A): claim 은 호출자가 했다는 전제. run → finish/fail
    run_selection_job  잡 하나(B)
    loop               claim_next 로 두 테이블을 번갈아 비운다

`__main__ --db --job-id` 경로도 같은 두 함수를 쓴다 — CLI 와 워커가 잡을 다르게 닫는 일이 없게.
"""

from __future__ import annotations

import logging
import time

import psycopg

from photoselect import jobs
from photoselect.config import Settings as BaseSettings

log = logging.getLogger(__name__)


def pipeline_of(settings):
    """v1/v2/v3 파사드 모듈. settings 는 루트 Settings 든 버전 Settings 든 `pipeline` 만 본다."""
    if settings.pipeline == "v1":
        from photoselect import v1
        return v1
    if settings.pipeline == "v3":
        from photoselect import v3
        return v3
    from photoselect import v2
    return v2


def version_settings(base: BaseSettings):
    return pipeline_of(base).settings_from(base)


def run_analysis_job(st, job_id: int | None, gallery_id: int, settings, *,
                     force: bool = False, use_vlm: bool = True, limit: int | None = None) -> dict:
    """A 한 갤러리. st 는 settings.pipeline 과 같은 버전의 DbStore. job_id 가 None 이면 잡 없이 돈다."""
    from photoselect.storage import PreviewStorage
    pipe = pipeline_of(settings)

    conn = st.conn
    try:
        if not settings.s3_bucket:   # try 안이어야 잡이 RUNNING 으로 남지 않는다
            raise RuntimeError("S3_BUCKET 이 없다 — 미리보기를 내려받을 버킷")
        refs = pipe.gallery.load_db(conn, PreviewStorage(settings.s3_bucket), gallery_id, settings.work_dir, limit=limit)
        if not refs:
            raise RuntimeError(f"갤러리 {gallery_id} 에 EMBEDDED 사진이 없다 — 임베더(embeddings/run)가 먼저다")
        result = pipe.analyze_module().run(st, str(gallery_id), refs, settings, force=force, use_vlm=use_vlm)
    except BaseException as exc:  # noqa: BLE001 — SystemExit 포함, 잡에 실패를 남긴다
        if job_id is not None:
            jobs.fail(conn, jobs.ANALYSIS, job_id, f"{type(exc).__name__}: {exc}")
        raise
    if job_id is not None:
        jobs.finish(conn, jobs.ANALYSIS, job_id, result)
    return result


def run_analysis_job_v3(st, job_id: int | None, gallery_id: int, mode: str, settings, *,
                        llm=None, force: bool = False, limit: int | None = None) -> dict:
    """v3 분석 잡 — FULL 은 사진별 분석 뒤 naming 까지 이어 돌고, NAMING 은 배정만 다시 만든다.

    FULL 잡(job_id 있음)은 naming 산출물(ai_concept_assignments)이 있어야 폴더 생성이 가능하므로
    llm 없이는 시작하지 않는다 — 30분 분석 뒤에 실패하는 것보다 여기서 죽는 게 낫다.
    job_id 없는 CLI 실행은 llm 없이 분석만 돌 수 있다(naming 은 skipped 로 표시).
    """
    from photoselect.storage import PreviewStorage
    pipe = pipeline_of(settings)

    conn = st.conn
    try:
        mode = (mode or "FULL").upper()
        if mode not in ("FULL", "NAMING"):
            raise RuntimeError(f"모르는 분석 잡 mode: {mode} (FULL|NAMING)")
        if llm is None and (mode == "NAMING" or job_id is not None):
            raise RuntimeError("v3 잡은 naming(Bedrock)까지가 산출물이다 — 워커를 --llm 으로 띄워라")
        if mode == "NAMING":
            result = pipe.naming_module().run(st, str(gallery_id), settings, llm, job_id=job_id)
        else:
            if not settings.s3_bucket:
                raise RuntimeError("S3_BUCKET 이 없다 — 미리보기를 내려받을 버킷")
            refs = pipe.gallery.load_db(conn, PreviewStorage(settings.s3_bucket), gallery_id,
                                        settings.work_dir, limit=limit)
            if not refs:
                raise RuntimeError(f"갤러리 {gallery_id} 에 EMBEDDED 사진이 없다 — 임베더(embeddings/run)가 먼저다")
            result = pipe.analyze_module().run(st, str(gallery_id), refs, settings, force=force)
            result["naming"] = (pipe.naming_module().run(st, str(gallery_id), settings, llm, job_id=job_id)
                                if llm is not None else "skipped (no --llm)")
    except BaseException as exc:  # noqa: BLE001
        if job_id is not None:
            jobs.fail(conn, jobs.ANALYSIS, job_id, f"{type(exc).__name__}: {exc}")
        raise
    if job_id is not None:
        jobs.finish(conn, jobs.ANALYSIS, job_id, result)
    return result


def run_selection_job(st, job_id: int | None, selection_id: int | str, settings, *,
                      llm=None, gallery: str | None = None, round_no: int | None = None,
                      top_k: int | None = None, target: int | None = None) -> dict:
    """B 한 라운드. `mode='draft'` 는 round_no=1 로, `refine` 은 None(마지막+1)으로 넘긴다."""
    pipe = pipeline_of(settings)

    conn = st.conn
    st.selection_id = int(selection_id)
    gallery = gallery or st.gallery_of_selection(selection_id)
    try:
        target = target or st.target_count(gallery)
        result = pipe.draft_module().run(st, gallery, settings, selection_id=str(selection_id),
                                         round_no=round_no, top_k=top_k, target=target, llm=llm)
    except BaseException as exc:  # noqa: BLE001
        if job_id is not None:
            jobs.fail(conn, jobs.SELECTION, job_id, f"{type(exc).__name__}: {exc}")
        raise
    if job_id is not None:
        jobs.finish(conn, jobs.SELECTION, job_id, result, round_no=result.get("round"))
    return result


def _one_analysis(settings, use_vlm: bool, llm=None) -> bool:
    st = pipeline_of(settings).store.DbStore(settings)
    try:
        job_id = jobs.claim_next(st.conn, jobs.ANALYSIS)
        if job_id is None:
            return False
        gallery_id, mode = jobs.analysis_job_info(st.conn, job_id)
        log.info("[analysis] job=%s gallery=%s mode=%s 시작", job_id, gallery_id, mode)
        try:
            if settings.pipeline == "v3":
                result = run_analysis_job_v3(st, job_id, gallery_id, mode, settings, llm=llm)
            elif mode.upper() == "NAMING":
                jobs.fail(st.conn, jobs.ANALYSIS, job_id,
                          f"NAMING 잡은 v3 파이프라인 전용이다 — 워커가 pipeline={settings.pipeline} 로 떠 있다")
                raise RuntimeError("NAMING 잡을 v3 아닌 워커가 집었다")
            else:
                result = run_analysis_job(st, job_id, gallery_id, settings, use_vlm=use_vlm)
            log.info("[analysis] job=%s 완료 %s", job_id, result)
        except Exception:  # noqa: BLE001 — 잡에는 이미 FAILED 가 적혔다. 워커는 계속 산다
            log.exception("[analysis] job=%s 실패", job_id)
        return True
    finally:
        st.conn.close()


_warned_no_selection_table = False


def _one_selection(settings, llm) -> bool:
    global _warned_no_selection_table
    st = pipeline_of(settings).store.DbStore(settings)
    try:
        try:
            job_id = jobs.claim_next(st.conn, jobs.SELECTION)
        except psycopg.errors.UndefinedTable:
            # V45 세대 스키마에는 추천 잡 테이블이 아직 없다(추천 재설계 대상) — 분석 잡만 처리한다
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


def loop(settings, *, use_vlm: bool = True, llm=None, poll_seconds: float = 2.0, once: bool = False) -> None:
    """두 테이블을 번갈아 비운다. `once` 면 지금 쌓인 것만 처리하고 돌아온다.

    잡마다 새 접속을 연다 — A 는 수십 분이라 접속 하나를 오래 붙들면 끊긴 뒤 finish 가 실패한다.
    """
    if isinstance(settings, BaseSettings) and type(settings) is BaseSettings:
        settings = version_settings(settings)
    log.info("워커 시작 (pipeline=%s, poll=%ss, vlm=%s, llm=%s)", settings.pipeline, poll_seconds, use_vlm, llm is not None)
    while True:
        try:
            did = _one_analysis(settings, use_vlm, llm) or _one_selection(settings, llm)
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
