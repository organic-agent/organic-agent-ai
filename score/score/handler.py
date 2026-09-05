"""Lambda 진입점.

페이로드는 `{"galleryId": 12, "jobId": 34, "force": false}`. `jobId` 는 wes `ai_analysis_jobs.id`(FULL 잡) —
없으면 잡 계약 밖의 점수 적재(재개·증분 확인용)다.

15분 타임아웃 앞에서 스스로 멈춘다(job.run 의 `remaining_seconds`). 멈췄고 이번 실행에서 한 장이라도 처리했으면
같은 갤러리·잡으로 자기 자신을 EVENT 재호출한다 — `force` 는 넘기지 않는다(재개는 MODEL_VERSION + CLIP 유무).
처리 0장이면 재호출하지 않는다: 같은 사진이 계속 실패하는 갤러리에서 무한히 도는 것을 막는다.

다 끝났고 잡이 있으면 categorize 를 깨운다(chain). 체인 호출이 실패하면 잡을 FAILED 로 닫는다 — RUNNING 에
매달린 잡은 wes 화면에서 영원히 "분석 중"이다.

재호출·체인 모두 Lambda API 다. NAT 없는 서브넷이라 인터페이스 VPC 엔드포인트(`com.amazonaws.<region>.lambda`)와
실행 역할의 `lambda:InvokeFunction`(자기 함수 + categorize 함수)이 있어야 한다.
"""

from __future__ import annotations

import json
import logging

from score import chain, db, job, jobs
from score.config import Settings

log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    gallery_id = event.get("galleryId")
    if gallery_id is None:
        raise ValueError("페이로드에 galleryId가 없습니다")
    gallery_id = int(gallery_id)
    job_id = int(event["jobId"]) if event.get("jobId") is not None else None

    result = job.run(
        gallery_id=gallery_id,
        force=bool(event.get("force", False)),
        settings=_SETTINGS,
        job_id=job_id,
        remaining_seconds=_remaining_seconds(context),
    )

    # 잠금 건너뜀만 여기서 끝낸다. "이미 점수가 있어 건너뛴 사진 수"(int)로 판단하면 전부 건너뛴 재실행에서
    # 체인이 열리지 않아 잡이 RUNNING 에 영원히 남는다(job.was_skipped 참고).
    if job.was_skipped(result):
        return result
    if result.get("stopped"):
        if result.get("processed", 0) > 0:
            result["reinvoked"] = reinvoke(context, gallery_id, job_id)
        return result
    if job_id is not None:
        result["chained"] = chain.invoke_categorize(_SETTINGS, gallery_id, job_id)
        if not result["chained"]:
            _fail_job(job_id, "categorize 호출 실패 — Lambda 엔드포인트·InvokeFunction 권한 확인")
    return result


def _fail_job(job_id: int, error: str) -> None:
    try:
        connection = db.connect(_SETTINGS)
        try:
            jobs.fail(connection, job_id, error)
        finally:
            connection.close()
    except Exception:
        log.exception("잡 %s 을 FAILED 로 닫지 못했다", job_id)


def _remaining_seconds(context):
    """Lambda context 의 남은 시간을 초 단위 함수로. 로컬 테스트처럼 context 가 없으면 None."""
    getter = getattr(context, "get_remaining_time_in_millis", None)
    if getter is None:
        return None
    return lambda: getter() / 1000.0


def reinvoke(context, gallery_id: int, job_id: int | None) -> bool:
    """같은 갤러리·잡으로 자기 자신을 EVENT 호출한다. 실패해도 예외를 올리지 않고 False."""
    function_name = getattr(context, "function_name", None)
    if not function_name:
        log.warning("갤러리 %s: 재호출할 함수 이름이 없다 (context.function_name)", gallery_id)
        return False
    payload = {"galleryId": gallery_id, "force": False}
    if job_id is not None:
        payload["jobId"] = job_id
    try:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "lambda",
            config=Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}),
        )
        client.invoke(FunctionName=function_name, InvocationType="Event",
                      Payload=json.dumps(payload).encode("utf-8"))
        log.info("갤러리 %s: 남은 사진을 위해 자기 재호출", gallery_id)
        return True
    except Exception:
        log.exception("갤러리 %s: 자기 재호출 실패 — 앱이 다시 부르면 이어서 한다", gallery_id)
        return False
