"""체인의 다음 칸 — categorize 를 깨운다.

운영은 Lambda EVENT(`CATEGORIZE_FUNCTION_NAME`), 로컬은 서브프로세스(`CATEGORIZE_COMMAND`)다. wes 의
`LambdaEmbeddingInvoker` / `LocalProcessEmbeddingInvoker` 와 같은 모양 — 기다리지 않고 띄우기만 한다.
결과는 categorize 가 DB(`ai_analysis_jobs`, `photo_analysis`, `ai_concept_assignments`)에 직접 쓴다.

Lambda 호출은 NAT 없는 서브넷이라 인터페이스 VPC 엔드포인트(`com.amazonaws.<region>.lambda`)와 실행 역할의
`lambda:InvokeFunction`(categorize 함수)이 있어야 한다. 없으면 False 로 끝나고 잡은 RUNNING 에 남는다 —
handler 가 그 경우 잡을 FAILED 로 닫는다.
"""

from __future__ import annotations

import json
import logging
import shlex
import subprocess

from score.config import Settings

log = logging.getLogger(__name__)


def invoke_categorize(settings: Settings, gallery_id: int, job_id: int | None) -> bool:
    payload = {"galleryId": gallery_id}
    if job_id is not None:
        payload["jobId"] = job_id

    if settings.categorize_function_name:
        try:
            import boto3
            from botocore.config import Config

            client = boto3.client(
                "lambda",
                config=Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}),
            )
            client.invoke(
                FunctionName=settings.categorize_function_name,
                InvocationType="Event",
                Payload=json.dumps(payload).encode("utf-8"),
            )
            log.info("갤러리 %s: categorize Lambda 호출 (job=%s)", gallery_id, job_id)
            return True
        except Exception:
            log.exception("갤러리 %s: categorize Lambda 호출 실패", gallery_id)
            return False

    if settings.categorize_command:
        argv = shlex.split(settings.categorize_command) + ["--gallery-id", str(gallery_id)]
        if job_id is not None:
            argv += ["--job-id", str(job_id)]
        try:
            proc = subprocess.Popen(argv)   # noqa: S603 — 운영자가 환경변수로 준 명령
            log.info("갤러리 %s: categorize 서브프로세스 시작 pid=%s (job=%s)", gallery_id, proc.pid, job_id)
            return True
        except OSError:
            log.exception("갤러리 %s: categorize 서브프로세스 시작 실패: %s", gallery_id, argv)
            return False

    log.warning("갤러리 %s: categorize 실행기가 없다 (CATEGORIZE_FUNCTION_NAME | CATEGORIZE_COMMAND)", gallery_id)
    return False
