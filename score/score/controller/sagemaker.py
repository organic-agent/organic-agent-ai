"""SageMaker training job 진입점 — GPU 벤치마크(#68). 운영 경로가 아니다.

SageMaker 는 컨테이너를 `train` 인자로 띄운다. 여기서는 Lambda 의 handler 와 같은 `job.run()` 을 환경변수로 부르고,
돌아가는 동안 GPU 사용률을 1초마다 표본해 요약을 남긴다 — 60% 아래면 GPU 가 아니라 CPU 디코드·S3 가 병목이라는 뜻.

    GALLERY_ID   필수. FORCE=1 이면 점수 있는 사진도 다시. LIMIT 은 앞에서 N장만.
    DB_* · S3_BUCKET · CLIP_BATCH · ARNIQA_BATCH · SCORE_FP16 · SCORE_DECODE_WORKERS 는 config/settings.py 의 Settings 그대로.

결과 JSON 은 stdout(CloudWatch) 과 /opt/ml/output/data/result.json(OutputDataConfig 의 S3) 두 곳에 남는다.
실패는 /opt/ml/output/failure 에 적어야 SageMaker 가 FailureReason 으로 보여준다.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

from score.infrastructure.gpu import GpuSampler

log = logging.getLogger("score.sagemaker")

OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
FAILURE_FILE = Path("/opt/ml/output/failure")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s", stream=sys.stdout)
    started_wall = time.time()
    gallery_id = int(os.environ["GALLERY_ID"])
    force = os.environ.get("FORCE", "0") not in ("0", "", "false", "no")
    limit = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
    log.info("SageMaker 벤치마크: 갤러리 %d force=%s limit=%s 컨테이너 시작 %s", gallery_id, force, limit,
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)))

    from score.config.settings import Settings
    from score.service import job

    settings = Settings.from_env()
    sampler = GpuSampler().start()
    t0 = time.monotonic()
    try:
        result = job.run(gallery_id=gallery_id, force=force, settings=settings, limit=limit)
    except BaseException as exc:  # noqa: BLE001
        gpu = sampler.stop()
        FAILURE_FILE.parent.mkdir(parents=True, exist_ok=True)
        FAILURE_FILE.write_text(f"{type(exc).__name__}: {exc}\n")
        log.exception("벤치마크 실패 (gpu=%s)", gpu)
        return 1
    gpu = sampler.stop()
    summary = {
        "galleryId": gallery_id, "force": force, "limit": limit,
        "containerStartedAt": started_wall, "wallSeconds": round(time.monotonic() - t0, 1),
        "instance": os.environ.get("SM_CURRENT_INSTANCE_TYPE"),
        "knobs": {"clipBatch": settings.knobs.clip_batch, "arniqaBatch": settings.knobs.arniqa_batch,
                  "fp16": settings.knobs.fp16, "decodeWorkers": settings.knobs.decode_workers,
                  "downloadWorkers": settings.download_workers, "device": settings.knobs.device},
        "gpu": gpu, "result": result,
    }
    line = json.dumps(summary, ensure_ascii=False)
    log.info("BENCHMARK_RESULT %s", line)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "result.json").write_text(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
