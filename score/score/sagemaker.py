"""SageMaker training job 진입점 — GPU 벤치마크(#68). 운영 경로가 아니다.

SageMaker 는 컨테이너를 `train` 인자로 띄운다. 여기서는 Lambda 의 handler 와 같은 `job.run()` 을 환경변수로 부르고,
돌아가는 동안 GPU 사용률을 1초마다 표본해 요약을 남긴다 — 60% 아래면 GPU 가 아니라 CPU 디코드·S3 가 병목이라는 뜻.

    GALLERY_ID   필수. FORCE=1 이면 점수 있는 사진도 다시. LIMIT 은 앞에서 N장만.
    DB_* · S3_BUCKET · CLIP_BATCH · ARNIQA_BATCH · SCORE_FP16 · SCORE_DECODE_WORKERS 는 config.Settings 그대로.

결과 JSON 은 stdout(CloudWatch) 과 /opt/ml/output/data/result.json(OutputDataConfig 의 S3) 두 곳에 남는다.
실패는 /opt/ml/output/failure 에 적어야 SageMaker 가 FailureReason 으로 보여준다.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("score.sagemaker")

OUTPUT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
FAILURE_FILE = Path("/opt/ml/output/failure")


class GpuSampler:
    """pynvml 로 1초마다 (util %, 메모리 MB) 표본. GPU 가 없으면 조용히 아무것도 안 한다."""

    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._handle = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.name = pynvml.nvmlDeviceGetName(self._handle)
        except Exception as exc:  # noqa: BLE001
            log.info("GPU 표본 없음: %s", exc)
            self.name = None

    def start(self) -> "GpuSampler":
        if self._handle is not None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
        return self

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                u = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
                m = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20
                self.samples.append((float(u), float(m)))
            except Exception:  # noqa: BLE001
                pass
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if not self.samples:
            return {"gpu": self.name, "samples": 0}
        util = sorted(u for u, _ in self.samples)
        mem = max(m for _, m in self.samples)
        n = len(util)
        return {
            "gpu": self.name, "samples": n,
            "utilAvg": round(sum(util) / n, 1), "utilP50": round(util[n // 2], 1), "utilMax": round(util[-1], 1),
            "utilAbove80Pct": round(100 * sum(1 for u in util if u >= 80) / n, 1),
            "memMaxMB": round(mem),
        }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s", stream=sys.stdout)
    started_wall = time.time()
    gallery_id = int(os.environ["GALLERY_ID"])
    force = os.environ.get("FORCE", "0") not in ("0", "", "false", "no")
    limit = int(os.environ["LIMIT"]) if os.environ.get("LIMIT") else None
    log.info("SageMaker 벤치마크: 갤러리 %d force=%s limit=%s 컨테이너 시작 %s", gallery_id, force, limit,
             time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started_wall)))

    from score import job
    from score.config import Settings

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
