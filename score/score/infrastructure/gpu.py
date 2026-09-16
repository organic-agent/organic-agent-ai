"""GPU 사용률 표본 — pynvml. SageMaker 벤치마크(#68)가 돌아가는 동안 1초마다 (util %, 메모리 MB) 를 모은다."""

from __future__ import annotations

import logging
import threading

log = logging.getLogger(__name__)


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
