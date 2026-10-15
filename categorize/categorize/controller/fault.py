"""장애 주입 — 부하 측정 전용(wes #286 R-3). `FAULT_EXIT_RATE`(0~1, 기본 0)의 확률로 호출을 받자마자 프로세스를 끝낸다.

파이썬 예외가 아니라 프로세스 종료라 R-2-3 의 `Runtime.ExitError` 와 같은 모양으로 실패한다 — Lambda 재시도와 wes 재배정이
실제 장애처럼 돈다. 끝내기 전에 `fault.injected` 한 줄을 남겨, 측정 스크립트가 "실패 시각 → 다시 처리 시작"을 짝짓는다.
운영에는 이 변수를 넣지 않는다(infra dev 변수만).
"""

from __future__ import annotations

import logging
import os
import random
import sys
from typing import Callable

log = logging.getLogger(__name__)

ENV_NAME = "FAULT_EXIT_RATE"


def maybe_exit(
    detail: str,
    rate: float | None = None,
    draw: Callable[[], float] = random.random,
    exit_process: Callable[[int], None] = os._exit,
) -> None:
    rate = float(os.environ.get(ENV_NAME, "0")) if rate is None else rate
    if rate <= 0 or draw() >= rate:
        return
    log.warning("fault.injected %s", detail)
    # os._exit 는 버퍼를 비우지 않는다 — 비우지 않으면 위 줄이 CloudWatch 에 닿지 않는다.
    for handler in logging.getLogger().handlers:
        handler.flush()
    sys.stdout.flush()
    sys.stderr.flush()
    exit_process(1)
