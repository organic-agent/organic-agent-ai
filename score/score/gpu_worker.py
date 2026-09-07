"""GPU 워커 루프(#75) — 파이프라인 v2 의 score 실행 모양. `python -m score worker --gpu`.

    부팅 → 러너 1회 로드(Scorer) → 루프:
        claim_batch(32)  ← photo_analysis 를 FOR UPDATE SKIP LOCKED 로 잠근 채
        미리보기 다운로드(16 스레드) → CLIP·ARNIQA·classical → write_scores(commit = 잠금 해제)
        집을 게 없으면 poll 초 대기. 연속 유휴가 idle_stop 초를 넘기면 자기 인스턴스를 StopInstances 하고 끝난다.

갤러리를 배정받지 않는다 — 임베딩이 끝난 사진이면 누구 것이든 집는다. 그래서 인스턴스 2대가 한 갤러리를 나눠 먹어도,
한 대가 두 갤러리를 섞어 먹어도 된다. 켜는 것·폴백 결정은 wes 의 몫이고, 워커는 켜지면 일하고 없으면 끈다.

실패 처리: 배치 처리 중 예외 → rollback(잠금 반환) → 다음 루프. 한 장이 계속 실패하면(pipeline 의 failed) 이 프로세스에서는
더 집지 않는다(독성 목록) — 워커가 재시작하면 다시 시도한다. 잡 테이블은 건드리지 않는다(완료는 wes 가 데이터로 관측).
"""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path

from score import db, pipeline
from score.config import Settings
from score.gallery import download_previews
from score.storage import PreviewStorage
from score.store import DbStore

log = logging.getLogger(__name__)

IMDS = "http://169.254.169.254/latest"


def instance_id() -> str | None:
    """IMDSv2 로 자기 인스턴스 id. EC2 가 아니면 None."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{IMDS}/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=1).read().decode()
        req = urllib.request.Request(f"{IMDS}/meta-data/instance-id", headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except Exception:  # noqa: BLE001
        return None


def stop_self() -> bool:
    """자기 인스턴스를 정지한다. 실패해도 예외를 내지 않는다 — wes 의 유휴 감시가 두 번째 안전장치다."""
    iid = instance_id()
    if not iid:
        log.info("[worker] EC2 가 아니라 자기 정지 생략")
        return False
    try:
        import boto3

        boto3.client("ec2").stop_instances(InstanceIds=[iid])
        log.info("[worker] 유휴 — StopInstances %s", iid)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("[worker] StopInstances 실패 (%s: %s) — wes 감시에 맡긴다", type(exc).__name__, exc)
        return False


def loop(settings: Settings, *, once: bool = False, stop_on_idle: bool = True, max_batches: int | None = None) -> dict:
    """루프 본체. 테스트·`--once` 를 위해 처리 요약을 돌려준다."""
    if not settings.s3_bucket:
        raise RuntimeError("S3_BUCKET 이 없다 — 미리보기 버킷")
    connection = db.connect(settings)
    store = DbStore(settings, connection)
    storage = PreviewStorage(settings.s3_bucket)
    work_dir = settings.work_dir / "worker"
    scorer = pipeline.Scorer(settings)
    poison: list[int] = []
    summary = {"batches": 0, "processed": 0, "failed": 0, "idleStopped": False}
    idle_since: float | None = None
    log.info("[worker] 시작 batch=%d poll=%.0fs idle_stop=%ds", settings.worker_batch, settings.worker_poll_seconds,
             settings.worker_idle_stop_seconds)
    try:
        while True:
            try:
                refs = store.claim_batch(settings.worker_batch, exclude=poison)
            except Exception as exc:  # noqa: BLE001 — 접속 끊김 등. 롤백하고 기다린다
                log.warning("[worker] 집기 실패 (%s: %s) — %.0fs 뒤 재시도", type(exc).__name__, str(exc).splitlines()[0],
                            settings.worker_poll_seconds)
                store.rollback()
                if once:
                    raise
                time.sleep(settings.worker_poll_seconds)
                continue

            if not refs:
                store.rollback()                    # 빈 SELECT 도 트랜잭션을 열었다 — 닫아야 다음 집기가 새 스냅샷을 본다
                now = time.monotonic()
                idle_since = idle_since or now
                idle = now - idle_since
                if once:
                    return summary
                if stop_on_idle and settings.worker_idle_stop_seconds > 0 and idle >= settings.worker_idle_stop_seconds:
                    log.info("[worker] %.0fs 동안 집을 사진이 없다 — 정지", idle)
                    summary["idleStopped"] = stop_self()
                    return summary
                time.sleep(settings.worker_poll_seconds)
                continue

            idle_since = None
            t0 = time.monotonic()
            try:
                refs = download_previews(storage, refs, work_dir, workers=settings.download_workers)
                # 집기가 곧 대상 선정이라 force=True — 재개 판정(read_analysis)을 건너뛴다. write_scores 가 commit = 잠금 해제.
                result = pipeline.run(store, "worker", refs, settings, force=True, scorer=scorer)
            except Exception as exc:  # noqa: BLE001 — 배치를 돌려주고 계속 산다
                log.exception("[worker] 배치 %d장 실패 — rollback: %s", len(refs), exc)
                store.rollback()
                if once:
                    raise
                time.sleep(settings.worker_poll_seconds)
                continue
            finally:
                shutil.rmtree(work_dir, ignore_errors=True)   # 미리보기 임시 파일 — 장기 실행이라 매 배치 비운다

            failed = [int(pid) for pid in result.get("failed", []) if str(pid).isdigit()]
            poison.extend(failed)
            summary["batches"] += 1
            summary["processed"] += result.get("processed", 0)
            summary["failed"] += len(failed)
            log.info("[worker] 배치 %d장 %.1fs (장당 %.3fs) 실패 %d 누적 %d장", len(refs), time.monotonic() - t0,
                     (time.monotonic() - t0) / max(1, len(refs)), len(failed), summary["processed"])
            if once or (max_batches is not None and summary["batches"] >= max_batches):
                return summary
    finally:
        connection.close()
