"""GPU 워커 루프(#75) — 파이프라인 v2 의 score 실행 모양. `python -m score worker --gpu`.

    부팅 → 러너 1회 로드(Scorer) → 루프:
        claim_batch(32)  ← photo_analysis 를 FOR UPDATE SKIP LOCKED 로 잠근 채
        미리보기 다운로드(16 스레드) → CLIP·ARNIQA·classical → write_scores(commit = 잠금 해제)
        집을 게 없으면 poll 초 대기. 연속 유휴가 idle_stop 초를 넘기면 **끝난다** — `stop_on_idle` 이면 그 전에
        자기 인스턴스를 StopInstances 한다(#103: 종료와 정지는 별개다). `WORKER_IDLE_STOP_SECONDS=0` 이면 끝나지 않는다.

갤러리를 배정받지 않는다 — 임베딩이 끝난 사진이면 누구 것이든 집는다. 그래서 인스턴스 2대가 한 갤러리를 나눠 먹어도,
한 대가 두 갤러리를 섞어 먹어도 된다. 켜는 것·폴백 결정은 wes 의 몫이고, 워커는 켜지면 일하고 없으면 끈다.

실패 처리: 배치 처리 중 예외 → rollback(잠금 반환) → 다음 루프. 사진 단위 결정적 실패 — 미리보기가 S3 에 없음(404)·
디코드 실패(pipeline 의 failed) — 는 `photo_analysis.error` 에 쓴다(#85, wes V15): 집기가 `error IS NULL` 이라 다시 안 집고,
wes 는 그 장을 기대 장수에서 뺀다. 독성 목록은 그 위의 이중 안전장치(같은 프로세스에서 error 쓰기 자체가 실패한 경우).
잡 테이블은 건드리지 않는다(완료는 wes 가 데이터로 관측).
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
#: `photo_analysis.error` 값(#85). wes 는 값을 해석하지 않고 "실패했다"로만 본다 — 짧은 코드로 둔다.
PREVIEW_MISSING = "PREVIEW_MISSING"
SCORE_FAILED = "SCORE_FAILED"


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


class _Lane:
    """커넥션 하나 + 그 커넥션이 잠근 배치(#79). 두 레인이 교대한다 — 한 레인이 점수를 내는 동안 다른 레인이 다음 배치를
    잠그고 미리보기를 내려받는다. 잠금이 트랜잭션에 묶여 있어 같은 커넥션으로는 다음 배치를 미리 잠글 수 없기 때문이다."""

    def __init__(self, settings: Settings, index: int) -> None:
        self.connection = db.connect(settings)
        self.store = DbStore(settings, self.connection)
        self.work_dir = settings.work_dir / f"worker-{index}"
        self.refs: list = []
        #: 이번 집기에서 미리보기가 S3 에 없던 사진(#85). 잠근 트랜잭션 안에서 error 를 쓰고 배치와 함께 commit 한다.
        self.missing: list[str] = []

    def claim_and_download(self, settings: Settings, storage, poison: list[int]) -> list:
        """잠그고 내려받는다. 예외는 호출자가 rollback 한다."""
        self.missing = []
        refs = self.store.claim_batch(settings.worker_batch, exclude=poison)
        if refs:
            refs = download_previews(storage, refs, self.work_dir, workers=settings.download_workers, missing=self.missing)
            if self.missing:
                self.store.write_errors(self.missing, PREVIEW_MISSING)
        self.refs = refs
        return refs

    def rollback(self) -> None:
        self.refs = []
        self.missing = []
        try:
            self.store.rollback()
        except Exception:  # noqa: BLE001
            pass

    def cleanup(self) -> None:
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def close(self) -> None:
        self.connection.close()


def loop(settings: Settings, *, once: bool = False, stop_on_idle: bool = True, max_batches: int | None = None) -> dict:
    """루프 본체. 테스트·`--once` 를 위해 처리 요약을 돌려준다.

    교대 규칙: 레인 A 의 배치를 점수 내기 전에 레인 B 의 claim+download 를 스레드에 건다. A 가 commit 하면 B 가 "현재"가 되고
    A 는 다음을 미리 받는다. 집을 게 없으면(빈 배치) 그 레인은 rollback 으로 스냅샷을 닫고 대기한다."""
    from concurrent.futures import ThreadPoolExecutor

    if not settings.s3_bucket:
        raise RuntimeError("S3_BUCKET 이 없다 — 미리보기 버킷")
    max_batches = max_batches if max_batches is not None else (settings.worker_max_batches or None)
    lanes = [_Lane(settings, 0), _Lane(settings, 1)]
    storage = PreviewStorage(settings.s3_bucket)
    scorer = pipeline.Scorer(settings)
    scorer.warm_up()
    prefetch = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")
    poison: list[int] = []
    summary = {"batches": 0, "processed": 0, "failed": 0, "idleStopped": False, "aborted": False}
    consecutive_failures = 0

    def failed_again(what: str) -> bool:
        """연속 실패를 세고 상한이면 True(루프 종료)."""
        nonlocal consecutive_failures
        consecutive_failures += 1
        cap = settings.worker_max_consecutive_failures
        if cap > 0 and consecutive_failures >= cap:
            log.error("[worker] %s %d회 연속 실패 — 루프 종료 (같은 오류로 헛돌지 않게)", what, consecutive_failures)
            summary["aborted"] = True
            return True
        return False
    idle_since: float | None = None
    log.info("[worker] 시작 batch=%d poll=%.0fs idle_stop=%ds", settings.worker_batch, settings.worker_poll_seconds,
             settings.worker_idle_stop_seconds)
    current, other = lanes
    pending = None            # other 레인의 claim+download Future
    try:
        while True:
            # 현재 레인의 배치를 확보한다 — 앞 루프가 미리 걸어 둔 Future 가 있으면 그 결과, 없으면 지금 집는다.
            try:
                refs = pending.result() if pending is not None else current.claim_and_download(settings, storage, poison)
            except Exception as exc:  # noqa: BLE001 — 접속 끊김·S3 오류. 롤백하고 기다린다
                log.warning("[worker] 집기/다운로드 실패 (%s: %s) — %.0fs 뒤 재시도", type(exc).__name__,
                            str(exc).splitlines()[0], settings.worker_poll_seconds)
                current.rollback()
                pending = None
                if once:
                    raise
                if failed_again("집기/다운로드"):
                    return summary
                time.sleep(settings.worker_poll_seconds)
                continue
            pending = None

            if not refs:
                if current.missing:                 # 집은 게 전부 미리보기 없음 — error 만 commit 하고 유휴로 세지 않는다
                    summary["failed"] += len(current.missing)
                    current.missing = []
                    current.store.commit()
                    current, other = other, current
                    continue
                current.rollback()                  # 빈 SELECT 도 트랜잭션을 열었다 — 닫아야 다음 집기가 새 스냅샷을 본다
                now = time.monotonic()
                idle_since = idle_since or now
                idle = now - idle_since
                if once:
                    return summary
                # 유휴 상한에 닿으면 **항상 끝난다**(#103). 인스턴스를 정지할지만 stop_on_idle 이 정한다 —
                # 옛 코드는 둘을 한 조건에 묶어 `--no-idle-stop` 이 종료까지 막았고, 그래서 로컬에서 큐를 비우고
                # 끝나는 실행이 불가능했다(EC2 가 아닌 곳에서 프로세스가 영원히 돈다).
                if settings.worker_idle_stop_seconds > 0 and idle >= settings.worker_idle_stop_seconds:
                    log.info("[worker] %.0fs 동안 집을 사진이 없다 — 종료%s", idle,
                             "" if stop_on_idle else " (인스턴스 정지 안 함)")
                    if stop_on_idle:
                        summary["idleStopped"] = stop_self()
                    return summary
                time.sleep(settings.worker_poll_seconds)
                continue

            idle_since = None
            # 다음 배치를 다른 레인이 미리 잠그고 받는다 — 지금 배치의 GPU 시간과 겹친다.
            pending = prefetch.submit(other.claim_and_download, settings, storage, poison)
            t0 = time.monotonic()
            try:
                # 집기가 곧 대상 선정이라 force=True — 재개 판정(read_analysis)을 건너뛴다. write_scores 가 commit = 잠금 해제.
                result = pipeline.run(current.store, "worker", refs, settings, force=True, scorer=scorer)
            except Exception as exc:  # noqa: BLE001 — 배치를 돌려주고 계속 산다
                log.exception("[worker] 배치 %d장 실패 — rollback: %s", len(refs), exc)
                current.rollback()
                current, other = other, current
                if once:
                    raise
                if failed_again("배치"):
                    return summary
                time.sleep(settings.worker_poll_seconds)
                continue
            finally:
                current.cleanup()                   # 미리보기 임시 파일 — 장기 실행이라 매 배치 비운다

            consecutive_failures = 0
            failed = [int(pid) for pid in result.get("failed", []) if str(pid).isdigit()]
            poison.extend(failed)
            # 결정적 실패를 남긴다(#85). write_scores 가 이미 commit 했으니(잠금 해제) 여기서 다시 commit — 실패한 장은 잠금이
            # 풀린 짧은 틈에 다른 워커가 집을 수 있지만, 그쪽도 같은 실패 → 같은 error 를 쓴다. 이 쓰기가 실패하면 독성 목록이 막는다.
            errors = len(current.missing) + len(failed)
            if failed:
                try:
                    current.store.write_errors([str(pid) for pid in failed], SCORE_FAILED)
                except Exception as exc:  # noqa: BLE001
                    log.warning("[worker] error 기록 실패 (%s) — 독성 목록으로만 막는다", exc)
            try:
                current.store.commit()
            except Exception as exc:  # noqa: BLE001
                log.warning("[worker] commit 실패 (%s)", exc)
            current.missing = []
            summary["batches"] += 1
            summary["processed"] += result.get("processed", 0)
            summary["failed"] += errors
            took = time.monotonic() - t0
            log.info("[worker] 배치 %d장 %.1fs (장당 %.3fs) 실패 %d 누적 %d장", len(refs), took, took / max(1, len(refs)),
                     errors, summary["processed"])
            # wes 와 합의한 한 줄(pipeline-v2-wes.md §3.7) — CloudWatch Logs Insights 가 key=value 로 표를 만든다.
            log.info("score worker batch=%d photos=%d failed=%d seconds=%.1f", settings.worker_batch,
                     result.get("processed", 0), errors, took)
            current, other = other, current
            if once or (max_batches is not None and summary["batches"] >= max_batches):
                if pending is not None:             # 미리 잠근 배치는 돌려준다
                    try:
                        pending.result()
                    except Exception:  # noqa: BLE001
                        pass
                    current.rollback()
                return summary
    finally:
        prefetch.shutdown(wait=False, cancel_futures=True)
        for lane in lanes:
            lane.close()
