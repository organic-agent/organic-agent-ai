"""갤러리 잡 — Lambda handler 와 로컬 CLI 가 같은 `run()` 을 부른다.

    잠금 → (잡이면 RUNNING) → EMBEDDED 사진 목록 + 미리보기 다운로드 → pipeline.run → (잡이면 result 기록)

샤딩(#54): wes 호출(shard 없음)은 **조정자**다 — 사진 수로 N 을 정하고(`plan_shards`) `fan_out(N)` 으로 자기 함수를 N 번 띄운
뒤 바로 끝난다(다운로드·모델 로드 없음). 각 **샤드**는 같은 순서의 목록에서 `index % total == index` 인 사진만 맡고, 잠금은
(갤러리, 샤드) 키다. 샤드가 끝나면 `jobs.shard_done` 카운터를 올리고, total 에 닿은 마지막 샤드만 `lastShard=True` 로 돌아가
handler 가 categorize 를 연다. N=1 이거나 fan_out 이 없으면(로컬 CLI) 지금처럼 한 실행이다.

`ai_analysis_jobs` 는 이 모듈이 **열기만** 한다. DONE 은 체인의 끝인 categorize 가 찍는다. 그래서 잡이 있을 때
categorize 실행기(chain)가 설정돼 있지 않으면 시작 전에 실패시킨다 — 30분 점수를 낸 뒤 매달리는 것보다 낫다.
결과 dict 의 `stopped`·`remaining`·`processed` 를 handler 가 보고 자기 재호출 / 체인 호출을 결정한다.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

from score import db, jobs, pipeline
from score.config import Settings
from score.gallery import download_previews, load_db
from score.storage import PreviewStorage
from score.store import DbStore

log = logging.getLogger(__name__)

ALREADY_RUNNING = "already running"


def was_skipped(result: dict) -> bool:
    """실행 자체를 건너뛰었나(다른 실행이 갤러리 잠금을 쥐고 있었다).

    결과의 `skipped` 는 두 뜻으로 쓰인다 — 잠금 건너뜀이면 [ALREADY_RUNNING] 문자열, 정상 실행이면 "이미 점수가 있어
    건너뛴 사진 수"(int). 후자는 전부 건너뛴 재실행에서 822 같은 참값이 되므로, 진위로 검사하면 "실행을 건너뛰었다"로
    오판해 categorize 체인이 열리지 않고 잡이 RUNNING 에 영원히 남는다. 반드시 이 함수로 판별한다.
    """
    return result.get("skipped") == ALREADY_RUNNING

#: 갤러리 advisory lock 의 앞쪽 키. embedder(0x454D42 'EMB')·다른 프로세스와 키 공간이 겹치지 않게 이 모듈만의 상수.
GALLERY_LOCK_NAMESPACE = 0x53434F  # 'SCO'
#: 뒤쪽 키 = gallery_id * LOCK_STRIDE + shard_index. 샤드가 최대 8 이라 여유 있게 64.
LOCK_STRIDE = 64


@dataclass(frozen=True)
class Shard:
    """N 개 중 index 번째. 사진 목록(고정 순서)에서 위치 % total == index 인 것만 맡는다."""

    index: int
    total: int

    def __post_init__(self) -> None:
        if not (self.total >= 1 and 0 <= self.index < self.total):
            raise ValueError(f"잘못된 샤드 {self.index}/{self.total}")

    def select(self, refs: list) -> list:
        return [r for i, r in enumerate(refs) if i % self.total == self.index]

    def to_payload(self) -> dict:
        return {"index": self.index, "total": self.total}

    @classmethod
    def from_payload(cls, value) -> "Shard | None":
        if not value:
            return None
        return cls(index=int(value["index"]), total=int(value["total"]))


def plan_shards(n_photos: int, settings: Settings) -> int:
    """사진 수로 샤드 수. ceil(n / shard_photos) 를 [1, max_shards] 로 자른다."""
    if settings.shard_photos <= 0 or n_photos <= 0:
        return 1
    return max(1, min(settings.max_shards, math.ceil(n_photos / settings.shard_photos)))


def _run_photo_ids(gallery_id: int, photo_ids: list[int], settings: Settings,
                   remaining_seconds: Callable[[], float] | None, started: float) -> dict:
    from score.gallery import load_by_ids

    connection = db.connect(settings)
    try:
        storage = PreviewStorage(settings.s3_bucket)
        refs = load_by_ids(connection, photo_ids)
        refs = download_previews(storage, refs, settings.work_dir / str(gallery_id), workers=settings.download_workers)
        store = DbStore(settings, connection)
        result = pipeline.run(store, str(gallery_id), refs, settings, force=True, remaining_seconds=remaining_seconds)
        result["photoIds"] = len(photo_ids)
        result["elapsedSeconds"] = round(time.monotonic() - started, 1)
        log.info("갤러리 %s 사진 %d장: 완료 %s", gallery_id, len(photo_ids), result)
        return result
    finally:
        connection.close()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def try_lock_gallery(connection, gallery_id: int, shard_index: int = 0) -> bool:
    """같은 갤러리·샤드의 SCORE 가 이미 돌고 있으면 False. 세션 잠금이라 연결이 닫히면 풀린다.
    샤드 없는 실행은 샤드 0 과 같은 키 — 둘이 겹쳐 돌 일은 없어야 한다."""
    with connection.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s, %s)",
                    (GALLERY_LOCK_NAMESPACE, gallery_id * LOCK_STRIDE + shard_index))
        row = cur.fetchone()
    return bool(row and row[0])


def run(gallery_id: int, force: bool = False, settings: Settings | None = None,
        job_id: int | None = None, limit: int | None = None,
        remaining_seconds: Callable[[], float] | None = None,
        shard: Shard | None = None, run_started_at: str | None = None,
        fan_out: Callable[[int, str | None], bool] | None = None,
        photo_ids: list[int] | None = None) -> dict:
    """shard 가 없고 fan_out 이 있으면 조정자: 샤드 수가 2 이상일 때 fan_out(N, run_started_at) 을 부르고
    `coordinator=True` 로 끝난다. run_started_at(ISO) 은 force 실행의 시작 시각 — 그 이후 점수만 "있음"으로 친다.
    `photo_ids` 가 있으면(v2 폴백, #75) 그 목록만 점수 — 잠금·잡·샤딩·체인 없음. wes 가 배정했으니 무조건 계산한다."""
    started = time.monotonic()
    settings = settings or Settings.from_env()
    if not settings.s3_bucket:
        raise RuntimeError("S3_BUCKET 이 없다 — 미리보기를 내려받을 버킷")
    if photo_ids is not None:
        return _run_photo_ids(gallery_id, photo_ids, settings, remaining_seconds, started)
    if force and run_started_at is None:
        run_started_at = now_iso()
    since = datetime.fromisoformat(run_started_at) if run_started_at else None
    tag = f"갤러리 {gallery_id}" + (f" 샤드 {shard.index}/{shard.total}" if shard else "")

    connection = db.connect(settings)
    try:
        if not try_lock_gallery(connection, gallery_id, shard.index if shard else 0):
            log.info("%s: 다른 실행이 진행 중 — 건너뜀", tag)
            return {"gallery": str(gallery_id), "pipeline": "v3", "mode": "score", "skipped": ALREADY_RUNNING,
                    "processed": 0, "stopped": False, "remaining": 0,
                    "elapsedSeconds": round(time.monotonic() - started, 1)}

        if job_id is not None:
            status = jobs.start(connection, job_id)
            if status not in ("RUNNING",):
                raise RuntimeError(f"잡 {job_id} 은 실행할 수 없는 상태다: {status}")
            if not settings.chain_configured:
                jobs.fail(connection, job_id, "categorize 실행기 미설정 (CATEGORIZE_FUNCTION_NAME | CATEGORIZE_COMMAND)")
                raise RuntimeError("잡은 categorize 까지가 산출물이다 — categorize 실행기를 설정하라")

        try:
            storage = PreviewStorage(settings.s3_bucket)
            all_refs = load_db(connection, storage, gallery_id, settings.work_dir, limit=limit, download=False)
            if not all_refs:
                raise RuntimeError(f"갤러리 {gallery_id} 에 EMBEDDED 사진이 없다 — embedder 가 먼저다")

            if shard is None and fan_out is not None:
                n = plan_shards(len(all_refs), settings)
                if n > 1:
                    if job_id is not None:
                        jobs.shards_begin(connection, job_id, n, run_started_at)
                        jobs.record(connection, job_id, {"score": {"targets": len(all_refs), "shards": n,
                                                                    "runStartedAt": run_started_at}})
                    ok = fan_out(n, run_started_at)
                    result = {"gallery": str(gallery_id), "pipeline": "v3", "mode": "score", "coordinator": True,
                              "targets": len(all_refs), "shards": n, "fannedOut": ok, "runStartedAt": run_started_at,
                              "elapsedSeconds": round(time.monotonic() - started, 1)}
                    if not ok:
                        raise RuntimeError(f"갤러리 {gallery_id}: 샤드 {n}개 호출 실패")
                    log.info("%s: 조정자 — %d장을 샤드 %d개로 (%s)", tag, len(all_refs), n, run_started_at)
                    return result

            mine = shard.select(all_refs) if shard else all_refs
            t_dl = time.monotonic()
            refs = download_previews(storage, mine, settings.work_dir / str(gallery_id), workers=settings.download_workers)
            log.info("%s: 미리보기 %d장 다운로드 %.1fs (workers=%d)", tag, len(refs), time.monotonic() - t_dl,
                     settings.download_workers)
            store = DbStore(settings, connection)
            result = pipeline.run(store, str(gallery_id), refs, settings, force=force and since is None,
                                  remaining_seconds=remaining_seconds, since=since)
        except BaseException as exc:  # noqa: BLE001 — SystemExit 포함, 잡에 실패를 남긴다
            if job_id is not None:
                jobs.fail(connection, job_id, f"{type(exc).__name__}: {exc}")
            raise

        if shard is not None:
            result["shard"] = shard.to_payload()
        if run_started_at:
            result["runStartedAt"] = run_started_at
        if job_id is not None:
            if shard is None:
                jobs.record(connection, job_id, {"score": result})
            else:
                jobs.record_shard(connection, job_id, shard.index, result)
                if not result.get("stopped"):
                    done, total = jobs.shard_done(connection, job_id)
                    result["shardsDone"], result["shardsTotal"] = done, total
                    result["lastShard"] = done >= total
                    log.info("%s: 끝 — 샤드 %d/%d 완료%s", tag, done, total, " (마지막 → categorize)" if done >= total else "")
        log.info("완료: %s", result)
        return result
    finally:
        connection.close()
