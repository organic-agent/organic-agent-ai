"""Lambda 진입점.

앱이 `POST /api/v1/galleries/{id}/embeddings/run`을 받으면 이 함수를 EVENT(비동기)로 부른다.
페이로드는 `{"galleryId": 1, "force": false}`. 관리자 사진 교체 outbox는 `{"jobId": …}`를 보낸다.
v2 스트리밍(#73)은 `{"galleryId": 1, "photoIds": [101, 102, …]}` — 그 목록만 임베딩하고 끝난다(잠금·샤딩·재호출 없음).
샤드 실행은 여기에 `"shard": {"index": i, "total": n}` 과 `"runStartedAt"` 이 붙는다(#56) — 조정자(wes 호출)가
자기 함수를 n번 EVENT 할 때 만드는 페이로드라 wes 계약은 그대로다.

갤러리 잡은 15분 타임아웃 앞에서 스스로 멈춘다(job.run의 `remaining_seconds`). 멈췄고 이번
실행에서 한 장이라도 처리했으면 같은 갤러리·샤드로 자기 자신을 EVENT 재호출해 이어 간다 --
`force`는 넘기지 않고 `runStartedAt` 만 넘긴다. 이미 끝난 사진(force 면 이번 실행이 적은 벡터)은
fetch_targets가 건너뛰므로 재호출은 곧 재개다.
처리 0장이면 재호출하지 않는다: 같은 사진이 계속 실패하는 갤러리에서 무한히 도는 것을 막는다.

재호출은 best-effort다. 이 함수가 붙는 서브넷에는 NAT가 없어 Lambda API에 닿으려면 인터페이스
VPC 엔드포인트(`com.amazonaws.<region>.lambda`)가 있어야 한다. 없으면 호출이 타임아웃으로
실패하는데, 그때도 결과의 `stopped: true, remaining: N`은 그대로 돌아가고 `reinvoked: false`로
드러난다 -- 진행분은 이미 commit돼 있으므로 앱이 다시 부르면 이어서 한다.
"""

from __future__ import annotations

import json
import logging

from embedder import admin_job, job
from embedder.admin_event import AdminPhotoEvent
from embedder.config import Settings

log = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

# 설정은 가볍게 읽되 모델은 올리지 않는다. DERIVATIVE/QUALITY_ANALYSIS는 torch나 DINO를
# 전혀 쓰지 않고, EMBEDDING의 첫 호출만 model.load_from의 프로세스 캐시를 채운다.
_SETTINGS = Settings.from_env()


def handler(event: dict, context) -> dict:
    if "jobId" in event:
        return admin_job.run(AdminPhotoEvent.from_payload(event), _SETTINGS)

    gallery_id = event.get("galleryId")
    if gallery_id is None:
        raise ValueError("페이로드에 galleryId가 없습니다")
    gallery_id = int(gallery_id)
    if event.get("photoIds") is not None:
        # v2 스트리밍(#73): wes 스위퍼가 배정한 사진 목록만. 잠금·fan-out·재호출 없음 — 남은 장은 wes 가 다시 배정한다.
        photo_ids = [int(pid) for pid in event["photoIds"]]
        log.info("갤러리 %s: 사진 %s장 배치", gallery_id, len(photo_ids))
        return job.run(gallery_id=gallery_id, settings=_SETTINGS, remaining_seconds=_remaining_seconds(context),
                       photo_ids=photo_ids)

    force = bool(event.get("force", False))
    shard = job.Shard.from_payload(event.get("shard"))
    run_started_at = event.get("runStartedAt") or (job.now_iso() if force else None)

    def fan_out(total: int, started_at: str | None) -> bool:
        return all(
            _invoke_self(context, gallery_id, _payload(gallery_id, force=force, shard=job.Shard(i, total),
                                                       run_started_at=started_at))
            for i in range(total)
        )

    result = job.run(
        gallery_id=gallery_id,
        force=force,
        settings=_SETTINGS,
        remaining_seconds=_remaining_seconds(context),
        shard=shard,
        run_started_at=run_started_at,
        fan_out=None if shard is not None else fan_out,
    )

    if result.get("stopped") and result.get("processed", 0) > 0:
        result["reinvoked"] = reinvoke(context, gallery_id, shard=shard, run_started_at=run_started_at)
    return result


def _remaining_seconds(context):
    """Lambda context의 남은 시간을 초 단위 함수로. 로컬 테스트처럼 context가 없으면 None."""
    getter = getattr(context, "get_remaining_time_in_millis", None)
    if getter is None:
        return None
    return lambda: getter() / 1000.0


def _payload(gallery_id: int, *, force: bool, shard: job.Shard | None, run_started_at: str | None) -> dict:
    """자기 호출 페이로드. 재호출은 force=False + runStartedAt(이번 실행 전 벡터는 재계산), 샤드는 shard 포함."""
    payload: dict = {"galleryId": gallery_id, "force": force}
    if shard is not None:
        payload["shard"] = shard.to_payload()
    if run_started_at:
        payload["runStartedAt"] = run_started_at
    return payload


def reinvoke(context, gallery_id: int, shard: job.Shard | None = None, run_started_at: str | None = None) -> bool:
    """같은 갤러리·샤드로 자기 자신을 EVENT 호출한다. 실패해도 예외를 올리지 않고 False."""
    ok = _invoke_self(context, gallery_id, _payload(gallery_id, force=False, shard=shard, run_started_at=run_started_at))
    if ok:
        log.info("갤러리 %s: 남은 사진을 위해 자기 재호출 (shard=%s)", gallery_id, shard)
    return ok


def _invoke_self(context, gallery_id: int, payload: dict) -> bool:
    function_name = getattr(context, "function_name", None)
    if not function_name:
        log.warning("갤러리 %s: 호출할 함수 이름이 없다 (context.function_name)", gallery_id)
        return False
    try:
        import boto3
        from botocore.config import Config

        # 엔드포인트가 없으면 연결 자체가 안 된다. 기본 재시도·타임아웃(수 분)을 기다리며 남은
        # 시간을 다 쓰지 않도록 짧게 끊는다 -- 어차피 실패하면 앱의 재호출로 이어진다.
        client = boto3.client(
            "lambda",
            config=Config(connect_timeout=5, read_timeout=10, retries={"max_attempts": 1}),
        )
        client.invoke(
            FunctionName=function_name,
            InvocationType="Event",
            Payload=json.dumps(payload).encode("utf-8"),
        )
        return True
    except Exception:
        log.exception("갤러리 %s: 자기 호출 실패 (%s) -- 앱이 다시 부르면 이어서 한다", gallery_id, payload)
        return False
