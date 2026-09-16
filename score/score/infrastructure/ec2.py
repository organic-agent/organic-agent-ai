"""EC2 자기 인스턴스 — IMDSv2 로 id·리전을 읽고 StopInstances 한다. GPU 워커의 유휴 정지(#75·#103)."""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

IMDS = "http://169.254.169.254/latest"


def _imds(path: str) -> str | None:
    """IMDSv2 로 `meta-data/<path>` 를 읽는다. EC2 가 아니면 None."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{IMDS}/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=1).read().decode()
        req = urllib.request.Request(f"{IMDS}/meta-data/{path}", headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except Exception:  # noqa: BLE001
        return None


def instance_id() -> str | None:
    """IMDSv2 로 자기 인스턴스 id. EC2 가 아니면 None."""
    return _imds("instance-id")


def instance_region() -> str | None:
    """IMDSv2 로 자기 인스턴스의 리전. 워커 컨테이너에는 AWS_REGION 이 없어 boto3 가 리전을 못 정하므로(NoRegionError,
    2026-09-09 운영 실측) 인스턴스 id 와 같은 곳에서 읽어 명시한다."""
    return _imds("placement/region")


def stop_self() -> bool:
    """자기 인스턴스를 정지한다. 실패해도 예외를 내지 않는다 — wes 의 유휴 감시가 두 번째 안전장치다."""
    iid = instance_id()
    if not iid:
        log.info("[worker] EC2 가 아니라 자기 정지 생략")
        return False
    try:
        import boto3

        boto3.client("ec2", region_name=instance_region()).stop_instances(InstanceIds=[iid])
        log.info("[worker] 유휴 — StopInstances %s", iid)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("[worker] StopInstances 실패 (%s: %s) — wes 감시에 맡긴다", type(exc).__name__, exc)
        return False
