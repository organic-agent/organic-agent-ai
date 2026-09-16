"""환경변수를 한곳에서 읽는다. Lambda 는 Terraform 이, 로컬은 셸이 값을 넣는다.

DB 접속은 원래 RDS IAM 토큰이었으나 조직 SCP 가 `rds-db:connect` 를 막아 임시로 `DB_PASSWORD` 를 쓴다.
원복 절차는 인프라 레포 `docs/runbook.md` "SCP 차단" 절.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    db_host: str
    db_port: int
    db_name: str
    db_user: str

    #: embedder 전용 사용자 비밀번호. SCP 우회용 임시 필드 — IAM 토큰으로 돌아가면 지운다.
    db_password: str

    #: IAM 토큰 서명용 호스트. 로컬 터널에서는 db_host(localhost)와 다르다. 지금은 안 쓰지만 원복 때 필요.
    db_auth_host: str

    #: Lambda 기본 verify-full. 로컬 터널은 호스트명이 안 맞아 require 로 낮춘다.
    db_sslmode: str
    db_sslrootcert: str

    s3_bucket: str

    embed_dim: int
    batch_size: int
    model_id: str

    #: 디코딩 직후 줄이는 긴 변 길이. 모델은 어차피 224 로 다시 줄이므로 목적은 메모리 절약이다.
    #: 미리보기 파생본도 이 크기로 나간다.
    resize_long_edge: int

    #: 파생본 JPEG 품질. 82 는 1024px 에서 열화가 안 보이면서 장당 ~200KB 인 지점.
    preview_quality: int

    #: Lambda 타임아웃 앞 여유(초). 남은 시간 < 가장 오래 걸린 배치 + 이 값이면 배치 경계에서 멈추고 commit 한다.
    stop_margin_seconds: int = 60

    #: 원본 GET 을 미리 받는 스레드 수. 순차 다운로드면 네트워크와 CPU 가 서로 기다린다.
    download_workers: int = 4

    #: 빌드 때 받은 모델 snapshot 과 런타임 로드를 같은 commit 으로 고정한다.
    model_revision: str = "5931719e67bbdb9737e363e781fb0c67687896bc"
    @staticmethod
    def from_env() -> "Settings":
        db_host = _required("DB_HOST")

        return Settings(
            db_host=db_host,
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=_required("DB_NAME"),
            db_user=_required("DB_USER"),
            db_password=_required("DB_PASSWORD"),
            db_auth_host=os.environ.get("DB_AUTH_HOST") or db_host,
            db_sslmode=os.environ.get("DB_SSLMODE", "verify-full"),
            db_sslrootcert=os.environ.get("DB_SSLROOTCERT", "/opt/rds-ca/global-bundle.pem"),
            s3_bucket=_required("S3_BUCKET"),
            embed_dim=int(os.environ.get("EMBED_DIM", "768")),
            batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "8")),
            model_id=os.environ.get("EMBED_MODEL_ID", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
            resize_long_edge=int(os.environ.get("RESIZE_LONG_EDGE", "1024")),
            preview_quality=int(os.environ.get("PREVIEW_QUALITY", "82")),
            stop_margin_seconds=int(os.environ.get("STOP_MARGIN_SECONDS", "60")),
            download_workers=int(os.environ.get("EMBED_DOWNLOAD_WORKERS", "4")),
            model_revision=os.environ.get(
                "EMBED_MODEL_REVISION",
                "5931719e67bbdb9737e363e781fb0c67687896bc",
            ),
        )


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        # 사진 절반을 처리한 뒤 발견하는 것보다 초기화 때 죽는 편이 낫다.
        raise RuntimeError(f"환경변수 {name}이(가) 비어 있습니다")
    return value
