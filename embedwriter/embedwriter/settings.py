"""환경변수를 한곳에서 읽는다. Lambda 는 Terraform 이, 로컬은 셸이 값을 넣는다.

DB 접속은 embedder 와 같다 — 같은 `embedder` DB 역할(V15 EMBEDDER_GRANT_CONTRACT 의 권한)과 비밀번호를 쓴다.
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
    #: embedder 역할 비밀번호. embedder 와 같은 SCP 우회 임시 필드 — IAM 토큰으로 돌아가면 지운다.
    db_password: str
    #: Lambda 기본 verify-full. 로컬 터널은 호스트명이 안 맞아 require 로 낮춘다.
    db_sslmode: str
    db_sslrootcert: str
    #: vector(n) 컬럼 차원. 메시지 벡터 길이가 다르면 그 메시지만 실패로 돌린다(묶음 전체를 깨지 않게).
    embed_dim: int = 768

    @staticmethod
    def from_env() -> "Settings":
        return Settings(
            db_host=_required("DB_HOST"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=_required("DB_NAME"),
            db_user=_required("DB_USER"),
            db_password=_required("DB_PASSWORD"),
            db_sslmode=os.environ.get("DB_SSLMODE", "verify-full"),
            db_sslrootcert=os.environ.get("DB_SSLROOTCERT", "/opt/rds-ca/global-bundle.pem"),
            embed_dim=int(os.environ.get("EMBED_DIM", "768")),
        )


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        # 메시지를 받은 뒤 발견하는 것보다 초기화 때 죽는 편이 낫다.
        raise RuntimeError(f"환경변수 {name}이(가) 비어 있습니다")
    return value
