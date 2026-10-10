"""wes 공유 Postgres 접속.

원래 RDS IAM 토큰 인증이었으나 조직 SCP 가 `rds-db:connect` 를 막아 지금은 비밀번호를 쓴다.
원복 절차는 인프라 레포 `docs/runbook.md` "SCP 차단" 절.

연결 상한에 막히면 잠깐 쉬었다 다시 연다. embedder 역할에는 `CONNECTION LIMIT 32` 가 걸려 있고(인프라 런북),
배치마다 짧게 접속하므로 동시 실행 48개가 같은 순간 적재하면 33번째부터 거절된다(wes #274 R-1: 4호출 실패 →
200장이 wes 재배정 10분을 기다렸다). 거절은 잠깐이라 몇 초 뒤면 자리가 난다.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Callable

import psycopg
from pgvector.psycopg import register_vector

from embedder.config.settings import Settings

log = logging.getLogger(__name__)

#: 연결 상한에 막혔을 때 다시 시도하기 전 쉬는 시간(초). 지터 ×0.5~1.5 를 곱해 동시에 막힌 호출들이 같은 순간
#: 다시 몰리지 않게 한다. 합은 최대 ≈ 23초로, Lambda 데드라인 여유(stop_margin_seconds 60)보다 짧다.
CONNECT_RETRY_DELAYS = (0.5, 1.0, 2.0, 4.0, 8.0)

#: 역할 상한(`too many connections for role`)과 서버 max_connections(`too many clients already`) 둘 다 SQLSTATE 53300 이지만,
#: 접속 단계 오류는 psycopg 가 sqlstate 를 채우지 않아 메시지로 가린다.
_CONNECTION_LIMIT_MESSAGES = ("too many connections", "too many clients")


def connect(settings: Settings, sleep: Callable[[float], None] = time.sleep) -> psycopg.Connection:
    for delay in CONNECT_RETRY_DELAYS:
        try:
            return _open(settings)
        except psycopg.OperationalError as error:
            if not _is_connection_limit(error):
                raise
            wait = delay * random.uniform(0.5, 1.5)
            log.warning("DB 연결 상한에 막혔습니다 — %.1f초 뒤 다시 엽니다: %s", wait, error)
            sleep(wait)
    # 마지막 한 번은 실패를 그대로 올린다 — 호출이 실패하면 wes 가 그 사진들을 다시 배정한다.
    return _open(settings)


def _open(settings: Settings) -> psycopg.Connection:
    # SCP 가 풀리면 password 대신 boto3 rds.generate_db_auth_token(DBHostname=settings.db_auth_host, ...) 으로
    # 돌아간다. 그때 DB 에 `GRANT rds_iam TO embedder;` 가 필요하고, 반대로 지금은 그 GRANT 가 있으면
    # pg_hba 가 PAM 으로 보내 `PAM authentication failed` 가 난다.
    connection = psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        # RDS PostgreSQL 15+ 는 rds.force_ssl 기본 1 이라 TLS 없이는 접속 자체가 거부된다.
        sslmode=settings.db_sslmode,
        sslrootcert=settings.db_sslrootcert,
        connect_timeout=10,
    )

    # 파이썬 리스트/ndarray 를 vector 컬럼에 그대로 바인딩하기 위해 필요하다.
    register_vector(connection)
    return connection


def _is_connection_limit(error: psycopg.OperationalError) -> bool:
    message = str(error)
    return any(text in message for text in _CONNECTION_LIMIT_MESSAGES)
