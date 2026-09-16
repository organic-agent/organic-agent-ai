"""wes 공유 Postgres 접속.

원래 RDS IAM 토큰 인증이었으나 조직 SCP 가 `rds-db:connect` 를 막아 지금은 비밀번호를 쓴다.
원복 절차는 인프라 레포 `docs/runbook.md` "SCP 차단" 절.
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from embedder.config.settings import Settings


def connect(settings: Settings) -> psycopg.Connection:
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
