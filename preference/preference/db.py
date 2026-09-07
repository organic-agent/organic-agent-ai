"""wes 공유 Postgres 접속. categorize/db.py 와 같은 모양 — SQL 은 전부 `store.DbStore` 에 있다."""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from preference.config import Settings


def connect(settings: Settings) -> psycopg.Connection:
    if not settings.db_enabled:
        raise SystemExit("DB 모드인데 DB_HOST/DB_NAME/DB_USER 가 없다 (DB_PASSWORD, DB_SSLMODE 도 확인)")
    kwargs = dict(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        sslmode=settings.db_sslmode,
        connect_timeout=10,
    )
    if settings.db_sslrootcert:
        kwargs["sslrootcert"] = settings.db_sslrootcert
    connection = psycopg.connect(**kwargs)
    register_vector(connection)
    return connection
