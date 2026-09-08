"""wes 공유 Postgres 접속. embedder/db.py의 connect()와 같은 모양이다.

여기엔 SQL이 없다 — 테이블을 읽고 쓰는 SQL은 전부 `store.DbStore`(파이프라인 데이터)와 `gallery`(대상 조회)에 있다.
잡 테이블은 건드리지 않는다(#98, wes 소유).
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from score.config import Settings


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
    # vector 컬럼을 ndarray로 주고받는다. 없으면 '[…]' 문자열을 손으로 조립해야 한다.
    register_vector(connection)
    return connection
