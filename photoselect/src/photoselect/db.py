"""wes 공유 Postgres 접속. embedder/db.py의 connect()와 같은 모양이다.

여기엔 SQL이 없다 — 테이블을 읽고 쓰는 SQL은 전부 `store.DbStore`(파이프라인 데이터)와
`jobs`(잡 상태)에 있다. 두 모듈이 한 커넥션을 나눠 쓰므로 트랜잭션 경계는 호출자가 잡는다.
"""

from __future__ import annotations

import psycopg
from pgvector.psycopg import register_vector

from photoselect.config import Settings


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
