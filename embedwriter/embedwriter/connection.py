"""wes 공유 Postgres 접속. embedder 의 repository/connection.py 와 같은 접속 인자다.

`pgvector.psycopg.register_vector` 는 부르지 않는다 — 벡터를 `'[…]'` 텍스트 배열로 넘기고 SQL 에서 `::vector` 로
바꾸므로(writer.py) 파이썬 쪽 vector 타입 어댑터가 필요 없다. 그만큼 의존성(pgvector·numpy)이 빠진다.

웜 컨테이너에서는 접속 하나를 다시 쓴다(`shared`). 동시 실행 ≤ 2 라 접속 수가 늘지 않고, 호출마다 TLS 접속을 새로
맺는 비용(수십~수백 ms)을 아낀다. 끊긴 접속은 다음 호출이 버리고 새로 맺는다.
"""

from __future__ import annotations

import psycopg

from embedwriter.settings import Settings

_shared: psycopg.Connection | None = None


def connect(settings: Settings) -> psycopg.Connection:
    return psycopg.connect(
        host=settings.db_host,
        port=settings.db_port,
        dbname=settings.db_name,
        user=settings.db_user,
        password=settings.db_password,
        # RDS PostgreSQL 15+ 는 rds.force_ssl 기본 1 이라 TLS 없이는 접속 자체가 거부된다.
        sslmode=settings.db_sslmode,
        sslrootcert=settings.db_sslrootcert,
        connect_timeout=10,
        # 세션 이름이 pg_stat_activity 에 보이면 DB 시간을 누가 쓰는지 바로 가른다(#286 측정).
        application_name="embedwriter",
    )


def shared(settings: Settings) -> psycopg.Connection:
    """웜 호출 사이에 재사용하는 접속. 닫혔거나 트랜잭션이 깨진 채 남았으면 새로 맺는다."""
    global _shared
    if _shared is not None and (_shared.closed or _shared.broken):
        _shared = None
    if _shared is None:
        _shared = connect(settings)
    return _shared


def discard() -> None:
    """접속 오류 뒤 부른다. 다음 호출이 새로 맺는다."""
    global _shared
    if _shared is not None:
        try:
            _shared.close()
        except Exception:
            pass
    _shared = None
