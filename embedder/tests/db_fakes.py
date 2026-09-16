"""repository 테스트용 가짜 psycopg 커넥션. 실행한 SQL 과 파라미터를 `executed` 에 모은다.

DB 드라이버·이미지·벡터를 실행하지 않는다. Lambda 의존성을 전부 설치하지 않은 로컬/CI에서도 SQL 경계를
검증할 수 있도록 import 자리만 최소 대체한다(`install_stubs`).
"""

from __future__ import annotations

import sys
import types


def install_stubs() -> None:
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        sys.modules["numpy"] = types.ModuleType("numpy")

    try:
        import psycopg  # noqa: F401
    except ModuleNotFoundError:
        psycopg_module = types.ModuleType("psycopg")
        psycopg_module.Error = RuntimeError
        sys.modules["psycopg"] = psycopg_module

    try:
        from pgvector.psycopg import register_vector  # noqa: F401
    except ModuleNotFoundError:
        pgvector_package = types.ModuleType("pgvector")
        pgvector_package.__path__ = []
        pgvector_psycopg = types.ModuleType("pgvector.psycopg")
        pgvector_psycopg.register_vector = lambda connection: None
        sys.modules["pgvector"] = pgvector_package
        sys.modules["pgvector.psycopg"] = pgvector_psycopg


class _Cursor:
    def __init__(self, connection: "_Connection"):
        self.connection = connection

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None

    def execute(self, sql: str, params: tuple) -> None:
        self.connection.executed.append((" ".join(sql.split()), params))
        self.rowcount = self.connection.next_rowcount()

    def executemany(self, sql: str, params) -> None:
        rows = tuple(params)
        self.connection.executed.append((" ".join(sql.split()), rows))
        self.rowcount = self.connection.next_rowcount()

    def fetchall(self):
        return self.connection.rows

    def fetchone(self):
        return self.connection.rows[0] if self.connection.rows else None


class _Connection:
    def __init__(self, rows: list[tuple], rowcounts: list[int] | None = None):
        self.rows = rows
        self.executed: list[tuple[str, tuple]] = []
        self.rowcounts = iter(rowcounts or [])

    def cursor(self) -> _Cursor:
        return _Cursor(self)

    def next_rowcount(self) -> int:
        return next(self.rowcounts, len(self.rows))


class _ManyCursor(_Cursor):
    def executemany(self, sql: str, rows) -> None:
        rows = list(rows)
        self.connection.executed.append((" ".join(sql.split()), rows))
        # 실제 드라이버처럼 배치 크기를 갱신 행 수로 본다. rowcounts를 넘긴 테스트는 그 값을 쓴다.
        self.rowcount = next(self.connection.rowcounts, len(rows))


class _ManyConnection(_Connection):
    def cursor(self) -> _ManyCursor:
        return _ManyCursor(self)
