"""psycopg 커넥션 가짜 — 실행된 SQL 과 파라미터를 기록하고, 준비된 행을 돌려준다. DB 는 붙지 않는다."""

from __future__ import annotations


class _Cursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def execute(self, sql, params=None):
        self.conn.executed.append((" ".join(sql.split()), params))

    def fetchall(self):
        return self.conn.rows


class RowConn:
    """SELECT 가짜 — fetchall 이 [rows] 를 돌려준다."""

    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def cursor(self):
        return _Cursor(self)


class JobConn:
    """UPDATE 가짜 — commit · rollback 횟수를 센다."""

    def __init__(self):
        self.rows = []
        self.executed = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass
