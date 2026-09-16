"""대상 선별 (wes V15) — 임베더가 지난 사진은 preview_key 로 알아본다."""

from __future__ import annotations

from pathlib import Path

from categorize.repository.photos import load_db
from tests.db_fakes import RowConn


def test_load_db_targets_previews_and_ignores_photo_status():
    """#93: wes V15(2026-09-08)가 photos.status 에서 EMBEDDED 를 없앴다 — 그 값을 조건에 걸면 대상이 0장이 된다.
    임베더가 지난 사진은 preview_key 로 알아본다(score 의 load_db 와 같은 규칙)."""
    conn = RowConn(rows=[(11, "previews/a.jpg", None, "Canon", "R5"), (12, "previews/b.jpg", None, None, None)])
    refs = load_db(conn, None, 7, Path("/tmp"), download=False)

    sql, params = conn.executed[0]
    assert "status" not in sql
    assert "JOIN" not in sql and sql.split("FROM")[1].split()[0] == "photos"
    assert "preview_key IS NOT NULL" in sql and "deleted_at IS NULL" in sql
    assert params == (7,)
    assert [(r.photo_id, r.camera, r.path) for r in refs] == [("11", "Canon R5", None), ("12", None, None)]
