"""한 문장 적재의 SQL 가드와 배열 파라미터. 실제 DB 실행은 test_writer_pg.py(선택)."""

from __future__ import annotations

import unittest

from helpers import DIM, FakeConnection, body, photo

from embedwriter import message, writer


def _rows(*photos):
    return message.parse(body(*photos), DIM)


def _flat(sql: str) -> str:
    return " ".join(sql.split())


class SqlGuardTest(unittest.TestCase):
    """store_embeddings 와 같은 CAS — 원본이 바뀐 늦은 결과·휴지통 사진·지워진 갤러리는 0행."""

    def test_live_join_guards_photo_and_gallery(self) -> None:
        sql = _flat(writer.SQL)
        self.assertIn("JOIN photos p ON p.id = i.photo_id AND p.storage_key = i.storage_key AND p.deleted_at IS NULL", sql)
        self.assertIn("JOIN galleries g ON g.id = p.gallery_id AND g.deleted_at IS NULL", sql)

    def test_analysis_upsert_only_swaps_vector_columns(self) -> None:
        sql = _flat(writer.SQL)
        self.assertIn("INSERT INTO photo_analysis (photo_id, embedding, embedding_model, created_at, updated_at) "
                      "SELECT photo_id, embedding::vector, embedding_model, now(), now() FROM live", sql)
        self.assertIn("ON CONFLICT (photo_id) DO UPDATE SET embedding = EXCLUDED.embedding, "
                      "embedding_model = EXCLUDED.embedding_model, version = photo_analysis.version + 1, "
                      "updated_at = now()", sql)

    def test_photos_update_overwrites_preview_and_coalesces_exif(self) -> None:
        sql = _flat(writer.SQL)
        self.assertIn("SET preview_key = i.preview_key,", sql)
        for column in ("taken_at", "camera_make", "camera_model", "exposure_time", "f_number", "iso",
                       "width", "height", "byte_size"):
            self.assertIn(f"{column} = COALESCE(i.{column}, p.{column})", sql)
        self.assertIn("version = p.version + 1, updated_at = now() FROM live i", sql)
        self.assertTrue(sql.endswith(
            "WHERE p.id = i.photo_id AND p.storage_key = i.storage_key AND p.deleted_at IS NULL"))

    def test_only_granted_columns_are_written(self) -> None:
        """embedder 역할의 photos UPDATE 권한(V15) 밖의 컬럼을 쓰면 묶음 전체가 permission denied 다."""
        granted = {"preview_key", "dispatched_at", "taken_at", "camera_make", "camera_model", "exposure_time",
                   "f_number", "iso", "width", "height", "byte_size", "version", "updated_at"}
        update = _flat(writer.SQL).split("UPDATE photos p SET ", 1)[1].split(" FROM live i", 1)[0]
        written = {assignment.split(" = ", 1)[0].strip() for assignment in update.split(", ")
                   if " = " in assignment and not assignment.startswith(("i.", "p."))}
        self.assertLessEqual(written, granted)
        self.assertNotIn("status", _flat(writer.SQL).split("UPDATE photos p", 1)[1])


class BuildParamsTest(unittest.TestCase):
    def test_arrays_are_column_aligned_and_sorted(self) -> None:
        params = writer.build_params(_rows(photo(3), photo(1, metadata=None), photo(2)))
        self.assertEqual([1, 2, 3], params["photo_id"])
        self.assertEqual(["galleries/7/1.jpg", "galleries/7/2.jpg", "galleries/7/3.jpg"], params["storage_key"])
        self.assertEqual(["previews/galleries/7/1.jpg", "previews/galleries/7/2.jpg", "previews/galleries/7/3.jpg"],
                         params["preview_key"])
        self.assertEqual([None, 8192, 8192], params["width"])
        self.assertEqual(["dinov3"] * 3, params["embedding_model"])
        self.assertEqual({len(v) for v in params.values()}, {3})

    def test_duplicate_photo_keeps_last(self) -> None:
        rows = _rows(photo(1, values=[1, 1, 1, 1])) + _rows(photo(1, values=[2, 2, 2, 2]))
        params = writer.build_params(rows)
        self.assertEqual([1], params["photo_id"])
        self.assertEqual(["[2,2,2,2]"], params["embedding"])

    def test_every_unnest_column_has_a_param(self) -> None:
        params = writer.build_params(_rows(photo(1)))
        for name, _ in writer._COLUMNS:
            self.assertIn(f"%({name})s::", writer.SQL)
            self.assertIn(name, params)


class WriteTest(unittest.TestCase):
    def test_one_statement_for_all_rows(self) -> None:
        conn = FakeConnection()
        written = writer.write(conn, _rows(photo(1), photo(2)) + _rows(photo(3)))
        self.assertEqual(3, written)
        self.assertEqual(1, len(conn.executed))
        self.assertEqual(0, conn.commits)   # 커밋은 handler 몫

    def test_empty_rows_do_not_touch_db(self) -> None:
        conn = FakeConnection()
        self.assertEqual(0, writer.write(conn, []))
        self.assertEqual([], conn.executed)


if __name__ == "__main__":
    unittest.main()
