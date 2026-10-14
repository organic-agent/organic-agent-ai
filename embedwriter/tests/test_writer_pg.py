"""실제 PostgreSQL + pgvector 에서 writer 한 문장을 embedder 역할 권한으로 돌린다(선택).

`EMBEDWRITER_PG_DSN` (슈퍼유저 DSN, 예: postgresql://postgres:pw@127.0.0.1:55439/postgres) 이 있을 때만 돈다.
`pgvector/pgvector:pg16` 컨테이너 하나면 된다. 스키마는 wes V1 의 관련 컬럼만, 권한은 V15 EMBEDDER_GRANT_CONTRACT 그대로다 —
그 밖의 컬럼을 건드리면 permission denied 로 깨진다.
"""

from __future__ import annotations

import os
import unittest

from helpers import body, photo

from embedwriter import message, writer

DSN = os.environ.get("EMBEDWRITER_PG_DSN")

SCHEMA = """
DROP TABLE IF EXISTS photo_analysis, photos, galleries;
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'embedder') THEN CREATE ROLE embedder LOGIN PASSWORD 'embedder'; END IF;
END $$;
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE galleries (id bigint PRIMARY KEY, deleted_at timestamptz);
CREATE TABLE photos (
    id bigint PRIMARY KEY, gallery_id bigint NOT NULL REFERENCES galleries(id), storage_key varchar(500) NOT NULL,
    status varchar(20) NOT NULL DEFAULT 'UPLOADED', created_at timestamptz, updated_at timestamptz,
    preview_key varchar(500), taken_at timestamp(6) without time zone, camera_make varchar(100), camera_model varchar(100),
    exposure_time varchar(30), f_number double precision, iso integer, width integer, height integer, byte_size bigint,
    deleted_at timestamptz, version bigint NOT NULL DEFAULT 0, dispatched_at timestamptz
);
CREATE TABLE photo_analysis (
    photo_id bigint PRIMARY KEY REFERENCES photos(id), embedding vector(4), embedding_model varchar(60),
    clip_embedding vector(4), version bigint NOT NULL DEFAULT 0, created_at timestamptz, updated_at timestamptz
);
-- V15 EMBEDDER_GRANT_CONTRACT (photos · galleries · photo_analysis 부분)
REVOKE ALL ON public.photos FROM embedder;
GRANT SELECT (id, deleted_at) ON public.galleries TO embedder;
GRANT SELECT (id, gallery_id, storage_key, status, deleted_at, preview_key, dispatched_at,
              taken_at, camera_make, camera_model, exposure_time, f_number, iso, width, height,
              byte_size, version) ON public.photos TO embedder;
GRANT UPDATE (preview_key, dispatched_at, taken_at, camera_make, camera_model,
              exposure_time, f_number, iso, width, height, byte_size, version, updated_at) ON public.photos TO embedder;
GRANT SELECT, INSERT, UPDATE ON public.photo_analysis TO embedder;
INSERT INTO galleries VALUES (7, NULL), (8, now());
INSERT INTO photos (id, gallery_id, storage_key, camera_make, iso) VALUES
    (1, 7, 'galleries/7/1.jpg', 'Old', 100),
    (2, 7, 'galleries/7/2.jpg', 'Old', 100),
    (3, 7, 'galleries/7/replaced.jpg', NULL, NULL),
    (4, 8, 'galleries/8/4.jpg', NULL, NULL);
INSERT INTO photos (id, gallery_id, storage_key, deleted_at) VALUES (5, 7, 'galleries/7/5.jpg', now());
INSERT INTO photo_analysis (photo_id, embedding, embedding_model, clip_embedding, version)
    VALUES (2, '[9,9,9,9]', 'old-model', '[1,1,1,1]', 3);
"""


@unittest.skipUnless(DSN, "EMBEDWRITER_PG_DSN 이 없어 실제 DB 검증을 건너뛴다")
class WriterPostgresTest(unittest.TestCase):
    def setUp(self) -> None:
        import psycopg

        with psycopg.connect(DSN, autocommit=True) as admin:
            admin.execute(SCHEMA)
        self.admin = psycopg.connect(DSN)
        self.conn = psycopg.connect(DSN, user="embedder", password="embedder")

    def tearDown(self) -> None:
        self.conn.close()
        self.admin.close()

    def test_bulk_upsert_with_guards_under_embedder_grants(self) -> None:
        rows = message.parse(body(
            photo(1, values=[1, 2, 3, 4]),
            photo(2, values=[0.1, 0.2, 0.3, 0.4], metadata=None),         # EXIF 실패 → 기존 값 유지
            photo(3, "galleries/7/3.jpg"),                                  # 원본이 바뀐 늦은 결과 → 0행
            photo(4, "galleries/8/4.jpg"),                                  # 지워진 갤러리 → 0행
            photo(5),                                                       # 휴지통 사진 → 0행
            photo(999),                                                     # 없는 사진 → 0행
        ), 4)

        written = writer.write(self.conn, rows)
        self.conn.commit()

        self.assertEqual(2, written)
        cur = self.admin.execute(
            "SELECT photo_id, embedding::text, embedding_model, clip_embedding::text, version, updated_at IS NOT NULL "
            "FROM photo_analysis ORDER BY photo_id")
        self.assertEqual([
            (1, "[1,2,3,4]", "dinov3", None, 0, True),
            (2, "[0.1,0.2,0.3,0.4]", "dinov3", "[1,1,1,1]", 4, True),    # 점수 쪽 컬럼은 그대로, version + 1
        ], cur.fetchall())
        cur = self.admin.execute(
            "SELECT id, preview_key, camera_make, iso, taken_at::text, byte_size, version FROM photos ORDER BY id")
        self.assertEqual([
            (1, "previews/galleries/7/1.jpg", "Canon", 400, "2026-10-01 03:04:05", 12_345_678, 1),
            (2, "previews/galleries/7/2.jpg", "Old", 100, None, None, 1),
            (3, None, None, None, None, None, 0),
            (4, None, None, None, None, None, 0),
            (5, None, None, None, None, None, 0),
        ], cur.fetchall())

    def test_rewrite_is_idempotent_and_all_null_metadata_arrays_bind(self) -> None:
        rows = message.parse(body(photo(1, metadata=None), photo(2, metadata=None)), 4)
        self.assertEqual(2, writer.write(self.conn, rows))
        self.assertEqual(2, writer.write(self.conn, rows))
        self.conn.commit()
        cur = self.admin.execute("SELECT count(*), max(version) FROM photo_analysis")
        self.assertEqual((2, 5), cur.fetchone())


if __name__ == "__main__":
    unittest.main()
