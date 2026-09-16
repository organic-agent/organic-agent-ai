"""DB 저장소 — `photo_analysis` 와 `ai_concept_assignments` 중 CATEGORIZE 가 읽고 쓰는 부분.

읽기: `read_gallery` **한 쿼리** — score 의 원점수·subjects·clip_parent(sub_scores)·clip_embedding 과 embedder 의
embedding(DINOv3) 을 한 번에. 그룹화와 naming 이 같은 결과를 나눠 쓰므로 갤러리당 한 번만 읽는다(7천 장이면 벡터 두 종류 44MB).
쓰기: `write_groups`(technical_pct · aesthetic_pct · sub_scores · cluster_id · cluster_rank · embed_group_id — UPDATE,
행은 score 가 만들어 두었다)와 `write_assignments`(ai_concept_assignments, job_id 에 매달림).
subjects · clip_embedding · model_version 은 score 의 것, embedding · embedding_model 은 embedder 의 것 — 건드리지 않는다.
photo_ratings · photo_selection_items 는 읽지도 않는다(CLAUDE.md).

레코드와 `Store` 프로토콜은 domain/analysis.py, 로컬 파일 구현은 repository/local.py.
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from categorize.domain.analysis import ConceptAssignment, GalleryRead, PhotoAnalysis

log = logging.getLogger(__name__)

#: 대표 사진 S3 다운로드 동시 수. Lambda 의 네트워크·/tmp 에 부담 없는 크기다.
PREVIEW_DOWNLOAD_WORKERS = 16


def _jsonb(value) -> str:
    """jsonb 컬럼용 — NaN/Inf 는 JSON 표준에 없어 null 로 보낸다."""
    def clean(v):
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [clean(x) for x in v]
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return json.dumps(clean(value), ensure_ascii=False)


class DbStore:
    """wes 공유 Postgres. id 규약: 읽을 때 str(), 쓸 때 int()."""

    ANALYSIS_COLUMNS = (
        "subjects", "technical_pct", "aesthetic_pct", "sub_scores",
        "cluster_id", "cluster_rank", "embed_group_id", "model_version",
    )

    def __init__(self, settings, connection=None) -> None:
        from categorize.repository import connection as connection_mod
        self.conn = connection or connection_mod.connect(settings)
        self._settings = settings
        self._storage = None

    def preview_paths(self, gallery: str, photo_ids: list[str]) -> dict[str, str]:
        """naming 대표 사진용 — work_dir 에 있으면 그것, 없으면 S3 에서 받는다(갤러리당 수십~수백 장).

        preview_key 는 SELECT 한 번으로 모아 받고, 다운로드는 스레드풀로 겹친다. 사진마다 SELECT + GET 을 직렬로 하면
        대표 240장에 10~25초가 들었다 — 왕복 지연이 곧 시간이라 병렬이 답이다. 실패한 사진은 결과에서 빠질 뿐이다."""
        dest_dir = Path(self._settings.work_dir) / str(gallery)
        out: dict[str, str] = {}
        need: list[str] = []
        for pid in photo_ids:
            dest = dest_dir / f"{pid}.jpg"
            if dest.is_file() and dest.stat().st_size > 0:
                out[pid] = str(dest)
            else:
                need.append(pid)
        if not need or not self._settings.s3_bucket:
            return out
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT id, preview_key FROM photos WHERE id = ANY(%s) AND deleted_at IS NULL AND preview_key IS NOT NULL",
                ([int(pid) for pid in need],),
            )
            keys = {str(photo_id): key for photo_id, key in cur.fetchall()}
        if not keys:
            return out
        if self._storage is None:
            from categorize.repository.storage import PreviewStorage
            self._storage = PreviewStorage(self._settings.s3_bucket, max_concurrency=PREVIEW_DOWNLOAD_WORKERS)
        storage = self._storage

        def fetch(pid: str) -> tuple[str, str | None]:
            try:
                return pid, str(storage.download(keys[pid], dest_dir / f"{pid}.jpg"))
            except Exception as exc:  # noqa: BLE001
                log.warning("미리보기 내려받기 실패 photo=%s: %s", pid, exc)
                return pid, None

        with ThreadPoolExecutor(max_workers=min(PREVIEW_DOWNLOAD_WORKERS, len(keys))) as pool:
            for pid, path in pool.map(fetch, list(keys)):
                if path is not None:
                    out[pid] = path
        return out

    # ── analysis + vectors ──
    def read_gallery(self, gallery: str) -> GalleryRead:
        """분석 행과 벡터 두 종류를 **한 쿼리**로. embedding_model 이 섞여 있으면 실패한다 — 다른 공간의 코사인은 무의미."""
        cols = ", ".join(f"a.{c}" for c in self.ANALYSIS_COLUMNS)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT a.photo_id, {cols}, a.embedding, a.embedding_model, a.clip_embedding "
                "FROM photo_analysis a JOIN photos p ON p.id = a.photo_id "
                "WHERE p.gallery_id = %s AND p.deleted_at IS NULL "
                "ORDER BY p.display_order, p.id",
                (int(gallery),),
            )
            raw = cur.fetchall()
        rows: list[PhotoAnalysis] = []
        embeddings: dict[str, np.ndarray] = {}
        clips: dict[str, np.ndarray] = {}
        models: set[str] = set()
        n_cols = len(self.ANALYSIS_COLUMNS)
        for r in raw:
            photo_id = str(r[0])
            embedding, embedding_model, clip = r[1 + n_cols], r[2 + n_cols], r[3 + n_cols]
            if embedding is not None:
                embeddings[photo_id] = np.asarray(embedding, dtype=np.float32)
                models.add(str(embedding_model))
            if clip is not None:
                clips[photo_id] = np.asarray(clip, dtype=np.float32)
            d = dict(zip(("photo_id",) + self.ANALYSIS_COLUMNS, r[:1 + n_cols]))
            if d["model_version"] is None:
                continue
            d["photo_id"] = photo_id
            d["sub_scores"] = dict(d["sub_scores"] or {})
            rows.append(PhotoAnalysis(**d))
        if len(models) > 1:
            raise RuntimeError(
                f"gallery {gallery}: embedding_model이 섞여 있다 {sorted(models)} — "
                "임베더 force 재실행으로 한 모델로 맞춘 뒤 분석하라")
        return GalleryRead(rows=rows, embeddings=embeddings, clip_embeddings=clips)

    def write_groups(self, gallery: str, rows: list[PhotoAnalysis]) -> None:
        """CATEGORIZE 의 컬럼만 UPDATE — 행은 score 가 만들어 두었다."""
        params = [(float(r.technical_pct), float(r.aesthetic_pct), _jsonb(r.sub_scores),
                   int(r.cluster_id), int(r.cluster_rank), int(r.embed_group_id), int(r.photo_id))
                  for r in rows]
        if not params:
            return
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                UPDATE photo_analysis
                SET technical_pct = %s, aesthetic_pct = %s, sub_scores = %s::jsonb,
                    cluster_id = %s, cluster_rank = %s, embed_group_id = %s,
                    analyzed_at = now(), updated_at = now(), version = version + 1
                WHERE photo_id = %s
                """,
                params,
            )
        self.conn.commit()
        log.info("photo_analysis 그룹 적재: gallery=%s %d행", gallery, len(params))

    # ── assignments ──
    def write_assignments(self, gallery: str, job_id: int | None,
                          rows: list[ConceptAssignment]) -> None:
        """naming 의 배정을 잡에 매달아 INSERT. 같은 잡의 재실행은 UPSERT 로 덮는다.

        잡이 없으면(CLI 확인용 실행) 저장하지 않는다 — ai_concept_assignments 는 job_id 에 매달리고, wes 는 최신 잡의
        배정을 읽는다. 이름은 로그·결과 payload 로만 남는다."""
        if job_id is None:
            log.warning("gallery %s: 잡이 없어 배정 %d그룹을 저장하지 않는다 (--job-id 가 있어야 ai_concept_assignments 에 남는다)",
                        gallery, len(rows))
            return
        params = [
            (
                int(job_id), int(gallery), int(r.embed_group_id), r.parent_name,
                r.proposed_parent, r.concept_name, float(r.confidence), r.clip_parent,
                r.assigned_by, bool(r.needs_review),
            )
            for r in rows
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO ai_concept_assignments
                    (job_id, gallery_id, embed_group_id, parent_name, proposed_parent,
                     concept_name, confidence, clip_parent, assigned_by, needs_review,
                     created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now(), now())
                ON CONFLICT (job_id, embed_group_id) DO UPDATE SET
                    parent_name = EXCLUDED.parent_name, proposed_parent = EXCLUDED.proposed_parent,
                    concept_name = EXCLUDED.concept_name, confidence = EXCLUDED.confidence,
                    clip_parent = EXCLUDED.clip_parent, assigned_by = EXCLUDED.assigned_by,
                    needs_review = EXCLUDED.needs_review,
                    updated_at = now(), version = ai_concept_assignments.version + 1
                """,
                params,
            )
        self.conn.commit()
        log.info("ai_concept_assignments 적재: gallery=%s job=%s %d그룹", gallery, job_id, len(params))
