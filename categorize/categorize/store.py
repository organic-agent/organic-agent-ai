"""저장소 — `photo_analysis` 와 `ai_concept_assignments` 중 CATEGORIZE 가 읽고 쓰는 부분.

읽기: score 의 원점수·subjects·clip_parent(sub_scores)·clip_embedding, embedder 의 embedding(DINOv3).
쓰기: `write_groups`(technical_pct · aesthetic_pct · sub_scores · cluster_id · cluster_rank · embed_group_id — UPDATE,
행은 score 가 만들어 두었다)와 `write_assignments`(ai_concept_assignments, job_id 에 매달림).
subjects · clip_embedding · model_version 은 score 의 것, embedding · embedding_model 은 embedder 의 것 — 건드리지 않는다.
photo_ratings · photo_selection_items 는 읽지도 않는다(CLAUDE.md).
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


# ── 레코드 ───────────────────────────────────────────────────────────────────
@dataclass
class PhotoAnalysis:
    """`photo_analysis` 한 행. 컬럼 이름을 그대로 필드로 쓴다."""

    photo_id: str
    subjects: str = "unknown"
    technical_pct: float = 50.0
    aesthetic_pct: float = 50.0
    sub_scores: dict = field(default_factory=dict)   # technical_score, aesthetic_score, sharpness, rank_reason, clip_parent …
    cluster_id: int = -1
    cluster_rank: int = 0
    embed_group_id: int = -1
    model_version: str = ""


@dataclass
class ConceptAssignment:
    """`ai_concept_assignments` 한 행 — 임베딩 그룹 → (큰 분류, 컨셉)."""

    embed_group_id: int
    parent_name: str
    concept_name: str
    confidence: float
    assigned_by: str                     # 'vlm' | 'nearest'
    proposed_parent: str | None = None   # parent_name='기타'일 때 VLM 제안
    clip_parent: str | None = None       # CLIP zero-shot 다수결 (검증)
    needs_review: bool = False


# ── 인터페이스 ───────────────────────────────────────────────────────────────
class Store(Protocol):
    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]: ...
    def read_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]: ...
    def read_clip_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]: ...
    def write_groups(self, gallery: str, rows: list[PhotoAnalysis]) -> None: ...
    def write_assignments(self, gallery: str, job_id: int | None,
                          rows: list[ConceptAssignment]) -> None: ...
    def preview_path(self, gallery: str, photo_id: str) -> str | None: ...


# ── 로컬 구현 ────────────────────────────────────────────────────────────────
class LocalStore:
    """out/v3/<갤러리 slug>/ 아래 파일 — score 의 LocalStore 와 같은 규약. 로컬은 임베더가 없어 CLIP 이 임베딩 역할을 겸한다."""

    def __init__(self, root: Path, dataset_root: Path | None = None) -> None:
        self.root = Path(root)
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def preview_path(self, gallery: str, photo_id: str) -> str | None:
        if self.dataset_root is None:
            return None
        p = self.dataset_root / photo_id
        return str(p) if p.is_file() else None

    def _dir(self, gallery: str) -> Path:
        d = self.root / "v3" / gallery.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]:
        p = self._dir(gallery) / "analysis.jsonl"
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as f:
            return [PhotoAnalysis(**json.loads(line)) for line in f if line.strip()]

    def _read_npy(self, gallery: str, name: str) -> tuple[list[str], np.ndarray]:
        d = self._dir(gallery)
        ids_p, emb_p = d / f"{name}_ids.json", d / f"{name}.npy"
        if not (ids_p.exists() and emb_p.exists()):
            return [], np.zeros((0, 0))
        return json.loads(ids_p.read_text(encoding="utf-8")), np.load(emb_p)

    def read_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        return self._read_npy(gallery, "embeddings")

    def read_clip_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        return self._read_npy(gallery, "clip_embeddings")

    def _write_rows(self, gallery: str, rows: list[PhotoAnalysis]) -> None:
        with (self._dir(gallery) / "analysis.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def _write_npy(self, gallery: str, name: str, ids: list[str], emb: np.ndarray) -> None:
        d = self._dir(gallery)
        (d / f"{name}_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        np.save(d / f"{name}.npy", emb)

    def write_groups(self, gallery: str, rows: list[PhotoAnalysis]) -> None:
        """CATEGORIZE 의 필드만 덮는다 — subjects · model_version 은 score 의 것이라 그대로."""
        by_id = {r.photo_id: r for r in self.read_analysis(gallery)}
        for r in rows:
            cur = by_id.get(r.photo_id)
            if cur is None:
                by_id[r.photo_id] = r
                continue
            for f in ("technical_pct", "aesthetic_pct", "sub_scores", "cluster_id", "cluster_rank", "embed_group_id"):
                setattr(cur, f, getattr(r, f))
        self._write_rows(gallery, list(by_id.values()))

    def write_analysis(self, gallery: str, rows: list[PhotoAnalysis],
                       embeddings: tuple[list[str], np.ndarray],
                       clip_embeddings: tuple[list[str], np.ndarray]) -> None:
        """전부 한 번에 (테스트·합성 데이터용). 임베더 벡터는 여기서만 쓴다."""
        self._write_rows(gallery, rows)
        for name, (ids, emb) in (("embeddings", embeddings), ("clip_embeddings", clip_embeddings)):
            self._write_npy(gallery, name, ids, emb)

    def write_assignments(self, gallery: str, job_id: int | None,
                          rows: list[ConceptAssignment]) -> None:
        p = self._dir(gallery) / "assignments.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"job_id": job_id, **asdict(r)}, ensure_ascii=False) + "\n")

    def read_assignments(self, gallery: str) -> list[ConceptAssignment]:
        p = self._dir(gallery) / "assignments.jsonl"
        if not p.exists():
            return []
        out = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    d.pop("job_id", None)
                    out.append(ConceptAssignment(**d))
        return out


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


# ── DB 구현 ─────────────────────────────────────────────────────────────────
class DbStore:
    """wes 공유 Postgres. id 규약: 읽을 때 str(), 쓸 때 int()."""

    ANALYSIS_COLUMNS = (
        "subjects", "technical_pct", "aesthetic_pct", "sub_scores",
        "cluster_id", "cluster_rank", "embed_group_id", "model_version",
    )

    def __init__(self, settings, connection=None) -> None:
        from categorize import db as db_mod
        self.conn = connection or db_mod.connect(settings)
        self._settings = settings
        self._storage = None

    def preview_path(self, gallery: str, photo_id: str) -> str | None:
        """naming 대표 사진용 — work_dir 에 있으면 그것, 없으면 S3 에서 받는다(갤러리당 수십 장)."""
        dest = Path(self._settings.work_dir) / str(gallery) / f"{photo_id}.jpg"
        if dest.is_file() and dest.stat().st_size > 0:
            return str(dest)
        if not self._settings.s3_bucket:
            return None
        with self.conn.cursor() as cur:
            cur.execute("SELECT preview_key FROM photos WHERE id = %s AND deleted_at IS NULL", (int(photo_id),))
            row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        if self._storage is None:
            from categorize.storage import PreviewStorage
            self._storage = PreviewStorage(self._settings.s3_bucket)
        try:
            return str(self._storage.download(row[0], dest))
        except Exception as exc:  # noqa: BLE001
            log.warning("미리보기 내려받기 실패 photo=%s: %s", photo_id, exc)
            return None

    # ── analysis ──
    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]:
        cols = ", ".join(f"a.{c}" for c in self.ANALYSIS_COLUMNS)
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT a.photo_id, {cols} FROM photo_analysis a JOIN photos p ON p.id = a.photo_id "
                "WHERE p.gallery_id = %s AND p.deleted_at IS NULL AND a.model_version IS NOT NULL "
                "ORDER BY p.display_order, p.id",
                (int(gallery),),
            )
            rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(zip(("photo_id",) + self.ANALYSIS_COLUMNS, r))
            d["photo_id"] = str(d["photo_id"])
            d["sub_scores"] = dict(d["sub_scores"] or {})
            out.append(PhotoAnalysis(**d))
        return out

    def _read_vectors(self, gallery: str, column: str, model_column: str | None) -> tuple[list[str], np.ndarray]:
        model_sel = f", a.{model_column}" if model_column else ""
        with self.conn.cursor() as cur:
            cur.execute(
                f"SELECT a.photo_id, a.{column}{model_sel} FROM photo_analysis a "
                "JOIN photos p ON p.id = a.photo_id "
                f"WHERE p.gallery_id = %s AND p.deleted_at IS NULL AND a.{column} IS NOT NULL "
                "ORDER BY p.display_order, p.id",
                (int(gallery),),
            )
            rows = cur.fetchall()
        if not rows:
            return [], np.zeros((0, 0))
        if model_column:
            models = {r[2] for r in rows}
            if len(models) > 1:
                raise RuntimeError(
                    f"gallery {gallery}: {model_column}이 섞여 있다 {sorted(map(str, models))} — "
                    "임베더 force 재실행으로 한 모델로 맞춘 뒤 분석하라")
        ids = [str(r[0]) for r in rows]
        return ids, np.stack([np.asarray(r[1], dtype=np.float32) for r in rows])

    def read_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        """임베더의 DINOv3. embedding_model 이 섞여 있으면 실패한다 — 다른 공간의 코사인은 무의미."""
        return self._read_vectors(gallery, "embedding", "embedding_model")

    def read_clip_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        """score 가 저장한 CLIP ViT-L/14."""
        return self._read_vectors(gallery, "clip_embedding", None)

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
        """naming 의 배정을 잡에 매달아 INSERT. 같은 잡의 재실행은 UPSERT 로 덮는다."""
        if job_id is None:
            raise SystemExit("DB 모드의 naming 배정은 잡 단위다 — --job-id (ai_analysis_jobs.id) 가 필요하다")
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
