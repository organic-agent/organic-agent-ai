"""저장소 — `photo_analysis` 와 `ai_concept_assignments` 중 CATEGORIZE 가 읽고 쓰는 부분.

읽기: `read_gallery` **한 쿼리** — score 의 원점수·subjects·clip_parent(sub_scores)·clip_embedding 과 embedder 의
embedding(DINOv3) 을 한 번에. 그룹화와 naming 이 같은 결과를 나눠 쓰므로 갤러리당 한 번만 읽는다(7천 장이면 벡터 두 종류 44MB).
쓰기: `write_groups`(technical_pct · aesthetic_pct · sub_scores · cluster_id · cluster_rank · embed_group_id — UPDATE,
행은 score 가 만들어 두었다)와 `write_assignments`(ai_concept_assignments, job_id 에 매달림).
subjects · clip_embedding · model_version 은 score 의 것, embedding · embedding_model 은 embedder 의 것 — 건드리지 않는다.
photo_ratings · photo_selection_items 는 읽지도 않는다(CLAUDE.md).
"""

from __future__ import annotations

import json
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

#: 대표 사진 S3 다운로드 동시 수. Lambda 의 네트워크·/tmp 에 부담 없는 크기다.
PREVIEW_DOWNLOAD_WORKERS = 16


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


@dataclass
class GalleryRead:
    """갤러리 한 번 읽기 — score 가 지난 분석 행과 벡터 두 종류. 파이프라인 한 실행에 한 번만 만든다."""

    rows: list[PhotoAnalysis]                 # model_version 있는 행, 화면 순(display_order, id)
    embeddings: dict[str, np.ndarray]         # DINOv3 — 없으면 빈 dict(로컬 데이터셋 모드)
    clip_embeddings: dict[str, np.ndarray]    # CLIP ViT-L/14


# ── 인터페이스 ───────────────────────────────────────────────────────────────
class Store(Protocol):
    def read_gallery(self, gallery: str) -> GalleryRead: ...
    def write_groups(self, gallery: str, rows: list[PhotoAnalysis]) -> None: ...
    def write_assignments(self, gallery: str, job_id: int | None,
                          rows: list[ConceptAssignment]) -> None: ...
    def preview_paths(self, gallery: str, photo_ids: list[str]) -> dict[str, str]:
        """대표 사진들의 로컬 JPEG 경로. 없는 사진은 빠진다 — 호출자가 nearest 배정으로 넘긴다."""
        ...


# ── 로컬 구현 ────────────────────────────────────────────────────────────────
class LocalStore:
    """out/v3/<갤러리 slug>/ 아래 파일 — score 의 LocalStore 와 같은 규약. 로컬은 임베더가 없어 CLIP 이 임베딩 역할을 겸한다."""

    def __init__(self, root: Path, dataset_root: Path | None = None) -> None:
        self.root = Path(root)
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def preview_paths(self, gallery: str, photo_ids: list[str]) -> dict[str, str]:
        if self.dataset_root is None:
            return {}
        found = {pid: self.dataset_root / pid for pid in photo_ids}
        return {pid: str(p) for pid, p in found.items() if p.is_file()}

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

    def read_gallery(self, gallery: str) -> GalleryRead:
        rows = [r for r in self.read_analysis(gallery) if r.model_version]
        emb_ids, E = self.read_embeddings(gallery)
        clip_ids, C = self.read_clip_embeddings(gallery)
        return GalleryRead(rows=rows,
                           embeddings=dict(zip(emb_ids, E)) if len(emb_ids) else {},
                           clip_embeddings=dict(zip(clip_ids, C)) if len(clip_ids) else {})

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
            from categorize.storage import PreviewStorage
            self._storage = PreviewStorage(self._settings.s3_bucket)
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
