"""v3 저장소 — V45 스키마 (`photo_analysis` 신판 + `ai_concept_assignments`).

v2 store의 복제·개편. V45에서 바뀐 계약:
  · photo_analysis에 scene/framing/lighting/expression/caption/rank_reason_code 컬럼이 없다 —
    rank_reason은 sub_scores jsonb로 들어간다
  · clip_embedding·embed_group_id를 분석 배치(여기)가 쓴다. embedding(DINOv3)·embedding_model은
    임베더 것이라 여전히 건드리지 않는다
  · naming 잡의 산출물은 ai_concept_assignments (job_id에 매달림)

접근 규칙(CLAUDE.md)은 그대로 인터페이스 모양으로 강제한다 — 쓰기는 `write_analysis`와
`write_assignments` 둘뿐이고, photo_ratings는 읽지도 않는다.
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
    """`photo_analysis` 한 행 (V45). 컬럼 이름을 그대로 필드로 쓴다."""

    photo_id: str
    subjects: str = "unknown"
    technical_pct: float = 50.0
    aesthetic_pct: float = 50.0
    face_boxes: dict = field(default_factory=dict)
    sub_scores: dict = field(default_factory=dict)   # technical_score, aesthetic_score, sharpness, rank_reason …
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


class FolderSetMissing(RuntimeError):
    """AI 폴더 세트가 없다 — wes 는 409(FOLDERS_NOT_READY)로 변환한다. 컨셉 폴백은 없다(§2.1)."""


@dataclass
class Folder:
    """추천 단위인 **자식 폴더** 하나 — wes `photo_folders` 또는 로컬 배정 파생.

    photo_ids 는 **현재 소속** 기준이다(§7 — 사용자가 옮긴 뒤 상태). DB 는 photo_folder_items,
    로컬은 assignments 의 embed_group_id 매핑에서 온다.
    """

    folder_id: str                  # DB: photo_folders.id (str), 로컬: "부모›컨셉"
    parent_name: str
    name: str
    photo_ids: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    """`ai_recommendations` 한 행 (V46 예정 계약 — §2.5). reason 은 2단계라 None 허용."""

    photo_id: str
    round: int
    rank: int                       # 폴더 안 순위
    score_breakdown: dict
    reason: str | None = None
    folder_id: str | None = None    # 미분류 가상 폴더는 None


@dataclass
class Evidence:
    """추천이 읽는 상태 신호. selected 는 photo_selection_items(읽기 전용), rejected 는
    ai_recommendations.rejected_at. photo_ratings 는 접근 금지(CLAUDE.md)."""

    selected: list[str] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)


# ── 인터페이스 ───────────────────────────────────────────────────────────────
class Store(Protocol):
    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]: ...
    def read_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]: ...
    def read_clip_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]: ...
    def write_analysis(self, gallery: str, rows: list[PhotoAnalysis],
                       embeddings: tuple[list[str], np.ndarray],
                       clip_embeddings: tuple[list[str], np.ndarray]) -> None: ...
    def write_assignments(self, gallery: str, job_id: int | None,
                          rows: list[ConceptAssignment]) -> None: ...
    def shoot_type(self, gallery: str) -> str: ...
    def preview_path(self, gallery: str, photo_id: str) -> str | None: ...
    def read_folder_set(self, gallery: str, analysis_job_id: int | None) -> list[Folder]: ...
    def read_evidence(self, gallery: str, selection_id: str | None) -> Evidence: ...
    def read_recommendations(self, gallery: str) -> list[Recommendation]: ...
    def write_recommendations(self, gallery: str, rows: list[Recommendation]) -> None: ...
    def update_reasons(self, gallery: str, round_no: int, reasons: dict[str, str]) -> None: ...


# ── 로컬 구현 ────────────────────────────────────────────────────────────────
class LocalStore:
    """out/v3/<갤러리 slug>/ 아래 파일. 로컬은 임베더가 없어 CLIP이 임베딩 역할을 겸한다."""

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

    def shoot_type(self, gallery: str) -> str:
        return "REHEARSAL"

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

    def write_analysis(self, gallery: str, rows: list[PhotoAnalysis],
                       embeddings: tuple[list[str], np.ndarray],
                       clip_embeddings: tuple[list[str], np.ndarray]) -> None:
        d = self._dir(gallery)
        with (d / "analysis.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        for name, (ids, emb) in (("embeddings", embeddings), ("clip_embeddings", clip_embeddings)):
            (d / f"{name}_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
            np.save(d / f"{name}.npy", emb)

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

    # ── 추천 (plan-v3-folder-compare.md §2) ──
    def read_folder_set(self, gallery: str, analysis_job_id: int | None = None) -> list[Folder]:
        """로컬 폴더 세트 — naming 배정에서 파생한다. wes 플래너처럼 같은 (부모, 컨셉) 이름의
        임베딩 그룹들을 폴더 하나로 합친다. analysis_job_id 는 로컬에서 무시(세트가 하나뿐)."""
        assignments = self.read_assignments(gallery)
        if not assignments:
            raise FolderSetMissing(f"갤러리 {gallery}: AI 폴더 세트가 없다 — naming 이 먼저다")
        by_group = {a.embed_group_id: a for a in assignments}
        folders: dict[str, Folder] = {}
        for r in self.read_analysis(gallery):
            a = by_group.get(r.embed_group_id)
            if a is None:
                continue
            fid = f"{a.parent_name}›{a.concept_name}"
            f = folders.setdefault(fid, Folder(folder_id=fid, parent_name=a.parent_name,
                                               name=a.concept_name))
            f.photo_ids.append(r.photo_id)
        return sorted(folders.values(), key=lambda f: (-len(f.photo_ids), f.folder_id))

    def target_count(self, gallery: str) -> int | None:
        return None

    def read_evidence(self, gallery: str, selection_id: str | None = None) -> Evidence:
        p = self._dir(gallery) / (f"evidence-{selection_id}.json" if selection_id else "evidence.json")
        if not p.exists():
            return Evidence()
        raw = json.loads(p.read_text(encoding="utf-8"))
        return Evidence(selected=[str(x) for x in raw.get("selected", [])],
                        rejected=[str(x) for x in raw.get("rejected", [])])

    def read_recommendations(self, gallery: str) -> list[Recommendation]:
        p = self._dir(gallery) / "recommendations.jsonl"
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as f:
            return [Recommendation(**json.loads(line)) for line in f if line.strip()]

    def write_recommendations(self, gallery: str, rows: list[Recommendation]) -> None:
        p = self._dir(gallery) / "recommendations.jsonl"
        existing = [r for r in self.read_recommendations(gallery) if not rows or r.round != rows[0].round]
        with p.open("w", encoding="utf-8") as f:
            for r in existing + rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def update_reasons(self, gallery: str, round_no: int, reasons: dict[str, str]) -> None:
        rows = self.read_recommendations(gallery)
        for r in rows:
            if r.round == round_no and r.photo_id in reasons:
                r.reason = reasons[r.photo_id]
        p = self._dir(gallery) / "recommendations.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def _jsonb(value) -> str:
    """jsonb 컬럼용 — NaN/Inf는 JSON 표준에 없어 null로 보낸다."""
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
    """wes 공유 Postgres (V45). id 규약은 v2와 같다 — 읽을 때 str(), 쓸 때 int()."""

    ANALYSIS_COLUMNS = (
        "subjects", "technical_pct", "aesthetic_pct", "face_boxes", "sub_scores",
        "cluster_id", "cluster_rank", "embed_group_id", "model_version",
    )

    def __init__(self, settings, selection_id: str | None = None, connection=None) -> None:
        from photoselect import db as db_mod
        self.conn = connection or db_mod.connect(settings)
        self.selection_id = int(selection_id) if selection_id is not None else None
        self._settings = settings
        self._storage = None

    def preview_path(self, gallery: str, photo_id: str) -> str | None:
        """analyze가 내려받은 미리보기가 work_dir에 있으면 그것, 없으면 S3에서 다시 받는다."""
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
            from photoselect.storage import PreviewStorage
            self._storage = PreviewStorage(self._settings.s3_bucket)
        try:
            return str(self._storage.download(row[0], dest))
        except Exception as exc:  # noqa: BLE001
            log.warning("미리보기 내려받기 실패 photo=%s: %s", photo_id, exc)
            return None

    # 워커 배관 호환 (run_selection_job이 v3로 들어와도 명확한 에러까지는 도달하게)
    def gallery_of_selection(self, selection_id: str | int) -> str:
        with self.conn.cursor() as cur:
            cur.execute("SELECT gallery_id FROM photo_selections WHERE id = %s AND deleted_at IS NULL",
                        (int(selection_id),))
            row = cur.fetchone()
        if row is None:
            raise SystemExit(f"셀렉이 없다: photo_selections.id={selection_id}")
        return str(row[0])

    def target_count(self, gallery: str) -> int | None:
        with self.conn.cursor() as cur:
            cur.execute("SELECT max_selectable_photo_count FROM galleries WHERE id = %s", (int(gallery),))
            row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def shoot_type(self, gallery: str) -> str:
        with self.conn.cursor() as cur:
            cur.execute("SELECT shoot_type FROM galleries WHERE id = %s", (int(gallery),))
            row = cur.fetchone()
        if row is None:
            raise SystemExit(f"갤러리가 없다: galleries.id={gallery}")
        return str(row[0])

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
            d["face_boxes"] = d["face_boxes"] or {}
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
        """임베더의 DINOv3. embedding_model이 섞여 있으면 실패한다 — 다른 공간의 코사인은 무의미."""
        return self._read_vectors(gallery, "embedding", "embedding_model")

    def read_clip_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        """이 배치가 저장한 CLIP ViT-L/14. naming 잡이 재분석 없이 다시 돌기 위한 저장분."""
        return self._read_vectors(gallery, "clip_embedding", None)

    def write_analysis(self, gallery: str, rows: list[PhotoAnalysis],
                       embeddings: tuple[list[str], np.ndarray],
                       clip_embeddings: tuple[list[str], np.ndarray]) -> None:
        """분석 컬럼 + clip_embedding UPSERT. embedding·embedding_model은 임베더 것 — 건드리지 않는다.

        `embeddings` 인자(임베더 벡터)는 DB 모드에서 무시한다 — 원본이 이미 DB에 있다.
        """
        clip_map = dict(zip(*clip_embeddings)) if clip_embeddings[0] else {}
        params = [
            (
                int(r.photo_id), r.subjects, float(r.technical_pct), float(r.aesthetic_pct),
                _jsonb(r.face_boxes), _jsonb(r.sub_scores),
                int(r.cluster_id), int(r.cluster_rank), int(r.embed_group_id),
                clip_map.get(r.photo_id), r.model_version,
            )
            for r in rows
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO photo_analysis
                    (photo_id, subjects, technical_pct, aesthetic_pct, face_boxes, sub_scores,
                     cluster_id, cluster_rank, embed_group_id, clip_embedding, model_version,
                     analyzed_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s,
                        now(), now(), now())
                ON CONFLICT (photo_id) DO UPDATE SET
                    subjects = EXCLUDED.subjects,
                    technical_pct = EXCLUDED.technical_pct, aesthetic_pct = EXCLUDED.aesthetic_pct,
                    face_boxes = EXCLUDED.face_boxes, sub_scores = EXCLUDED.sub_scores,
                    cluster_id = EXCLUDED.cluster_id, cluster_rank = EXCLUDED.cluster_rank,
                    embed_group_id = EXCLUDED.embed_group_id, clip_embedding = EXCLUDED.clip_embedding,
                    model_version = EXCLUDED.model_version,
                    analyzed_at = now(), updated_at = now(), version = photo_analysis.version + 1
                """,
                params,
            )
        self.conn.commit()
        log.info("photo_analysis 적재(V45): gallery=%s %d행", gallery, len(params))

    # ── assignments ──
    def write_assignments(self, gallery: str, job_id: int | None,
                          rows: list[ConceptAssignment]) -> None:
        """naming 잡의 배정을 잡에 매달아 INSERT. 같은 잡의 재실행은 UPSERT로 덮는다."""
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

    # ── 추천 (plan-v3-folder-compare.md §2 — wes V46 계약) ──
    #: ai_selection_jobs·ai_recommendations 는 추천 재설계 마이그레이션(V46 예정)에서 온다.
    #: 이 코드는 §2.5 계약(folder_id·UNIQUE(selection, round, photo))을 전제한다.
    _V46_HINT = ("wes 에 추천 테이블이 아직 없다 — V46 마이그레이션"
                 "(ai_selection_jobs 재생성 + ai_recommendations.folder_id) 뒤에 돌려라")

    def _v46(self, exc: Exception) -> RuntimeError:
        self.conn.rollback()
        return RuntimeError(f"{self._V46_HINT} ({type(exc).__name__})")

    def read_folder_set(self, gallery: str, analysis_job_id: int | None = None) -> list[Folder]:
        """최신(또는 지정) AI 세트의 자식 폴더들 — **현재 photo_folder_items 기준**(§7).
        photo_folder_* 는 wes 소유라 읽기 전용이다. 세트가 없으면 FolderSetMissing(→ 409)."""
        with self.conn.cursor() as cur:
            if analysis_job_id is None:
                cur.execute(
                    "SELECT max(analysis_job_id) FROM photo_folder_groups "
                    "WHERE gallery_id = %s AND origin = 'AI' AND analysis_job_id IS NOT NULL",
                    (int(gallery),),
                )
                row = cur.fetchone()
                analysis_job_id = int(row[0]) if row and row[0] is not None else None
            if analysis_job_id is None:
                raise FolderSetMissing(f"갤러리 {gallery}: AI 폴더 세트가 없다 (FOLDERS_NOT_READY)")
            cur.execute(
                """
                SELECT f.id, g.name, f.name, i.photo_id
                FROM photo_folder_groups g
                JOIN photo_folders f ON f.group_id = g.id
                LEFT JOIN photo_folder_items i ON i.folder_id = f.id
                WHERE g.gallery_id = %s AND g.origin = 'AI' AND g.analysis_job_id = %s
                ORDER BY f.id, i.sort_order, i.id
                """,
                (int(gallery), int(analysis_job_id)),
            )
            rows = cur.fetchall()
        if not rows:
            raise FolderSetMissing(f"갤러리 {gallery}: AI 세트 {analysis_job_id} 에 폴더가 없다")
        folders: dict[str, Folder] = {}
        for fid, parent, name, photo_id in rows:
            f = folders.setdefault(str(fid), Folder(folder_id=str(fid), parent_name=str(parent),
                                                    name=str(name)))
            if photo_id is not None:
                f.photo_ids.append(str(photo_id))
        return sorted(folders.values(), key=lambda f: (-len(f.photo_ids), int(f.folder_id)))

    def _require_selection(self) -> int:
        if self.selection_id is None:
            raise SystemExit("DB 모드의 추천은 셀렉 단위다 — --selection-id 가 필요하다")
        return self.selection_id

    def read_evidence(self, gallery: str, selection_id: str | None = None) -> Evidence:
        import psycopg

        sid = int(selection_id) if selection_id is not None else self.selection_id
        if sid is None:
            return Evidence()
        with self.conn.cursor() as cur:
            cur.execute("SELECT photo_id FROM photo_selection_items WHERE selection_id = %s", (sid,))
            selected = [str(r[0]) for r in cur.fetchall()]
            try:
                cur.execute(
                    "SELECT photo_id FROM ai_recommendations WHERE selection_id = %s AND rejected_at IS NOT NULL",
                    (sid,),
                )
                rejected = [str(r[0]) for r in cur.fetchall()]
            except psycopg.errors.UndefinedTable:
                self.conn.rollback()   # V46 전 — 거절 신호가 아직 없다
                rejected = []
        return Evidence(selected=selected, rejected=rejected)

    def read_recommendations(self, gallery: str) -> list[Recommendation]:
        import psycopg

        sid = self._require_selection()
        try:
            with self.conn.cursor() as cur:
                cur.execute(
                    "SELECT photo_id, round, rank, score_breakdown, reason, folder_id "
                    "FROM ai_recommendations WHERE selection_id = %s ORDER BY round, folder_id, rank",
                    (sid,),
                )
                rows = cur.fetchall()
        except psycopg.errors.UndefinedTable as exc:
            raise self._v46(exc) from exc
        return [Recommendation(photo_id=str(p), round=int(rd), rank=int(rk), score_breakdown=bd or {},
                               reason=reason, folder_id=str(fid) if fid is not None else None)
                for p, rd, rk, bd, reason, fid in rows]

    def write_recommendations(self, gallery: str, rows: list[Recommendation]) -> None:
        """한 라운드를 INSERT (reason 은 2단계라 보통 NULL). UNIQUE(selection, round, photo) 전제."""
        import psycopg

        sid = self._require_selection()
        params = [
            (sid, int(r.photo_id), int(r.round), int(r.rank),
             int(r.folder_id) if r.folder_id is not None else None,
             _jsonb(r.score_breakdown), r.reason)
            for r in rows
        ]
        try:
            with self.conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO ai_recommendations
                        (selection_id, photo_id, round, rank, folder_id, score_breakdown, reason,
                         presented_at, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, now(), now(), now())
                    """,
                    params,
                )
            self.conn.commit()
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as exc:
            raise self._v46(exc) from exc
        log.info("ai_recommendations 적재: selection=%s round=%s %d장", sid,
                 rows[0].round if rows else "-", len(params))

    def update_reasons(self, gallery: str, round_no: int, reasons: dict[str, str]) -> None:
        import psycopg

        sid = self._require_selection()
        try:
            with self.conn.cursor() as cur:
                cur.executemany(
                    "UPDATE ai_recommendations SET reason = %s, updated_at = now(), "
                    "version = version + 1 WHERE selection_id = %s AND round = %s AND photo_id = %s",
                    [(text, sid, int(round_no), int(pid)) for pid, text in reasons.items()],
                )
            self.conn.commit()
        except (psycopg.errors.UndefinedTable, psycopg.errors.UndefinedColumn) as exc:
            raise self._v46(exc) from exc
        log.info("추천 이유 UPDATE: selection=%s round=%s %d장", sid, round_no, len(reasons))
