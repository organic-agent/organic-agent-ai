"""저장소 — 이 모듈이 읽고 쓰는 테이블.

읽기: photos(original_file_name · display_order) ⋈ photo_analysis(embedding · clip_embedding · pct · subjects · cluster · group),
      photo_selection_items(부부의 최종 선택, 읽기 전용), galleries(CLOSED 목록 · shoot_type).
쓰기: `preference_models` 한 행. 테이블은 wes Flyway 소유 — 없으면 None 을 돌려주고 호출자가 로컬 파일에 남긴다.
photo_ratings 는 읽지 않는다.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class GalleryData:
    """갤러리 하나의 학습 재료. 행 순서 = display_order, id. 전부 같은 길이 n."""

    gallery_id: str
    photo_ids: list[str]
    file_names: list[str]
    technical_pct: np.ndarray      # (n,) 0~100
    aesthetic_pct: np.ndarray      # (n,)
    sharpness_pct: np.ndarray      # (n,) sub_scores.sharpness_pct, 없으면 50
    subjects: list[str]            # bride · groom · couple · group · unknown
    cluster_id: np.ndarray         # (n,) int — 연사 클러스터. 음수면 단독
    cluster_rank: np.ndarray       # (n,) int — 0 이 대표
    embed_group_id: np.ndarray     # (n,) int — 컨셉 그룹
    display_order: np.ndarray      # (n,) int
    embedding: np.ndarray          # (n, 768) DINOv3
    clip_embedding: np.ndarray     # (n, 768) CLIP
    embedding_model: str = ""
    model_version: str = ""
    shoot_type: str | None = None

    def __post_init__(self) -> None:
        n = len(self.photo_ids)
        for name in ("file_names", "technical_pct", "aesthetic_pct", "sharpness_pct", "subjects", "cluster_id",
                     "cluster_rank", "embed_group_id", "display_order", "embedding", "clip_embedding"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"GalleryData.{name} 길이 {len(getattr(self, name))} ≠ {n}")

    @property
    def n(self) -> int:
        return len(self.photo_ids)

    def index_of(self, photo_ids: list[str]) -> np.ndarray:
        """photo_id 목록 → 행 인덱스. 없는 id 는 조용히 버린다(삭제된 사진 등)."""
        by_id = {p: i for i, p in enumerate(self.photo_ids)}
        return np.array(sorted({by_id[p] for p in photo_ids if p in by_id}), dtype=int)


@dataclass
class LabeledGallery:
    """학습 단위 — 갤러리 재료 + 그 부부의 최종 선택(photo_id)."""

    data: GalleryData
    selected_ids: list[str] = field(default_factory=list)

    @property
    def positive_idx(self) -> np.ndarray:
        return self.data.index_of(self.selected_ids)


class Store(Protocol):
    def read_gallery(self, gallery_id: str) -> GalleryData: ...
    def read_selected(self, gallery_id: str) -> list[str]: ...
    def list_closed_galleries(self) -> list[str]: ...
    def write_model(self, row: dict) -> int | None: ...


# ── 로컬 ─────────────────────────────────────────────────────────────────────
class LocalStore:
    """out/preference/<gallery>/data.npz + selected.json. DB 에서 한 번 내려받아(export) 두면 터널 없이 돈다."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, gallery_id: str) -> Path:
        d = self.root / "preference" / str(gallery_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def read_gallery(self, gallery_id: str) -> GalleryData:
        p = self._dir(gallery_id) / "data.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} 가 없다 — `python -m preference export --gallery-id {gallery_id}` 먼저")
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return GalleryData(
            gallery_id=str(gallery_id),
            photo_ids=list(z["photo_ids"]), file_names=list(z["file_names"]),
            technical_pct=z["technical_pct"], aesthetic_pct=z["aesthetic_pct"], sharpness_pct=z["sharpness_pct"],
            subjects=list(z["subjects"]), cluster_id=z["cluster_id"], cluster_rank=z["cluster_rank"],
            embed_group_id=z["embed_group_id"], display_order=z["display_order"],
            embedding=z["embedding"], clip_embedding=z["clip_embedding"],
            embedding_model=meta.get("embedding_model", ""), model_version=meta.get("model_version", ""),
            shoot_type=meta.get("shoot_type"),
        )

    def write_gallery(self, gd: GalleryData) -> Path:
        p = self._dir(gd.gallery_id) / "data.npz"
        meta = json.dumps({"embedding_model": gd.embedding_model, "model_version": gd.model_version,
                           "shoot_type": gd.shoot_type})
        np.savez_compressed(
            p, meta=np.array(meta),
            photo_ids=np.array(gd.photo_ids), file_names=np.array(gd.file_names),
            technical_pct=gd.technical_pct, aesthetic_pct=gd.aesthetic_pct, sharpness_pct=gd.sharpness_pct,
            subjects=np.array(gd.subjects), cluster_id=gd.cluster_id, cluster_rank=gd.cluster_rank,
            embed_group_id=gd.embed_group_id, display_order=gd.display_order,
            embedding=gd.embedding.astype(np.float32), clip_embedding=gd.clip_embedding.astype(np.float32),
        )
        return p

    def read_selected(self, gallery_id: str) -> list[str]:
        p = self._dir(gallery_id) / "selected.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def write_selected(self, gallery_id: str, photo_ids: list[str]) -> None:
        (self._dir(gallery_id) / "selected.json").write_text(json.dumps(photo_ids), encoding="utf-8")

    def list_closed_galleries(self) -> list[str]:
        base = self.root / "preference"
        if not base.exists():
            return []
        return sorted(d.name for d in base.iterdir() if (d / "data.npz").exists() and (d / "selected.json").exists())

    def write_model(self, row: dict) -> int | None:
        """로컬은 항상 파일 — models/<n>.json."""
        d = self.root / "preference" / "models"
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("*.json"))) + 1
        (d / f"{n:04d}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        return n


# ── DB ───────────────────────────────────────────────────────────────────────
_GALLERY_SQL = """
SELECT p.id, p.original_file_name, p.display_order,
       a.technical_pct, a.aesthetic_pct, a.sub_scores, a.subjects,
       a.cluster_id, a.cluster_rank, a.embed_group_id,
       a.embedding, a.clip_embedding, a.embedding_model, a.model_version
FROM photos p
JOIN photo_analysis a ON a.photo_id = p.id
WHERE p.gallery_id = %s AND p.deleted_at IS NULL
  AND a.model_version IS NOT NULL AND a.embedding IS NOT NULL AND a.clip_embedding IS NOT NULL
ORDER BY p.display_order, p.id
"""

_SELECTED_SQL = """
SELECT i.photo_id
FROM photo_selection_items i
JOIN photo_selections s ON s.id = i.selection_id
WHERE s.gallery_id = %s AND s.deleted_at IS NULL
"""


class DbStore:
    """wes 공유 Postgres. id 규약: 읽을 때 str(), 쓸 때 int()."""

    def __init__(self, settings, connection=None) -> None:
        from preference import db as db_mod
        self.conn = connection or db_mod.connect(settings)

    def _has_column(self, table: str, column: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT 1 FROM information_schema.columns WHERE table_name = %s AND column_name = %s",
                        (table, column))
            return cur.fetchone() is not None

    def _has_table(self, table: str) -> bool:
        with self.conn.cursor() as cur:
            cur.execute("SELECT to_regclass(%s)", (table,))
            return cur.fetchone()[0] is not None

    def read_gallery(self, gallery_id: str) -> GalleryData:
        with self.conn.cursor() as cur:
            cur.execute("SELECT shoot_type FROM galleries WHERE id = %s", (int(gallery_id),))
            g = cur.fetchone()
            shoot_type = g[0] if g else None
            cur.execute(_GALLERY_SQL, (int(gallery_id),))
            rows = cur.fetchall()
        if not rows:
            raise ValueError(f"갤러리 {gallery_id}: 분석이 끝난 사진이 없다 (score 가 먼저 돌아야 한다)")
        sub = [r[5] or {} for r in rows]
        versions = {(r[12], r[13]) for r in rows}
        if len(versions) > 1:
            log.warning("갤러리 %s 에 모델 버전이 섞여 있다: %s — 다수를 쓴다", gallery_id, versions)
        emb_model, model_version = max(versions, key=lambda v: sum(1 for r in rows if (r[12], r[13]) == v))
        return GalleryData(
            gallery_id=str(gallery_id),
            photo_ids=[str(r[0]) for r in rows],
            file_names=[r[1] for r in rows],
            technical_pct=np.array([r[3] for r in rows], dtype=float),
            aesthetic_pct=np.array([r[4] for r in rows], dtype=float),
            sharpness_pct=np.array([float(s.get("sharpness_pct", 50.0) or 50.0) for s in sub], dtype=float),
            subjects=[r[6] or "unknown" for r in rows],
            cluster_id=np.array([r[7] if r[7] is not None else -1 for r in rows], dtype=int),
            cluster_rank=np.array([r[8] if r[8] is not None else 0 for r in rows], dtype=int),
            embed_group_id=np.array([r[9] if r[9] is not None else -1 for r in rows], dtype=int),
            display_order=np.array([r[2] for r in rows], dtype=int),
            embedding=np.stack([np.asarray(r[10], dtype=np.float32) for r in rows]),
            clip_embedding=np.stack([np.asarray(r[11], dtype=np.float32) for r in rows]),
            embedding_model=emb_model or "", model_version=model_version or "", shoot_type=shoot_type,
        )

    def read_selected(self, gallery_id: str) -> list[str]:
        """부부의 최종 선택. `source` 컬럼이 생기면(P3) MANUAL 만 — AI 초안이 학습에 되먹임되면 안 된다."""
        sql = _SELECTED_SQL
        if self._has_column("photo_selection_items", "source"):
            sql += " AND i.source = 'MANUAL'"
        with self.conn.cursor() as cur:
            cur.execute(sql, (int(gallery_id),))
            return [str(r[0]) for r in cur.fetchall()]

    def list_closed_galleries(self) -> list[str]:
        """CLOSED 이고 선택이 1장 이상 있고 분석이 끝난 갤러리 — 마감 순."""
        with self.conn.cursor() as cur:
            cur.execute("""
                SELECT g.id FROM galleries g
                WHERE g.status = 'CLOSED'
                  AND EXISTS (SELECT 1 FROM photo_selections s JOIN photo_selection_items i ON i.selection_id = s.id
                              WHERE s.gallery_id = g.id AND s.deleted_at IS NULL)
                  AND EXISTS (SELECT 1 FROM photos p JOIN photo_analysis a ON a.photo_id = p.id
                              WHERE p.gallery_id = g.id AND p.deleted_at IS NULL AND a.model_version IS NOT NULL)
                ORDER BY g.updated_at, g.id
            """)
            return [str(r[0]) for r in cur.fetchall()]

    def write_model(self, row: dict) -> int | None:
        """`preference_models` 한 행. 테이블이 없으면 None — 호출자가 로컬 파일에 남긴다. active 는 부분 유니크라
        켤 때 이전 active 를 같은 트랜잭션에서 내린다."""
        if not self._has_table("preference_models"):
            log.warning("preference_models 테이블이 없다 (wes Flyway 미적용) — DB 에 쓰지 않는다")
            return None
        with self.conn.transaction(), self.conn.cursor() as cur:
            if row["active"]:
                cur.execute("UPDATE preference_models SET active = false WHERE active")
            cur.execute(
                """INSERT INTO preference_models
                   (embedding_model, model_version, feature_spec, w_scalar, w_emb, bias, lambda,
                    n_galleries, n_positives, train_gallery_ids, holdout, active)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s) RETURNING id""",
                (row["embedding_model"], row["model_version"], row["feature_spec"], list(row["w_scalar"]),
                 np.asarray(row["w_emb"], dtype=np.float32), row["bias"], row["lambda"], row["n_galleries"],
                 row["n_positives"], [int(g) for g in row["train_gallery_ids"]],
                 json.dumps(row["holdout"], ensure_ascii=False), bool(row["active"])),
            )
            return int(cur.fetchone()[0])
