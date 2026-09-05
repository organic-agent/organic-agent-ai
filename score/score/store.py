"""저장소 — `photo_analysis` 중 SCORE 가 읽고 쓰는 부분.

쓰기는 `write_scores` 하나다: subjects · sub_scores · clip_embedding · model_version. 백분위·연사·그룹은
categorize 의 컬럼, embedding·embedding_model 은 embedder 의 컬럼이라 **건드리지 않는다**(UPSERT 의 SET 절이
그 경계다). photo_ratings · photo_selection_items 는 읽지도 않는다(CLAUDE.md).

읽기는 재개 판정용이다 — 같은 MODEL_VERSION 이고 CLIP 벡터가 저장된 사진은 건너뛴다.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
import logging
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class PhotoAnalysis:
    """`photo_analysis` 한 행. 컬럼 이름을 그대로 필드로 쓴다. SCORE 는 subjects · sub_scores · model_version 만 채운다."""

    photo_id: str
    subjects: str = "unknown"
    technical_pct: float = 50.0
    aesthetic_pct: float = 50.0
    sub_scores: dict = field(default_factory=dict)   # technical_score, aesthetic_score, sharpness, clip_parent …
    cluster_id: int = -1
    cluster_rank: int = 0
    embed_group_id: int = -1
    model_version: str = ""
    #: 마지막으로 점수를 쓴 시각(DB timestamptz | 로컬 ISO 문자열). force 재계산의 "이번 실행 전 점수" 판정(#54).
    analyzed_at: object | None = None


class Store(Protocol):
    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]: ...
    def read_clip_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]: ...
    def write_scores(self, gallery: str, rows: list[PhotoAnalysis],
                     clip_embeddings: tuple[list[str], np.ndarray]) -> None: ...


# ── 로컬 구현 ────────────────────────────────────────────────────────────────
class LocalStore:
    """out/v3/<갤러리 slug>/ 아래 파일. categorize 의 LocalStore 와 같은 파일 규약이라 두 CLI 를 이어 돌릴 수 있다."""

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

    def read_clip_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        d = self._dir(gallery)
        ids_p, emb_p = d / "clip_embeddings_ids.json", d / "clip_embeddings.npy"
        if not (ids_p.exists() and emb_p.exists()):
            return [], np.zeros((0, 0))
        return json.loads(ids_p.read_text(encoding="utf-8")), np.load(emb_p)

    def write_scores(self, gallery: str, rows: list[PhotoAnalysis],
                     clip_embeddings: tuple[list[str], np.ndarray]) -> None:
        """SCORE 의 필드만 덮는다 — 기존 행의 백분위·클러스터·그룹은 그대로, 없던 사진은 새 행."""
        by_id = {r.photo_id: r for r in self.read_analysis(gallery)}
        stamp = datetime.now(timezone.utc).isoformat()
        for r in rows:
            r.analyzed_at = stamp
            cur = by_id.get(r.photo_id)
            if cur is None:
                by_id[r.photo_id] = r
                continue
            for f in ("subjects", "sub_scores", "model_version", "analyzed_at"):
                setattr(cur, f, getattr(r, f))
        with (self._dir(gallery) / "analysis.jsonl").open("w", encoding="utf-8") as f:
            for r in by_id.values():
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

        ids, emb = clip_embeddings
        if len(ids):
            prev_ids, prev = self.read_clip_embeddings(gallery)
            merged = dict(zip(prev_ids, prev)) if len(prev_ids) else {}
            merged.update(zip(ids, emb))
            all_ids = list(merged)
            d = self._dir(gallery)
            (d / "clip_embeddings_ids.json").write_text(json.dumps(all_ids, ensure_ascii=False), encoding="utf-8")
            np.save(d / "clip_embeddings.npy", np.stack([merged[i] for i in all_ids]))


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
    """wes 공유 Postgres. id 규약: 읽을 때 str(), 쓸 때 int(). 트랜잭션은 write_scores 가 배치마다 commit."""

    def __init__(self, settings, connection=None) -> None:
        from score import db as db_mod
        self.conn = connection or db_mod.connect(settings)

    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]:
        """재개 판정에 필요한 것만 — photo_id · model_version · analyzed_at."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT a.photo_id, a.model_version, a.analyzed_at FROM photo_analysis a JOIN photos p ON p.id = a.photo_id "
                "WHERE p.gallery_id = %s AND p.deleted_at IS NULL AND a.model_version IS NOT NULL",
                (int(gallery),),
            )
            rows = cur.fetchall()
        return [PhotoAnalysis(photo_id=str(r[0]), model_version=str(r[1]), analyzed_at=r[2]) for r in rows]

    def read_clip_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        """재개 판정용 — 어느 사진에 CLIP 벡터가 있는가. 벡터 값은 categorize 가 읽는다."""
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT a.photo_id FROM photo_analysis a JOIN photos p ON p.id = a.photo_id "
                "WHERE p.gallery_id = %s AND p.deleted_at IS NULL AND a.clip_embedding IS NOT NULL",
                (int(gallery),),
            )
            rows = cur.fetchall()
        ids = [str(r[0]) for r in rows]
        # 값은 필요 없다 — 자리 표시용 빈 벡터. pipeline 은 id 집합만 본다.
        return ids, np.zeros((len(ids), 0), dtype=np.float32)

    def write_scores(self, gallery: str, rows: list[PhotoAnalysis],
                     clip_embeddings: tuple[list[str], np.ndarray]) -> None:
        """SCORE 의 컬럼만 UPSERT — subjects · sub_scores · clip_embedding · model_version.
        백분위·클러스터·그룹은 categorize 의 것, embedding·embedding_model 은 embedder 의 것 — 건드리지 않는다."""
        clip_map = dict(zip(*clip_embeddings)) if clip_embeddings[0] else {}
        params = [(int(r.photo_id), r.subjects, _jsonb(r.sub_scores), clip_map.get(r.photo_id), r.model_version)
                  for r in rows]
        if not params:
            return
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO photo_analysis
                    (photo_id, subjects, sub_scores, clip_embedding, model_version,
                     analyzed_at, created_at, updated_at)
                VALUES (%s, %s, %s::jsonb, %s, %s, now(), now(), now())
                ON CONFLICT (photo_id) DO UPDATE SET
                    subjects = EXCLUDED.subjects, sub_scores = EXCLUDED.sub_scores,
                    clip_embedding = EXCLUDED.clip_embedding, model_version = EXCLUDED.model_version,
                    analyzed_at = now(), updated_at = now(), version = photo_analysis.version + 1
                """,
                params,
            )
        self.conn.commit()
        log.info("photo_analysis 점수 적재: gallery=%s %d행", gallery, len(params))
