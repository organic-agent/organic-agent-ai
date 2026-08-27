"""저장소 — 파이프라인이 읽고 쓰는 유일한 통로.

왜 인터페이스인가: DB 스키마(`photo_analysis`, `ai_recommendations`)는 wes Flyway가 소유하고
아직 협의 중이다(plan.md §5). 그런데 파이프라인은 지금 실제 데이터로 돌려 봐야 한다.
그래서 분석·추천 코드는 `Store`만 보고, 구현이 둘이다.

    LocalStore   지금.  out/<갤러리>/ 아래 JSONL·npy.  데이터셋 폴더로 테스트한다.
    DbStore      나중.  RDS. 컬럼 매핑은 아래 주석에 미리 적어 둔다.

접근 규칙(CLAUDE.md)은 인터페이스 모양으로 강제한다 —
  · `photo_selection_items`·`photo_ratings`·`pair_comparison_events`는 **읽기 메서드만** 있다
  · `photo_selections.status`를 만지는 메서드는 아예 없다
  · 쓰기는 `write_analysis`와 `write_recommendations` 둘뿐이다
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np


# ── 레코드 ───────────────────────────────────────────────────────────────────
@dataclass
class PhotoAnalysis:
    """`photo_analysis` 한 행. 컬럼 이름을 그대로 필드로 쓴다."""

    photo_id: str
    scene: str = "unknown"
    framing: str = "full"
    lighting: str = "natural"
    expression: str = "none"
    subjects: str = "none"
    caption: str = ""
    technical_pct: float = 50.0
    aesthetic_pct: float = 50.0
    face_boxes: dict = field(default_factory=dict)       # face_count, max_face_ratio
    sub_scores: dict = field(default_factory=dict)       # 원점수: technical_score, aesthetic_score, eyes_open, smile, …
    cluster_id: int = -1
    cluster_rank: int = 0
    rank_reason_code: str = ""
    model_version: str = ""

    def tags(self) -> dict[str, str]:
        return {
            "scene": self.scene, "framing": self.framing, "lighting": self.lighting,
            "expression": self.expression, "subjects": self.subjects,
        }


@dataclass
class Recommendation:
    """`ai_recommendations` 한 행."""

    photo_id: str
    round: int
    rank: int
    score_breakdown: dict
    reason: str


@dataclass
class Evidence:
    """취향 신호 묶음. DB에서는 세 테이블을 읽어 만들고, 로컬에서는 JSON 파일 하나다.

    selected  — `photo_selection_items`(MANUAL)의 photo_id들. **읽기 전용**
    ratings   — `photo_ratings`: photo_id → 1~5. **읽기 전용**
    pairs     — `pair_comparison_events`: (chosen, rejected, axis)
    feedback  — 자연어 피드백 (ai_recommendations 반응 또는 별도 이벤트)
    """

    selected: list[str] = field(default_factory=list)
    ratings: dict[str, int] = field(default_factory=dict)
    pairs: list[tuple[str, str, str]] = field(default_factory=list)
    #: 자연어 피드백 문장들 ("가족 사진 더"). LLM이 {axis, tag, delta}로 번역한다. 선택은 못 바꾼다.
    feedback: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.selected or self.ratings or self.pairs or self.feedback)


# ── 인터페이스 ───────────────────────────────────────────────────────────────
class Store(Protocol):
    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]: ...
    def read_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]: ...
    def write_analysis(self, gallery: str, rows: list[PhotoAnalysis],
                       embeddings: tuple[list[str], np.ndarray]) -> None: ...
    def read_evidence(self, gallery: str, selection_id: str | None) -> Evidence: ...
    def read_recommendations(self, gallery: str) -> list[Recommendation]: ...
    def write_recommendations(self, gallery: str, rows: list[Recommendation]) -> None: ...


# ── 로컬 구현 ────────────────────────────────────────────────────────────────
class LocalStore:
    """out/<갤러리 slug>/ 아래에 파일로 둔다.

        analysis.jsonl          photo_analysis 행들
        embeddings.npy + ids    임베딩 (photo_id 순서는 embedding_ids.json)
        evidence.json           취향 신호 (review 페이지가 내보내거나 손으로 만든다)
        recommendations.jsonl   초안 (round별 누적)
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, gallery: str) -> Path:
        d = self.root / gallery.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        return d

    # analysis
    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]:
        p = self._dir(gallery) / "analysis.jsonl"
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as f:
            return [PhotoAnalysis(**json.loads(line)) for line in f if line.strip()]

    def read_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        d = self._dir(gallery)
        ids_p, emb_p = d / "embedding_ids.json", d / "embeddings.npy"
        if not (ids_p.exists() and emb_p.exists()):
            return [], np.zeros((0, 0))
        return json.loads(ids_p.read_text(encoding="utf-8")), np.load(emb_p)

    def write_analysis(self, gallery: str, rows: list[PhotoAnalysis],
                       embeddings: tuple[list[str], np.ndarray]) -> None:
        d = self._dir(gallery)
        with (d / "analysis.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
        ids, emb = embeddings
        (d / "embedding_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        np.save(d / "embeddings.npy", emb)

    # evidence (읽기 전용)
    def read_evidence(self, gallery: str, selection_id: str | None = None) -> Evidence:
        p = self._dir(gallery) / (f"evidence-{selection_id}.json" if selection_id else "evidence.json")
        if not p.exists():
            return Evidence()
        raw = json.loads(p.read_text(encoding="utf-8"))
        return Evidence(
            selected=list(raw.get("selected", [])),
            ratings={k: int(v) for k, v in raw.get("ratings", {}).items()},
            pairs=[tuple(x) for x in raw.get("pairs", [])],
            feedback=[str(x) for x in raw.get("feedback", []) if str(x).strip()],
        )

    # recommendations
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


# ── DB 구현 자리 ─────────────────────────────────────────────────────────────
class DbStore:
    """RDS 구현. wes V22가 머지되면 embedder/db.py의 connect()를 그대로 가져와 채운다.

    매핑 (plan.md §5):
      read_analysis        SELECT … FROM photo_analysis WHERE photo_id IN (SELECT id FROM photos WHERE gallery_id=?)
      read_embeddings      SELECT id, embedding FROM photos WHERE gallery_id=?        (embedder가 채운 DINOv2)
      write_analysis       INSERT … ON CONFLICT (photo_id) DO UPDATE                  (멱등)
      read_evidence        photo_selection_items(MANUAL) + photo_ratings + pair_comparison_events — 전부 SELECT만
      write_recommendations INSERT INTO ai_recommendations (selection_id, photo_id, round, rank, score_breakdown, reason)

    금지: photo_selections.status UPDATE, photo_selection_items INSERT/UPDATE, photo_ratings INSERT/UPDATE.
    """

    def __init__(self, settings) -> None:
        raise NotImplementedError("wes V22 스키마 머지 후 구현 — 그전까지는 LocalStore를 쓴다")
