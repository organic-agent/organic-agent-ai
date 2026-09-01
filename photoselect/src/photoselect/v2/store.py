"""v2 저장소 — v1/store.py 의 복제 + concept_id. v2 규격이 바뀌면 여기만 고친다.

왜 인터페이스인가: DB 스키마(`photo_analysis`, `ai_recommendations`)는 wes Flyway가 소유하고
아직 협의 중이다(plan.md §5). 그런데 파이프라인은 지금 실제 데이터로 돌려 봐야 한다.
그래서 분석·추천 코드는 `Store`만 보고, 구현이 둘이다.

    LocalStore   out/<갤러리>/ 아래 JSONL·npy.  데이터셋 폴더로 테스트한다.
    DbStore      wes 공유 Postgres (V29: photo_analysis · ai_recommendations …). 운영·로컬 E2E.

접근 규칙(CLAUDE.md)은 인터페이스 모양으로 강제한다 —
  · `photo_selection_items`·`pair_comparison_events`는 **읽기 메서드만** 있다
  · `photo_ratings`는 읽지도 않는다 (정책상 AI 입력에서 제외) — DbStore의 ratings는 항상 빈 dict
  · `photo_selections.status`를 만지는 메서드는 아예 없다
  · 쓰기는 `write_analysis`와 `write_recommendations` 둘뿐이다
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
    #: v2 컨셉 그룹(촬영 세트 단위). DB에는 아직 컬럼이 없어 `sub_scores.concept_id` 로 오간다 —
    #: wes 에 `photo_analysis.concept_id` 추가를 요청한 상태(docs/plan-v2-slim.md §4).
    concept_id: int = -1

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

    selected  — `photo_selection_items`의 photo_id들. **읽기 전용**
    ratings   — 로컬 review 페이지의 별점. DB 모드에서는 비어 있다 (`photo_ratings` 접근 금지)
    pairs     — `pair_comparison_events`: (chosen, rejected, axis)
    rejected  — `ai_recommendations.rejected_at`이 찍힌 photo_id들. 후보에서 다시 안 보여 주는 근거
    feedback  — 자연어 피드백 (ai_recommendations 반응 또는 별도 이벤트)
    """

    selected: list[str] = field(default_factory=list)
    ratings: dict[str, int] = field(default_factory=dict)
    pairs: list[tuple[str, str, str]] = field(default_factory=list)
    rejected: list[str] = field(default_factory=list)
    #: 자연어 피드백 문장들 ("가족 사진 더"). LLM이 {axis, tag, delta}로 번역한다. 선택은 못 바꾼다.
    feedback: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not (self.selected or self.ratings or self.pairs or self.rejected or self.feedback)


# ── 인터페이스 ───────────────────────────────────────────────────────────────
class Store(Protocol):
    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]: ...
    def read_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]: ...
    def write_analysis(self, gallery: str, rows: list[PhotoAnalysis],
                       embeddings: tuple[list[str], np.ndarray]) -> None: ...
    def read_evidence(self, gallery: str, selection_id: str | None) -> Evidence: ...
    def read_recommendations(self, gallery: str) -> list[Recommendation]: ...
    def write_recommendations(self, gallery: str, rows: list[Recommendation]) -> None: ...
    def preview_path(self, gallery: str, photo_id: str) -> str | None: ...


# ── 로컬 구현 ────────────────────────────────────────────────────────────────
class LocalStore:
    """out/v2/<갤러리 slug>/ 아래에 파일로 둔다 (v1 과 폴더를 나눈다).

        analysis.jsonl          photo_analysis 행들
        embeddings.npy + ids    임베딩 (photo_id 순서는 embedding_ids.json)
        evidence.json           취향 신호 (review 페이지가 내보내거나 손으로 만든다)
        recommendations.jsonl   초안 (round별 누적)
    """

    def __init__(self, root: Path, dataset_root: Path | None = None) -> None:
        self.root = Path(root)
        #: 사진 파일이 있는 곳. photo_id 가 이 루트 기준 상대 경로다 (gallery.load_local). 없으면 LLM 에 사진을 못 보낸다.
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def preview_path(self, gallery: str, photo_id: str) -> str | None:
        if self.dataset_root is None:
            return None
        p = self.dataset_root / photo_id
        return str(p) if p.is_file() else None

    def _dir(self, gallery: str) -> Path:
        # v1 은 out/<gallery>/, v2 는 out/v2/<gallery>/ — 두 버전의 analysis.jsonl 형식이 달라 같은 폴더를 쓰면 안 된다
        d = self.root / "v2" / gallery.replace("/", "__")
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


def _jsonb(value) -> str:
    """jsonb 컬럼용. 얼굴이 없는 사진의 eyes_open·smile 은 NaN 인데 Postgres jsonb 는 NaN 을
    거절한다(JSON 표준에 없다). null 로 보내고, 읽는 쪽은 .get() 기본값으로 처리한다."""
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
    """wes 공유 Postgres. 스키마는 wes Flyway(V29)가 소유하고 여기는 읽고 쓰기만 한다.

    id 규약: 파이프라인은 photo_id·gallery를 문자열로 다루므로 읽을 때 `str()`, 쓸 때 `int()`.
    `gallery`는 `photos.gallery_id`다. 추천은 셀렉 단위라 `selection_id`를 생성자나
    `read_evidence` 인자로 받는다 — `Store` 시그니처엔 gallery만 있어서다.

    임베딩은 **읽기만 한다.** `photo_analysis.embedding`은 임베더 Lambda의 DINOv3(ViT-B/16, 768d)이고 wes의
    연사 클러스터 탭도 같은 벡터를 쓴다. analyze가 만드는 CLIP 임베딩은 미학 점수 계산에만
    쓰고 저장하지 않는다 — 같은 768차원이라 DB가 거절하지 않으므로 여기서 막는다.
    그래서 `write_analysis`의 `embeddings` 인자는 무시되고, 클러스터·선호 유사도는 DINOv3 위에서 돈다.
    """

    ANALYSIS_COLUMNS = (
        "scene", "framing", "lighting", "expression", "subjects", "caption",
        "technical_pct", "aesthetic_pct", "face_boxes", "sub_scores",
        "cluster_id", "cluster_rank", "rank_reason_code", "model_version",
    )

    def __init__(self, settings, selection_id: str | None = None, connection=None) -> None:
        from photoselect import db as db_mod
        self.conn = connection or db_mod.connect(settings)
        self.selection_id = int(selection_id) if selection_id is not None else None
        self._settings = settings
        self._storage = None

    def preview_path(self, gallery: str, photo_id: str) -> str | None:
        """analyze 가 내려받은 미리보기가 work_dir 에 남아 있으면 그것, 없으면 S3 에서 다시 받는다
        (draft 는 analyze 와 다른 프로세스·Lambda 에서 돌 수 있다). 버킷 설정이 없으면 None."""
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
        except Exception as exc:  # noqa: BLE001 — 사진 없이 텍스트만으로 이유를 만든다
            log.warning("미리보기 내려받기 실패 photo=%s: %s", photo_id, exc)
            return None

    # ── 셀렉 ↔ 갤러리 ──
    def gallery_of_selection(self, selection_id: str | int) -> str:
        with self.conn.cursor() as cur:
            cur.execute("SELECT gallery_id FROM photo_selections WHERE id = %s AND deleted_at IS NULL",
                        (int(selection_id),))
            row = cur.fetchone()
        if row is None:
            raise SystemExit(f"셀렉이 없다: photo_selections.id={selection_id}")
        return str(row[0])

    def target_count(self, gallery: str) -> int | None:
        """계약 장수(`galleries.max_selectable_photo_count`). 없으면 None — config 기본값을 쓴다."""
        with self.conn.cursor() as cur:
            cur.execute("SELECT max_selectable_photo_count FROM galleries WHERE id = %s", (int(gallery),))
            row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None

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
            d["caption"] = d["caption"] or ""
            d["rank_reason_code"] = d["rank_reason_code"] or ""
            d["face_boxes"] = d["face_boxes"] or {}
            d["sub_scores"] = dict(d["sub_scores"] or {})
            d["concept_id"] = int(d["sub_scores"].pop("concept_id", -1))
            out.append(PhotoAnalysis(**d))
        return out

    def read_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        """한 갤러리의 임베딩 행렬. **`embedding_model`이 둘 이상 섞여 있으면 실패한다** —
        임베더 모델 교체(DINOv2→DINOv3) 뒤 일부만 재임베딩된 갤러리에서 코사인을 재면 다른
        공간의 벡터를 비교하게 되어 클러스터·MMR이 조용히 틀어진다. 전량 force 재임베딩이 먼저다.
        """
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT a.photo_id, a.embedding, a.embedding_model FROM photo_analysis a "
                "JOIN photos p ON p.id = a.photo_id "
                "WHERE p.gallery_id = %s AND p.deleted_at IS NULL AND a.embedding IS NOT NULL "
                "ORDER BY p.display_order, p.id",
                (int(gallery),),
            )
            rows = cur.fetchall()
        if not rows:
            return [], np.zeros((0, 0))
        models = {r[2] for r in rows}
        if len(models) > 1:
            raise RuntimeError(
                f"gallery {gallery}: embedding_model이 섞여 있다 {sorted(map(str, models))} — "
                "임베더 force 재실행으로 한 모델로 맞춘 뒤 분석하라")
        ids = [str(r[0]) for r in rows]
        return ids, np.stack([np.asarray(r[1], dtype=np.float32) for r in rows])

    def write_analysis(self, gallery: str, rows: list[PhotoAnalysis],
                       embeddings: tuple[list[str], np.ndarray]) -> None:
        """분석 컬럼만 UPSERT. embedding·embedding_model은 임베더 것이라 건드리지 않는다.

        행이 없을 수도 있다(임베더가 아직 안 돈 사진) — 그때는 임베딩 없는 행이 INSERT 되고,
        임베더가 나중에 같은 행에 벡터를 채운다. 한 트랜잭션이라 갤러리 절반만 새 백분위로
        남는 일은 없다.
        """
        params = [
            (
                int(r.photo_id), r.scene, r.framing, r.lighting, r.expression, r.subjects,
                r.caption or None, float(r.technical_pct), float(r.aesthetic_pct),
                _jsonb(r.face_boxes), _jsonb({**r.sub_scores, "concept_id": int(r.concept_id)}),
                int(r.cluster_id), int(r.cluster_rank), r.rank_reason_code or None, r.model_version,
            )
            for r in rows
        ]
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO photo_analysis
                    (photo_id, scene, framing, lighting, expression, subjects, caption,
                     technical_pct, aesthetic_pct, face_boxes, sub_scores,
                     cluster_id, cluster_rank, rank_reason_code, model_version,
                     analyzed_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s,
                        now(), now(), now())
                ON CONFLICT (photo_id) DO UPDATE SET
                    scene = EXCLUDED.scene, framing = EXCLUDED.framing, lighting = EXCLUDED.lighting,
                    expression = EXCLUDED.expression, subjects = EXCLUDED.subjects, caption = EXCLUDED.caption,
                    technical_pct = EXCLUDED.technical_pct, aesthetic_pct = EXCLUDED.aesthetic_pct,
                    face_boxes = EXCLUDED.face_boxes, sub_scores = EXCLUDED.sub_scores,
                    cluster_id = EXCLUDED.cluster_id, cluster_rank = EXCLUDED.cluster_rank,
                    rank_reason_code = EXCLUDED.rank_reason_code, model_version = EXCLUDED.model_version,
                    analyzed_at = now(), updated_at = now(), version = photo_analysis.version + 1
                """,
                params,
            )
        self.conn.commit()
        log.info("photo_analysis 적재: gallery=%s %d행 (임베딩은 임베더 것 유지)", gallery, len(params))

    # ── evidence (읽기 전용) ──
    def read_evidence(self, gallery: str, selection_id: str | None = None) -> Evidence:
        sid = int(selection_id) if selection_id is not None else self.selection_id
        if sid is None:
            return Evidence()
        with self.conn.cursor() as cur:
            cur.execute("SELECT photo_id FROM photo_selection_items WHERE selection_id = %s", (sid,))
            selected = [str(r[0]) for r in cur.fetchall()]
            cur.execute(
                "SELECT photo_a, photo_b, chosen_photo_id, axis FROM pair_comparison_events "
                "WHERE selection_id = %s ORDER BY id",
                (sid,),
            )
            pairs = [
                (str(chosen), str(a if chosen == b else b), str(axis))
                for a, b, chosen, axis in cur.fetchall()
            ]
            cur.execute(
                "SELECT photo_id FROM ai_recommendations WHERE selection_id = %s AND rejected_at IS NOT NULL",
                (sid,),
            )
            rejected = [str(r[0]) for r in cur.fetchall()]
        return Evidence(selected=selected, pairs=pairs, rejected=rejected)

    # ── recommendations ──
    def _require_selection(self) -> int:
        if self.selection_id is None:
            raise SystemExit("DB 모드의 추천은 셀렉 단위다 — --selection-id 가 필요하다")
        return self.selection_id

    def read_recommendations(self, gallery: str) -> list[Recommendation]:
        sid = self._require_selection()
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT photo_id, round, rank, score_breakdown, reason FROM ai_recommendations "
                "WHERE selection_id = %s ORDER BY round, rank",
                (sid,),
            )
            rows = cur.fetchall()
        return [Recommendation(photo_id=str(p), round=int(rd), rank=int(rk),
                               score_breakdown=bd or {}, reason=reason or "")
                for p, rd, rk, bd, reason in rows]

    def write_recommendations(self, gallery: str, rows: list[Recommendation]) -> None:
        """한 라운드를 INSERT. (selection, photo) 유니크는 지키는 게 아니라 전제다 — draft가
        이전 라운드를 제외하므로 충돌은 버그이고 그대로 실패시킨다."""
        sid = self._require_selection()
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO ai_recommendations
                    (selection_id, photo_id, round, rank, score_breakdown, reason,
                     presented_at, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, now(), now(), now())
                """,
                [(sid, int(r.photo_id), r.round, r.rank, _jsonb(r.score_breakdown), r.reason or None)
                 for r in rows],
            )
        self.conn.commit()
        log.info("ai_recommendations 적재: selection=%s round=%s %d행", sid,
                 rows[0].round if rows else None, len(rows))
