"""SCORE 잡 — 사진별 점수. 갤러리 전수, CPU, torch. VLM 없음.

    ARNIQA(spaq)                → technical_score
    CLIP ViT-L/14 + LAION MLP   → aesthetic_score
        CLIP 벡터는 저장한다 (photo_analysis.clip_embedding) — CATEGORIZE 가 재계산 없이 읽는다
        같은 벡터에 텍스트 프롬프트를 대어 subjects(피사체)·clip_parent(부모 검증 라벨)도 여기서
    고전 지표                    → sharpness · highlight_clip · shadow_clip · mean_luma (sub_scores)
    → store.write_scores  (subjects · sub_scores · clip_embedding · model_version 만 — 백분위·클러스터·그룹은
                          CATEGORIZE 의 컬럼이라 건드리지 않는다)

사진마다 독립이라 재개가 쉽다: 같은 MODEL_VERSION 이고 CLIP 벡터가 저장된 사진은 건너뛴다.
임베더(DINOv3) 벡터 유무는 보지 않는다 — 그건 CATEGORIZE 의 입력 조건이다. (#26)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from photoselect.config import MODEL_VERSION, Settings
from photoselect.gallery import PhotoRef
from photoselect.store import PhotoAnalysis, Store

log = logging.getLogger(__name__)

UNKNOWN = "unknown"


@dataclass
class ScoreResult:
    gallery: str
    pipeline: str = "v3"
    mode: str = "score"
    targets: int = 0
    processed: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    subjects_used: bool = False
    elapsed_seconds: float = 0.0
    per_stage_seconds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gallery": self.gallery, "pipeline": self.pipeline, "mode": self.mode,
            "targets": self.targets, "processed": self.processed, "skipped": self.skipped,
            "failed": self.failed, "subjectsUsed": self.subjects_used,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
            "perStageSeconds": {k: round(v, 1) for k, v in self.per_stage_seconds.items()},
        }


def run(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings, force: bool = False) -> dict:
    started = time.monotonic()
    k = settings.knobs
    result = ScoreResult(gallery=gallery, targets=len(refs))
    stage: dict[str, float] = {}

    previous = {} if force else {
        r.photo_id: r for r in store.read_analysis(gallery) if r.model_version == MODEL_VERSION}
    prev_clip_ids, prev_clip = store.read_clip_embeddings(gallery)
    prev_clip_map = {} if force else (dict(zip(prev_clip_ids, prev_clip)) if len(prev_clip_ids) else {})
    todo = [r for r in refs if r.photo_id not in previous or r.photo_id not in prev_clip_map]
    result.skipped = len(refs) - len(todo)
    log.info("[score] 갤러리 %s: 대상 %d장, 이미 점수 있음 %d장", gallery, len(refs), result.skipped)
    if not todo:
        result.elapsed_seconds = time.monotonic() - started
        return result.to_dict()

    t0 = time.monotonic()
    from photoselect import classical
    from photoselect.runners import ArniqaRunner, LaionRunner
    from photoselect.subjects import ParentTagger, SubjectsTagger
    laion, arniqa = LaionRunner(), ArniqaRunner()
    tagger = SubjectsTagger(laion) if k.subjects_zero_shot else None
    parent_tagger = ParentTagger(laion)
    result.subjects_used = tagger is not None
    stage["load"] = time.monotonic() - t0

    rows: list[PhotoAnalysis] = []
    clips: dict[str, np.ndarray] = {}
    t_photo = 0.0
    for i, ref in enumerate(todo, 1):
        try:
            t0 = time.monotonic()
            clip_emb = laion.embed(ref.path)
            aes = laion.score_from_embedding(clip_emb)
            tech = arniqa.score(ref.path)["technical_score"]
            cl = classical.measure(ref.path)
            sub = {"technical_score": tech, "aesthetic_score": aes, **cl}
            subjects = UNKNOWN
            if tagger is not None:
                subjects, margin = tagger.tag(clip_emb)
                sub["subjects_margin"] = margin
            sub["clip_parent"] = parent_tagger.tag(clip_emb)
            rows.append(PhotoAnalysis(photo_id=ref.photo_id, subjects=subjects,
                                      sub_scores=sub, model_version=MODEL_VERSION))
            clips[ref.photo_id] = clip_emb
            result.processed += 1
            t_photo += time.monotonic() - t0
            if i % 20 == 0 or i == len(todo):
                log.info("  %d/%d  (%.2fs/장)", i, len(todo), t_photo / i)
        except Exception as exc:  # noqa: BLE001 — 한 장 실패가 잡을 죽이면 안 된다
            log.exception("사진 실패 %s: %s", ref.photo_id, exc)
            result.failed.append(ref.photo_id)
    stage["photos"] = t_photo

    ids = [r.photo_id for r in rows]
    C = np.stack([clips[i] for i in ids]) if ids else np.zeros((0, 768))
    store.write_scores(gallery, rows, (ids, C))
    result.per_stage_seconds = stage
    result.elapsed_seconds = time.monotonic() - started
    return result.to_dict()
