"""v2 배치 A — 갤러리 전수 분석. VLM·얼굴 없음, CPU 로 사진당 ~1초.

    ARNIQA(spaq)                → technical_score → technical_pct
    CLIP ViT-L/14 + LAION MLP   → aesthetic_score → aesthetic_pct   (CLIP 임베딩은 zero-shot 피사체에도 씀)
    고전 지표                    → sharpness · highlight_clip · shadow_clip (sub_scores)
    임베더 벡터 (DB: DINOv3 / 로컬: CLIP)
        → 연사 클러스터 (순서 ≤ window ∧ cos ≥ threshold)  → cluster_id · cluster_rank · rank_reason_code
        → 컨셉 그룹 (계층 클러스터, 거리 ≤ concept_distance)  → concept_id
    → store.write_analysis

VLM 컬럼(scene·framing·lighting·expression)은 DB CHECK 가 NOT NULL 을 요구해 "unknown" 을 쓴다 —
wes 마이그레이션(CHECK 완화·concept_id 컬럼)까지의 임시. 재개: 같은 V2_MODEL_VERSION 인 사진은 건너뛴다.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from photoselect.v2 import cluster
from photoselect.v2.config import V2_MODEL_VERSION, Settings
from photoselect.v2.gallery import PhotoRef
from photoselect.v2.store import PhotoAnalysis, Store
from photoselect.v2 import concept

log = logging.getLogger(__name__)

UNKNOWN = "unknown"


@dataclass
class RunResult:
    gallery: str
    pipeline: str = "v2"
    targets: int = 0
    processed: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    clusters: int = 0
    concepts: dict = field(default_factory=dict)
    concept_distance: float = 0.0
    similarity_profile: dict = field(default_factory=dict)
    subjects_used: bool = False
    elapsed_seconds: float = 0.0
    per_stage_seconds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gallery": self.gallery, "pipeline": self.pipeline, "targets": self.targets,
            "processed": self.processed, "skipped": self.skipped, "failed": self.failed,
            "clusters": self.clusters, "concepts": {k: round(v, 3) for k, v in self.concepts.items()},
            "conceptDistance": self.concept_distance,
            "similarityProfile": {k: round(v, 3) for k, v in self.similarity_profile.items()},
            "subjectsUsed": self.subjects_used,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
            "perStageSeconds": {k: round(v, 1) for k, v in self.per_stage_seconds.items()},
        }


def percentile(values: list[float]) -> list[float]:
    """갤러리 내 백분위 0~100. NaN 은 50 — 없는 신호가 순위를 흔들면 안 된다."""
    arr = np.array(values, dtype=float)
    ok = ~np.isnan(arr)
    out = np.full(len(arr), 50.0)
    if ok.sum() > 1:
        ranks = np.argsort(np.argsort(arr[ok]))
        out[ok] = 100.0 * ranks / (ok.sum() - 1)
    return out.tolist()


def _rep_key(a: PhotoAnalysis) -> tuple:
    return (-a.technical_pct, -a.sub_scores.get("sharpness", 0.0), -a.aesthetic_pct, a.photo_id)


def _rep_reason(best: PhotoAnalysis, others: list[PhotoAnalysis]) -> str:
    if not others:
        return "single"
    if best.technical_pct - max(o.technical_pct for o in others) > 5:
        return "technical"
    s = best.sub_scores.get("sharpness", 0.0)
    if s > 1.25 * max(o.sub_scores.get("sharpness", 0.0) for o in others):
        return "sharpness"
    if best.aesthetic_pct - max(o.aesthetic_pct for o in others) > 5:
        return "aesthetic"
    return "tie"


def assign_ranks(rows: list[PhotoAnalysis]) -> None:
    """연사 클러스터 안 대표(cluster_rank 0)와 사유. 얼굴 신호 없이 기술 → 선명도 → 미학 순."""
    by_cluster: dict[int, list[PhotoAnalysis]] = {}
    for r in rows:
        by_cluster.setdefault(r.cluster_id, []).append(r)
    for members in by_cluster.values():
        members.sort(key=_rep_key)
        for rank, m in enumerate(members):
            m.cluster_rank = rank
            m.rank_reason_code = _rep_reason(m, members[1:]) if rank == 0 else ""


def run(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings, force: bool = False,
        **_ignored) -> dict:
    started = time.monotonic()
    k = settings.v2
    result = RunResult(gallery=gallery, targets=len(refs))
    stage: dict[str, float] = {}

    previous = {} if force else {
        r.photo_id: r for r in store.read_analysis(gallery) if r.model_version == V2_MODEL_VERSION}
    prev_ids, prev_emb = store.read_embeddings(gallery)
    prev_emb_map = dict(zip(prev_ids, prev_emb)) if len(prev_ids) else {}
    todo = [r for r in refs if r.photo_id not in previous or r.photo_id not in prev_emb_map]
    result.skipped = len(refs) - len(todo)
    log.info("[v2] 갤러리 %s: 대상 %d장, 이미 분석됨 %d장", gallery, len(refs), result.skipped)

    t0 = time.monotonic()
    from photoselect.v2.runners import ArniqaRunner, LaionRunner
    from photoselect.v2 import classical
    laion, arniqa = LaionRunner(), ArniqaRunner()
    tagger = None
    if k.subjects_zero_shot:
        from photoselect.v2.subjects import SubjectsTagger
        tagger = SubjectsTagger(laion)
    result.subjects_used = tagger is not None
    stage["load"] = time.monotonic() - t0

    rows: dict[str, PhotoAnalysis] = dict(previous)
    embs: dict[str, np.ndarray] = dict(prev_emb_map)
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
            rows[ref.photo_id] = PhotoAnalysis(
                photo_id=ref.photo_id, scene=UNKNOWN, framing=UNKNOWN, lighting=UNKNOWN,
                expression=UNKNOWN, subjects=subjects, caption="",
                sub_scores=sub, model_version=V2_MODEL_VERSION,
            )
            embs.setdefault(ref.photo_id, clip_emb)   # DB 모드는 임베더 벡터가 이미 있어 CLIP 은 버려진다
            result.processed += 1
            t_photo += time.monotonic() - t0
            if i % 20 == 0 or i == len(todo):
                log.info("  %d/%d  (%.2fs/장)", i, len(todo), t_photo / i)
        except Exception as exc:  # noqa: BLE001 — 한 장 실패가 잡을 죽이면 안 된다
            log.exception("사진 실패 %s: %s", ref.photo_id, exc)
            result.failed.append(ref.photo_id)
    stage["photos"] = t_photo

    t0 = time.monotonic()
    ordered = [rows[r.photo_id] for r in refs if r.photo_id in rows]
    ids = [r.photo_id for r in ordered]
    E = np.stack([embs[i] for i in ids]) if ids else np.zeros((0, 768))

    for key, col in (("technical_score", "technical_pct"), ("aesthetic_score", "aesthetic_pct")):
        for r, v in zip(ordered, percentile([r.sub_scores.get(key, math.nan) for r in ordered])):
            setattr(r, col, v)
    for r, v in zip(ordered, percentile([r.sub_scores.get("sharpness", math.nan) for r in ordered])):
        r.sub_scores["sharpness_pct"] = v

    if len(E):
        cids = cluster.cluster_bursts(E, k.burst_threshold, k.burst_window)
        gids, used_d = concept.concept_groups(E, k.concept_distance, k.concept_min_groups, k.concept_max_share)
    else:
        cids, gids, used_d = np.zeros(0, int), np.zeros(0, int), k.concept_distance
    for r, c, g in zip(ordered, cids, gids):
        r.cluster_id, r.concept_id = int(c), int(g)
    assign_ranks(ordered)
    result.clusters = int(cids.max()) + 1 if len(cids) else 0
    result.concepts = concept.group_profile(gids)
    result.concept_distance = used_d
    result.similarity_profile = cluster.similarity_profile(E, k.burst_window) if len(E) > 1 else {}
    stage["postprocess"] = time.monotonic() - t0

    store.write_analysis(gallery, ordered, (ids, E))
    result.per_stage_seconds = stage
    result.elapsed_seconds = time.monotonic() - started
    return result.to_dict()
