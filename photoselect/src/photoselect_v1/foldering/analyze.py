"""v3 배치 A(FULL의 사진별 분석) — 갤러리 전수, CPU. VLM 없음.

    ARNIQA(spaq)                → technical_score → technical_pct
    CLIP ViT-L/14 + LAION MLP   → aesthetic_score → aesthetic_pct
        CLIP 벡터는 저장한다 (photo_analysis.clip_embedding) — naming 잡이 재분석 없이 다시 돌고,
        소그룹 최근접 배정·부모 검증이 쓴다
    고전 지표                    → sharpness · highlight_clip · shadow_clip · saturation (sub_scores)
    임베더 벡터 E (DB: DINOv3 / 로컬: CLIP이 겸임)
        → 연사 클러스터 (순서 ≤ window ∧ cos ≥ threshold)   → cluster_id · cluster_rank
        → 임베딩 그룹: concat(E ⊕ CLIP) 계층 클러스터        → embed_group_id
          (ai-folder-structure.md 실측 — concat이 DINOv3 경계를 지키며(ARI 0.95) VLM 이름을 안정시킨다.
           로컬은 E=CLIP이라 concat이 CLIP 단독으로 퇴화한다 — 데이터셋 테스트 한정 감수)
    → store.write_analysis (V45)

FULL 잡은 이 뒤에 naming(foldering.naming)까지 이어 돈다 — 그 배선은 worker가 한다.
재개: 같은 MODEL_VERSION 이고 임베딩·CLIP이 모두 저장된 사진은 건너뛴다.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from photoselect_v1.foldering import cluster, concept
from photoselect_v1.config import MODEL_VERSION, Settings
from photoselect_v1.gallery import PhotoRef
from photoselect_v1.store import PhotoAnalysis, Store

log = logging.getLogger(__name__)

UNKNOWN = "unknown"


@dataclass
class RunResult:
    gallery: str
    pipeline: str = "v3"
    targets: int = 0
    processed: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    clusters: int = 0
    groups: dict = field(default_factory=dict)
    group_distance: float = 0.0
    similarity_profile: dict = field(default_factory=dict)
    subjects_used: bool = False
    elapsed_seconds: float = 0.0
    per_stage_seconds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gallery": self.gallery, "pipeline": self.pipeline, "targets": self.targets,
            "processed": self.processed, "skipped": self.skipped, "failed": self.failed,
            "clusters": self.clusters, "groups": {k: round(v, 3) for k, v in self.groups.items()},
            "groupDistance": self.group_distance,
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


def concat_space(E: np.ndarray, C: np.ndarray) -> np.ndarray:
    """DINOv3 ⊕ CLIP — 각 벡터를 정규화해 이어 붙이고 다시 정규화. 코사인 = 두 공간 코사인의 평균."""
    def norm(M: np.ndarray) -> np.ndarray:
        return M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-8, None)
    X = np.concatenate([norm(E), norm(C)], axis=1)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


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
    """연사 클러스터 안 대표(cluster_rank 0)와 사유. V45에는 컬럼이 없어 sub_scores.rank_reason 으로."""
    by_cluster: dict[int, list[PhotoAnalysis]] = {}
    for r in rows:
        by_cluster.setdefault(r.cluster_id, []).append(r)
    for members in by_cluster.values():
        members.sort(key=_rep_key)
        for rank, m in enumerate(members):
            m.cluster_rank = rank
            if rank == 0:
                m.sub_scores["rank_reason"] = _rep_reason(m, members[1:])


def run(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings, force: bool = False,
        **_ignored) -> dict:
    started = time.monotonic()
    k = settings.knobs
    result = RunResult(gallery=gallery, targets=len(refs))
    stage: dict[str, float] = {}

    previous = {} if force else {
        r.photo_id: r for r in store.read_analysis(gallery) if r.model_version == MODEL_VERSION}
    prev_ids, prev_emb = store.read_embeddings(gallery)
    prev_emb_map = dict(zip(prev_ids, prev_emb)) if len(prev_ids) else {}
    prev_clip_ids, prev_clip = store.read_clip_embeddings(gallery)
    prev_clip_map = {} if force else (dict(zip(prev_clip_ids, prev_clip)) if len(prev_clip_ids) else {})
    todo = [r for r in refs
            if r.photo_id not in previous or r.photo_id not in prev_emb_map or r.photo_id not in prev_clip_map]
    result.skipped = len(refs) - len(todo)
    log.info("[v3] 갤러리 %s: 대상 %d장, 이미 분석됨 %d장", gallery, len(refs), result.skipped)

    t0 = time.monotonic()
    from photoselect_v1.foldering import classical
    from photoselect_v1.foldering.runners import ArniqaRunner, LaionRunner
    laion, arniqa = LaionRunner(), ArniqaRunner()
    tagger = None
    if k.subjects_zero_shot:
        from photoselect_v1.subjects import SubjectsTagger
        tagger = SubjectsTagger(laion)
    result.subjects_used = tagger is not None
    stage["load"] = time.monotonic() - t0

    rows: dict[str, PhotoAnalysis] = dict(previous)
    embs: dict[str, np.ndarray] = dict(prev_emb_map)
    clips: dict[str, np.ndarray] = dict(prev_clip_map)
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
                photo_id=ref.photo_id, subjects=subjects,
                sub_scores=sub, model_version=MODEL_VERSION,
            )
            clips[ref.photo_id] = clip_emb
            embs.setdefault(ref.photo_id, clip_emb)   # DB 모드는 임베더 벡터가 이미 있어 여기선 안 쓰인다
            result.processed += 1
            t_photo += time.monotonic() - t0
            if i % 20 == 0 or i == len(todo):
                log.info("  %d/%d  (%.2fs/장)", i, len(todo), t_photo / i)
        except Exception as exc:  # noqa: BLE001 — 한 장 실패가 잡을 죽이면 안 된다
            log.exception("사진 실패 %s: %s", ref.photo_id, exc)
            result.failed.append(ref.photo_id)
    stage["photos"] = t_photo

    t0 = time.monotonic()
    ordered_refs = [r for r in refs
                    if r.photo_id in rows and r.photo_id in embs and r.photo_id in clips]
    ordered = [rows[r.photo_id] for r in ordered_refs]
    ids = [r.photo_id for r in ordered]
    E = np.stack([embs[i] for i in ids]) if ids else np.zeros((0, 768))
    C = np.stack([clips[i] for i in ids]) if ids else np.zeros((0, 768))

    for key, col in (("technical_score", "technical_pct"), ("aesthetic_score", "aesthetic_pct")):
        for r, v in zip(ordered, percentile([r.sub_scores.get(key, math.nan) for r in ordered])):
            setattr(r, col, v)
    for r, v in zip(ordered, percentile([r.sub_scores.get("sharpness", math.nan) for r in ordered])):
        r.sub_scores["sharpness_pct"] = v

    if len(E):
        parts = cluster.partition_order([r.camera for r in ordered_refs],
                                        [r.taken_at for r in ordered_refs])
        cids = cluster.cluster_bursts_partitioned(E, parts, k.burst_threshold, k.burst_window)
        X = concat_space(E, C)
        gids, used_d = concept.concept_groups(X, k.group_distance, k.group_min_groups,
                                              k.group_max_share, k.group_frag_share)
    else:
        cids, gids, used_d = np.zeros(0, int), np.zeros(0, int), k.group_distance
    for r, c, g in zip(ordered, cids, gids):
        r.cluster_id, r.embed_group_id = int(c), int(g)
    assign_ranks(ordered)
    result.clusters = int(cids.max()) + 1 if len(cids) else 0
    result.groups = concept.group_profile(gids)
    result.group_distance = used_d
    result.similarity_profile = cluster.similarity_profile(E, k.burst_window) if len(E) > 1 else {}
    stage["postprocess"] = time.monotonic() - t0

    store.write_analysis(gallery, ordered, (ids, E), (ids, C))
    result.per_stage_seconds = stage
    result.elapsed_seconds = time.monotonic() - started
    return result.to_dict()
