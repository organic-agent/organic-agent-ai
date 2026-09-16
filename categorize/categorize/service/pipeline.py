"""CATEGORIZE 본체 — 갤러리 단위 그룹화 + 이름 짓기. **torch 없음** (numpy · scipy · Bedrock).

입력은 전부 DB(또는 로컬 파일)에 저장된 것이다:
    E  임베더의 DINOv3 (photo_analysis.embedding)          ← 로컬 데이터셋 모드는 CLIP 이 겸한다
    C  score 가 저장한 CLIP (photo_analysis.clip_embedding)
    score 의 원점수·subjects·clip_parent (sub_scores)

    백분위          technical_score · aesthetic_score · sharpness → *_pct (갤러리 안 순위)
    연사 클러스터    E, 카메라 파티션 ∧ 순서 창 ∧ cos ≥ threshold      → cluster_id · cluster_rank
    임베딩 그룹      X = concat(E ⊕ C), 평균연결 계층 클러스터, 적응 임계 → embed_group_id
    → store.write_groups (pct · cluster · group · sub_scores 만 — subjects·clip_embedding 은 SCORE 의 것)
    → naming.run (Bedrock 이름 · 소그룹 최근접 · 저장된 clip_parent 다수결 검증) → ai_concept_assignments

갤러리는 **한 번만 읽는다** — `store.read_gallery` 한 쿼리로 행·벡터를 받고, 그룹화가 만든 행과 concat 공간([Grouped])을
naming 에 그대로 넘긴다. 예전엔 naming 이 셋을 다시 읽고 X 를 다시 만들었다(7천 장이면 벡터 44MB 를 두 번).
항상 갤러리 전체를 다시 계산한다 — 결정적이고 싸다(822장 수 초). 재개는 score 의 일이다.
FULL 잡은 score Lambda 가 끝에서 이 함수를 체인으로 부르고, NAMING 잡은 wes 가 직접 부른다. (#26·#35)
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from categorize.config.settings import MODEL_VERSION, Settings
from categorize.domain.analysis import PhotoAnalysis, Store
from categorize.domain.photo import PhotoRef
from categorize.domain.run import CategorizeResult, Grouped
from categorize.service import cluster, concept

log = logging.getLogger(__name__)


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


def group(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings) -> tuple[Grouped, CategorizeResult]:
    """백분위·연사·임베딩 그룹을 계산해 저장한다. naming 은 하지 않는다 — 대신 naming 이 쓸 [Grouped] 를 돌려준다."""
    started = time.monotonic()
    k = settings.knobs
    result = CategorizeResult(gallery=gallery)

    data = store.read_gallery(gallery)
    # 운영 7,189장에서 START→그룹 적재가 45.8s 인데 로그가 없어 읽기/계산을 못 나눴다(#111) — 단계별 소요를 남긴다.
    log.info("[categorize] 갤러리 %s 읽기: %d행 · dinov3 %d · clip %d · %.1fs",
             gallery, len(data.rows), len(data.embeddings), len(data.clip_embeddings), time.monotonic() - started)
    scored = {r.photo_id: r for r in data.rows if r.model_version == MODEL_VERSION}
    clips = data.clip_embeddings
    embs = data.embeddings
    if not embs:
        # 로컬 데이터셋 모드 — 임베더가 없다. concat 이 CLIP 단독으로 퇴화하는 것을 감수한다.
        log.warning("[categorize] 갤러리 %s: 임베더 벡터가 없다 — CLIP 을 E 자리에 쓴다 (로컬 한정)", gallery)
        embs = clips
        result.embeddings_source = "clip"

    ordered_refs = [r for r in refs if r.photo_id in scored and r.photo_id in embs and r.photo_id in clips]
    missing = len(refs) - len(ordered_refs)
    if missing:
        log.warning("[categorize] 갤러리 %s: 점수나 벡터가 없는 사진 %d장은 제외 — SCORE·임베더가 먼저다", gallery, missing)
    if not ordered_refs:
        raise RuntimeError(f"갤러리 {gallery}: 점수와 벡터가 모두 있는 사진이 없다 — SCORE(FULL) 이 먼저다")
    ordered = [scored[r.photo_id] for r in ordered_refs]
    ids = [r.photo_id for r in ordered]
    E = np.stack([embs[i] for i in ids])
    C = np.stack([clips[i] for i in ids])

    for key, col in (("technical_score", "technical_pct"), ("aesthetic_score", "aesthetic_pct")):
        for r, v in zip(ordered, percentile([r.sub_scores.get(key, math.nan) for r in ordered])):
            setattr(r, col, v)
    for r, v in zip(ordered, percentile([r.sub_scores.get("sharpness", math.nan) for r in ordered])):
        r.sub_scores["sharpness_pct"] = v

    t0 = time.monotonic()
    parts = cluster.partition_order([r.camera for r in ordered_refs], [r.taken_at for r in ordered_refs])
    cids = cluster.cluster_bursts_partitioned(E, parts, k.burst_threshold, k.burst_window)
    t1 = time.monotonic()
    X = concat_space(E, C)
    gids, used_d = concept.concept_groups(X, k.group_distance, k.group_min_groups,
                                          k.group_max_share, k.group_frag_share)
    for r, c, g in zip(ordered, cids, gids):
        r.cluster_id, r.embed_group_id = int(c), int(g)
    assign_ranks(ordered)
    t2 = time.monotonic()
    log.info("[categorize] 갤러리 %s 그룹화: %d장 · 연사 %d (%.1fs) · 그룹 %d (%.1fs) · 읽기 뒤 누적 %.1fs",
             gallery, len(ordered), int(cids.max()) + 1 if len(cids) else 0, t1 - t0,
             int(gids.max()) + 1 if len(gids) else 0, t2 - t1, t2 - started)

    store.write_groups(gallery, ordered)

    result.photos = len(ordered)
    result.clusters = int(cids.max()) + 1 if len(cids) else 0
    result.groups = concept.group_profile(gids)
    result.group_distance = used_d
    result.similarity_profile = cluster.similarity_profile(E, k.burst_window) if len(E) > 1 else {}
    result.elapsed_seconds = time.monotonic() - started
    return Grouped(rows=ordered, X=X), result


def run(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings, llm,
        job_id: int | None = None) -> dict:
    """그룹화 뒤 naming 까지. llm 이 None 이면 그룹화만 하고 naming 은 skipped."""
    from categorize.service import naming

    started = time.monotonic()
    grouped, result = group(store, gallery, refs, settings)
    if llm is not None:
        result.naming = naming.run(store, gallery, settings, llm, job_id=job_id, grouped=grouped)
    else:
        result.naming = "skipped (no --llm)"
    result.elapsed_seconds = time.monotonic() - started
    return result.to_dict()
