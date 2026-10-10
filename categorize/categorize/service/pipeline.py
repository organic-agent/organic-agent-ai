"""CATEGORIZE 본체 — 갤러리 단위 구간 · 컨셉 · 2층 그룹 · 이름. **torch 없음** (numpy · scipy · Bedrock).

입력은 전부 DB(또는 로컬 파일)에 저장된 것이다:
    E  임베더의 DINOv3 (photo_analysis.embedding)          ← 로컬 데이터셋 모드는 CLIP 이 겸한다
    C  score 가 저장한 CLIP (photo_analysis.clip_embedding)
    score 의 원점수 (sub_scores), photos.taken_at · camera

    백분위     technical_score · aesthetic_score · sharpness → *_pct (갤러리 안 순위)
    연사       E, 카메라 파티션 ∧ 순서 창 ∧ cos ≥ threshold      → burst_id · burst_rank
    구간       촬영 시각 공백(없으면 DINOv3 Ward)                  → segment.Segmentation   (service/segment.py)
    1층        구간 타일을 VLM 이 컨셉으로 묶고 이름                → naming.name_concepts   (conceptCount 는 선택)
    2층        1층마다 concat(E ⊕ C) 평균연결 계층 클러스터       → embed_group_id (갤러리 전체에서 유일)
    → store.write_groups (pct · burst · embed_group · sub_scores 만 — subjects·clip_embedding 은 SCORE 의 것)
    → naming.name_details (1층별 2층 이름 · 최근접 · 검증) → concept_assignments

백분위·연사 대표 순위는 화질 점수(score 2단계)가 갤러리 전부에 찼을 때만 매긴다(wes #274 2물결). 한 장이라도 덜 찼으면
NULL 로 두고 폴더용(연사 묶음·구간·그룹·이름)만 쓴다 — 점수 없는 사진을 50 으로 채우면 가짜 순위가 완료처럼 보인다.
나중에 화질 점수가 다 차면 wes 가 `rank()`(rank 모드)를 불러 순위만 채운다.

갤러리는 **한 번만 읽는다**. 항상 갤러리 전체를 다시 계산한다 — 결정적이고 싸다. 재개는 score 의 일이다.
llm 이 없으면(로컬 확인용) 구간 하나를 1층 하나로 보고 2층까지 저장한 뒤 이름은 건너뛴다.
근거: docs/experiments/concept-segmentation-2026-09-30.md
"""

from __future__ import annotations

import logging
import math
import time

import numpy as np

from categorize.config.settings import PIPELINE_VERSION, Settings
from categorize.domain.analysis import PhotoAnalysis, Store
from categorize.domain.photo import PhotoRef
from categorize.domain.run import CategorizeResult, Grouped
from categorize.service import burst, grouping, segment

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
    """연사 대표(burst_rank 0)와 사유. V45에는 컬럼이 없어 sub_scores.rank_reason 으로."""
    by_burst: dict[int, list[PhotoAnalysis]] = {}
    for r in rows:
        by_burst.setdefault(r.burst_id, []).append(r)
    for members in by_burst.values():
        members.sort(key=_rep_key)
        for rank, m in enumerate(members):
            m.burst_rank = rank
            if rank == 0:
                m.sub_scores["rank_reason"] = _rep_reason(m, members[1:])


def set_percentiles(rows: list[PhotoAnalysis]) -> None:
    """화질·미학·선명도의 갤러리 안 백분위. 순위의 재료라 화질 점수가 다 찬 행에만 부른다."""
    for key, col in (("technical_score", "technical_pct"), ("aesthetic_score", "aesthetic_pct")):
        for r, v in zip(rows, percentile([r.sub_scores.get(key, math.nan) for r in rows])):
            setattr(r, col, v)
    for r, v in zip(rows, percentile([r.sub_scores.get("sharpness", math.nan) for r in rows])):
        r.sub_scores["sharpness_pct"] = v


def clear_ranks(rows: list[PhotoAnalysis]) -> None:
    """순위 없음 — 백분위·순위·순위 사유를 비운다. 이전 실행이 남긴 값이 새 그룹 위에 남지 않게 한다."""
    for r in rows:
        r.technical_pct = r.aesthetic_pct = r.burst_rank = None
        r.sub_scores.pop("sharpness_pct", None)
        r.sub_scores.pop("rank_reason", None)


def rank(store: Store, gallery: str) -> dict:
    """rank 모드 — 그룹(폴더)이 이미 있는 사진의 백분위·연사 대표 순위만 다시 매긴다. Bedrock·잡 없음.

    연사 묶음은 full 이 저장한 `burst_id` 를 그대로 쓴다(다시 묶으면 폴더와 어긋날 수 있다). 화질 점수가 덜 찬 사진이 있으면
    매기지 않고 돌아간다 — 백분위는 상대 순위라 덜 찬 채로 매기면 빈 사진이 섞인다. wes 는 다 찼을 때만 부르므로 그 경우는
    그 사이에 사진이 바뀐 것이고, wes 가 기다릴 시간 뒤 다시 보낸다.
    """
    started = time.monotonic()
    data = store.read_gallery(gallery)
    rows = [r for r in data.rows if r.pipeline_version == PIPELINE_VERSION and r.embed_group_id >= 0 and r.burst_id >= 0]
    pending = sum(1 for r in rows if not r.quality_scored)
    if not rows or pending:
        log.warning("[categorize] 갤러리 %s rank: 매길 수 없다 — 그룹 있는 사진 %d장 중 화질 점수 대기 %d장", gallery, len(rows), pending)
        return {"gallery": gallery, "mode": "rank", "ranked": 0, "pending": pending}
    set_percentiles(rows)
    assign_ranks(rows)
    store.write_ranks(gallery, rows)
    elapsed = round(time.monotonic() - started, 1)
    log.info("[categorize] 갤러리 %s rank: %d장 · %.1fs", gallery, len(rows), elapsed)
    return {"gallery": gallery, "mode": "rank", "ranked": len(rows), "pending": 0, "elapsedSeconds": elapsed}


def group(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings) -> tuple[Grouped, CategorizeResult]:
    """백분위 · 연사 · 구간을 계산한다. 저장하지 않는다 — embed_group_id 는 1층이 정해진 뒤에 매긴다."""
    started = time.monotonic()
    k = settings.knobs
    result = CategorizeResult(gallery=gallery)

    data = store.read_gallery(gallery)
    # 운영 7,189장에서 START→그룹 적재가 45.8s 인데 로그가 없어 읽기/계산을 못 나눴다(#111) — 단계별 소요를 남긴다.
    log.info("[categorize] 갤러리 %s 읽기: %d행 · dinov3 %d · clip %d · %.1fs",
             gallery, len(data.rows), len(data.embeddings), len(data.clip_embeddings), time.monotonic() - started)
    scored = {r.photo_id: r for r in data.rows if r.pipeline_version == PIPELINE_VERSION}
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
    norm = lambda M: M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-8, None)   # noqa: E731
    E = norm(np.stack([embs[i] for i in ids]))
    C = np.stack([clips[i] for i in ids])

    ranked = all(r.quality_scored for r in ordered)
    if ranked:
        set_percentiles(ordered)

    t0 = time.monotonic()
    parts = burst.partition_order([r.camera for r in ordered_refs], [r.taken_at for r in ordered_refs])
    burst_ids = burst.cluster_bursts_partitioned(E, parts, k.burst_threshold, k.burst_window)
    for r, b in zip(ordered, burst_ids):
        r.burst_id = int(b)
    if ranked:
        assign_ranks(ordered)
    else:
        clear_ranks(ordered)
        log.info("[categorize] 갤러리 %s: 화질 점수가 덜 찬 사진 %d장 — 순위는 비워 두고 폴더용만 쓴다(rank 모드가 뒤에 채운다)",
                 gallery, sum(1 for r in ordered if not r.quality_scored))
    t1 = time.monotonic()
    seg = segment.build([r.taken_at for r in ordered_refs], E, [int(b) for b in burst_ids], k)
    t2 = time.monotonic()
    log.info("[categorize] 갤러리 %s 연사·구간: %d장 · 연사 %d (%.1fs) · 구간 %d %s (%.1fs) · 읽기 뒤 누적 %.1fs",
             gallery, len(ordered), int(burst_ids.max()) + 1 if len(burst_ids) else 0, t1 - t0,
             len(seg.units), seg.mode, t2 - t1, t2 - started)

    result.photos = len(ordered)
    result.ranked = ranked
    result.bursts = int(burst_ids.max()) + 1 if len(burst_ids) else 0
    result.segment_mode, result.segments = seg.mode, len(seg.units)
    result.similarity_profile = burst.similarity_profile(E, k.burst_window) if len(E) > 1 else {}
    result.elapsed_seconds = time.monotonic() - started
    return Grouped(rows=ordered, X=concat_space(E, C), E=E, segmentation=seg), result


def run(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings, llm,
        job_id: int | None = None, concept_count: int | None = None) -> dict:
    """구간 → 1층(VLM) → 2층 그룹 저장 → 2층 이름. llm 이 None 이면 구간 = 1층, 이름은 skipped."""
    from categorize.service import naming

    started = time.monotonic()
    k = settings.knobs
    grouped, result = group(store, gallery, refs, settings)
    units = grouped.segmentation.units

    if llm is not None:
        unit_concept, concepts, summary = naming.name_concepts(store, gallery, grouped, settings, llm, concept_count)
    else:
        unit_concept = list(range(len(units)))
        concepts, summary = [], {"segmentMode": grouped.segmentation.mode, "segments": len(units)}
    concept_of_row = [0] * len(grouped.rows)
    for u, members in enumerate(units):
        for i in members:
            concept_of_row[i] = unit_concept[u]

    embed_group_ids = grouping.detail_groups(grouped.X, concept_of_row, k.group_distance, k.group_frag_share)
    for r, g in zip(grouped.rows, embed_group_ids):
        r.embed_group_id = int(g)
    store.write_groups(gallery, grouped.rows)
    result.embed_groups = grouping.group_profile(embed_group_ids)
    result.concepts = summary

    if llm is not None:
        result.naming = naming.name_details(store, gallery, grouped, concepts, concept_of_row, settings, llm, job_id=job_id)
        result.naming["llmCalls"] += 1                      # 1층 호출
    else:
        result.naming = "skipped (no --llm)"
    result.elapsed_seconds = time.monotonic() - started
    return result.to_dict()
