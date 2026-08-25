"""갤러리 하나를 전수 분석한다. 진입점(`__main__ analyze` / EC2 루프)이 이 함수를 부른다.

    경량 3종 (faces · laion · arniqa)  →  원점수
    CLIP 임베딩                        →  근접 중복 클러스터 (A-6)
    VLM 태그·캡션 (있으면)              →  5축 + caption
    갤러리 내 백분위                    →  technical_pct · aesthetic_pct
    클러스터 대표 선정 (A-7)            →  cluster_rank · rank_reason_code
    → store.write_analysis

재실행이 안전하다: 이미 분석된 사진(같은 model_version)은 건너뛴다. `force`로 전량 재계산.
한 장이 실패해도 잡을 죽이지 않는다 — `failed`에 모아 돌려준다.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field

import numpy as np

from photoselect.analyze import cluster, represent
from photoselect.config import MODEL_VERSION, Settings
from photoselect.gallery import PhotoRef
from photoselect.store import PhotoAnalysis, Store

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    gallery: str
    targets: int = 0
    processed: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    vlm_used: bool = False
    clusters: int = 0
    similarity_profile: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    per_stage_seconds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gallery": self.gallery, "targets": self.targets, "processed": self.processed,
            "skipped": self.skipped, "failed": self.failed, "vlmUsed": self.vlm_used,
            "clusters": self.clusters, "similarityProfile": {k: round(v, 3) for k, v in self.similarity_profile.items()},
            "elapsedSeconds": round(self.elapsed_seconds, 1),
            "perStageSeconds": {k: round(v, 1) for k, v in self.per_stage_seconds.items()},
        }


def _percentile(values: list[float]) -> list[float]:
    """갤러리 내 백분위 0~100. NaN은 중앙(50)으로 — 없는 신호가 순위를 흔들면 안 된다."""
    arr = np.array(values, dtype=float)
    ok = ~np.isnan(arr)
    out = np.full(len(arr), 50.0)
    if ok.sum() > 1:
        ranks = np.argsort(np.argsort(arr[ok]))
        out[ok] = 100.0 * ranks / (ok.sum() - 1)
    return out.tolist()


def run(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings,
        force: bool = False, use_vlm: bool = True) -> dict:
    started = time.monotonic()
    result = RunResult(gallery=gallery, targets=len(refs))
    knobs = settings.analyze
    stage = {}

    # 재개: 같은 버전으로 이미 분석된 사진은 건너뛴다
    previous = {} if force else {r.photo_id: r for r in store.read_analysis(gallery) if r.model_version == MODEL_VERSION}
    prev_ids, prev_emb = store.read_embeddings(gallery)
    prev_emb_map = dict(zip(prev_ids, prev_emb)) if len(prev_ids) else {}
    todo = [r for r in refs if r.photo_id not in previous or r.photo_id not in prev_emb_map]
    result.skipped = len(refs) - len(todo)
    log.info("갤러리 %s: 대상 %d장, 이미 분석됨 %d장", gallery, len(refs), result.skipped)

    # ── 모델 로딩 (여기서 한 번) ─────────────────────────────────────────
    t0 = time.monotonic()
    from photoselect.analyze.runners import ArniqaRunner, FacesRunner, LaionRunner
    faces, laion, arniqa = FacesRunner(), LaionRunner(), ArniqaRunner()
    tagger = None
    if use_vlm:
        from photoselect.analyze.vlm import OllamaTagger
        cand = OllamaTagger(knobs.vlm_host, knobs.vlm_model, knobs.vlm_long_edge, knobs.vlm_timeout)
        if cand.available():
            tagger = cand
        else:
            log.warning("VLM(%s)에 연결할 수 없다 — 태그 없이 진행. 이유 문장·커버리지가 빈약해진다", knobs.vlm_host)
    result.vlm_used = tagger is not None
    stage["load"] = time.monotonic() - t0

    # ── 사진별 점수·임베딩 ───────────────────────────────────────────────
    rows: dict[str, PhotoAnalysis] = dict(previous)
    embs: dict[str, np.ndarray] = dict(prev_emb_map)
    t_light = t_vlm = 0.0
    for i, ref in enumerate(todo, 1):
        try:
            t0 = time.monotonic()
            f = faces.score(ref.path)
            emb = laion.embed(ref.path)
            aes = laion.score_from_embedding(emb)
            tech = arniqa.score(ref.path)["technical_score"]
            t_light += time.monotonic() - t0

            row = PhotoAnalysis(
                photo_id=ref.photo_id,
                face_boxes={"face_count": f["face_count"], "max_face_ratio": f["max_face_ratio"]},
                sub_scores={"technical_score": tech, "aesthetic_score": aes,
                            "eyes_open": f["eyes_open"], "smile": f["smile"]},
                model_version=MODEL_VERSION,
            )
            if tagger is not None:
                t0 = time.monotonic()
                tags = tagger.tag(ref.path)
                t_vlm += time.monotonic() - t0
                row.scene, row.framing, row.lighting = tags["scene"], tags["framing"], tags["lighting"]
                row.expression, row.subjects, row.caption = tags["expression"], tags["subjects"], tags["caption"]
            rows[ref.photo_id] = row
            embs[ref.photo_id] = emb
            result.processed += 1
            if i % 10 == 0 or i == len(todo):
                log.info("  %d/%d  (경량 %.2fs/장, VLM %.2fs/장)", i, len(todo),
                         t_light / i, t_vlm / i if tagger else 0.0)
        except Exception as exc:  # noqa: BLE001 — 한 장 실패가 잡을 죽이면 안 된다
            log.exception("사진 실패 %s: %s", ref.photo_id, exc)
            result.failed.append(ref.photo_id)
    stage["lightweight"] = t_light
    stage["vlm"] = t_vlm

    # ── 갤러리 단위 후처리: 순서 고정 → 백분위 → 클러스터 → 대표 ────────────
    t0 = time.monotonic()
    ordered = [rows[r.photo_id] for r in refs if r.photo_id in rows]
    ids = [r.photo_id for r in ordered]
    E = np.stack([embs[i] for i in ids]) if ids else np.zeros((0, 768))

    tech_pct = _percentile([r.sub_scores.get("technical_score", math.nan) for r in ordered])
    aes_pct = _percentile([r.sub_scores.get("aesthetic_score", math.nan) for r in ordered])
    for r, tp, ap in zip(ordered, tech_pct, aes_pct):
        r.technical_pct, r.aesthetic_pct = tp, ap

    cids = cluster.cluster_bursts(E, knobs.cluster_threshold, knobs.cluster_window) if len(E) else np.zeros(0, int)
    for r, c in zip(ordered, cids):
        r.cluster_id = int(c)
    represent.assign_ranks(ordered)
    result.clusters = int(cids.max()) + 1 if len(cids) else 0
    result.similarity_profile = cluster.similarity_profile(E, knobs.cluster_window) if len(E) > 1 else {}
    stage["postprocess"] = time.monotonic() - t0

    store.write_analysis(gallery, ordered, (ids, E))
    result.per_stage_seconds = stage
    result.elapsed_seconds = time.monotonic() - started
    return result.to_dict()
