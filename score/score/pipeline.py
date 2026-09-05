"""SCORE 본체 — 사진별 점수. 갤러리 전수, CPU, torch. VLM 없음.

    ARNIQA(spaq)                → technical_score
    CLIP ViT-L/14 + LAION MLP   → aesthetic_score
        CLIP 벡터는 저장한다 (photo_analysis.clip_embedding) — categorize 가 재계산 없이 읽는다
        같은 벡터에 텍스트 프롬프트를 대어 subjects(피사체)·clip_parent(부모 검증 라벨)도 여기서
    고전 지표                    → sharpness · highlight_clip · shadow_clip · mean_luma (sub_scores)
    → store.write_scores  (subjects · sub_scores · clip_embedding · model_version 만)

한 장은 한 번만 디코드해서(1600px PIL) 세 러너에 넘기고, CLIP 은 `clip_batch` 장씩 한 forward 로 묶는다(#51).
배치 CLIP 이 실패하면 그 묶음만 한 장씩으로 물러난다 — 한 장 실패가 묶음·잡을 죽이지 않는다.
사진마다 독립이라 재개가 쉽다: 같은 MODEL_VERSION 이고 CLIP 벡터가 저장된 사진은 건너뛴다.
`write_batch` 장마다 commit 하고, `remaining_seconds` 가 있으면(Lambda) 배치 경계에서 데드라인을 보고 멈춘다 —
결과의 `stopped`·`remaining` 으로 드러나고, 재호출은 handler 의 몫이다(embedder #24 와 같은 규칙).
임베더(DINOv3) 벡터 유무는 보지 않는다 — 그건 categorize 의 입력 조건이다.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import numpy as np

from score.config import MODEL_VERSION, Settings
from score.gallery import PhotoRef
from score.images import load_image
from score.store import PhotoAnalysis, Store

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
    #: 데드라인 때문에 배치 경계에서 멈췄다. 남은 사진은 remaining.
    stopped: bool = False
    remaining: int = 0
    subjects_used: bool = False
    elapsed_seconds: float = 0.0
    per_stage_seconds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gallery": self.gallery, "pipeline": self.pipeline, "mode": self.mode,
            "targets": self.targets, "processed": self.processed, "skipped": self.skipped,
            "failed": self.failed, "stopped": self.stopped, "remaining": self.remaining,
            "subjectsUsed": self.subjects_used,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
            "perStageSeconds": {k: round(v, 1) for k, v in self.per_stage_seconds.items()},
        }


def _compute_env() -> str:
    """torch 스레드 · CPU 수 — Lambda 에서 실제로 몇 코어를 쓰는지 로그로 남긴다(#51)."""
    import os

    try:
        import torch

        return f"torch_threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()} cpu_count={os.cpu_count()}"
    except Exception:  # noqa: BLE001 — 테스트의 가짜 러너 환경
        return f"cpu_count={os.cpu_count()}"


def _load_runners():
    """torch 러너는 여기서만 import 한다 — 재개 판정만 하고 끝나는 호출은 모델을 올리지 않는다.
    `TORCH_NUM_THREADS` 가 있으면 torch 스레드 수를 그 값으로 — Lambda 의 cpu_count 와 실제 vCPU 가 다를 때 실험용(#51)."""
    import os

    threads = os.environ.get("TORCH_NUM_THREADS")
    if threads:
        import torch

        torch.set_num_threads(int(threads))
    from score import classical
    from score.runners import ArniqaRunner, LaionRunner
    from score.subjects import ParentTagger, SubjectsTagger
    return classical, ArniqaRunner, LaionRunner, ParentTagger, SubjectsTagger


def _as_dt(value) -> datetime | None:
    """DB 는 timestamptz(datetime), 로컬은 ISO 문자열. tz 없는 값은 UTC 로 본다."""
    if value is None:
        return None
    dt = datetime.fromisoformat(value) if isinstance(value, str) else value
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def run(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings, force: bool = False,
        remaining_seconds: Callable[[], float] | None = None, since: datetime | None = None) -> dict:
    """`since` 가 있으면 그 시각 이후에 쓴 점수만 "있음"으로 친다 — force 실행의 시작 시각을 재호출·샤드에 넘겨,
    force 를 잃어도 이번 실행 전 점수는 다시 계산한다(#54). force 는 since 없는 로컬 전체 재계산."""
    started = time.monotonic()
    k = settings.knobs
    result = ScoreResult(gallery=gallery, targets=len(refs))
    stage: dict[str, float] = {}

    def fresh(r: PhotoAnalysis) -> bool:
        if r.model_version != MODEL_VERSION:
            return False
        if since is None:
            return True
        at = _as_dt(r.analyzed_at)
        return at is not None and at >= since

    previous = {} if force else {r.photo_id: r for r in store.read_analysis(gallery) if fresh(r)}
    prev_clip_ids, _ = store.read_clip_embeddings(gallery)
    prev_clip = set() if force else set(prev_clip_ids)
    todo = [r for r in refs if r.photo_id not in previous or r.photo_id not in prev_clip]
    result.skipped = len(refs) - len(todo)
    log.info("[score] 갤러리 %s: 대상 %d장, 이미 점수 있음 %d장", gallery, len(refs), result.skipped)
    if not todo:
        result.elapsed_seconds = time.monotonic() - started
        return result.to_dict()

    t0 = time.monotonic()
    classical, ArniqaRunner, LaionRunner, ParentTagger, SubjectsTagger = _load_runners()
    laion, arniqa = LaionRunner(), ArniqaRunner(long_edge=k.arniqa_long_edge)
    tagger = SubjectsTagger(laion) if k.subjects_zero_shot else None
    parent_tagger = ParentTagger(laion)
    result.subjects_used = tagger is not None
    stage["load"] = time.monotonic() - t0
    log.info("[score] 러너 로드 %.1fs · %s", stage["load"], _compute_env())
    #: 장별 누적 시간 — 어디서 시간이 가는지 Lambda 로그로 본다(#51). decode+clip 은 묶음 단위라 묶음 시간을 장수로 나눈다.
    t_stage = {"decode": 0.0, "clip": 0.0, "arniqa": 0.0, "classical": 0.0, "tag": 0.0, "write": 0.0}

    def flush(rows: list[PhotoAnalysis], clips: dict[str, np.ndarray]) -> None:
        ids = [r.photo_id for r in rows]
        C = np.stack([clips[i] for i in ids]) if ids else np.zeros((0, 768))
        store.write_scores(gallery, rows, (ids, C))

    def embed_chunk(chunk: list[PhotoRef]) -> list[tuple[PhotoRef, object, np.ndarray | None, Exception | None]]:
        """(ref, 이미지, CLIP 벡터, 오류) — 디코드는 한 번, CLIP 은 묶어서. 실패한 장은 오류를 들고 나온다."""
        loaded: list[tuple[PhotoRef, object]] = []
        out: dict[str, tuple] = {}
        t = time.monotonic()
        for ref in chunk:
            try:
                loaded.append((ref, load_image(ref.path)))
            except Exception as exc:  # noqa: BLE001
                out[ref.photo_id] = (ref, None, None, exc)
        t_stage["decode"] += time.monotonic() - t
        t = time.monotonic()
        if loaded:
            try:
                embs = laion.embed_batch([img for _, img in loaded])
                for (ref, img), emb in zip(loaded, embs):
                    out[ref.photo_id] = (ref, img, emb, None)
            except Exception as exc:  # noqa: BLE001 — 묶음이 죽으면 한 장씩 물러난다
                log.warning("CLIP 배치 %d장 실패(%s) — 한 장씩 재시도", len(loaded), exc)
                for ref, img in loaded:
                    try:
                        out[ref.photo_id] = (ref, img, laion.embed(img), None)
                    except Exception as exc1:  # noqa: BLE001
                        out[ref.photo_id] = (ref, img, None, exc1)
        t_stage["clip"] += time.monotonic() - t
        return [out[ref.photo_id] for ref in chunk]

    rows: list[PhotoAnalysis] = []
    clips: dict[str, np.ndarray] = {}
    t_photo = 0.0
    longest_batch = 0.0
    batch_started = time.monotonic()
    done = 0
    i = 0
    chunk_size = max(1, k.clip_batch)
    for start in range(0, len(todo), chunk_size):
        t_chunk = time.monotonic()
        prepared = embed_chunk(todo[start:start + chunk_size])
        t_shared = (time.monotonic() - t_chunk) / max(1, len(prepared))   # 디코드+CLIP 몫을 장별로 나눈다
        for ref, img, clip_emb, err in prepared:
            i += 1
            t0 = time.monotonic()
            try:
                if err is not None:
                    raise err
                t = time.monotonic()
                aes = laion.score_from_embedding(clip_emb)
                tech = arniqa.score(img)["technical_score"]
                t_stage["arniqa"] += time.monotonic() - t
                t = time.monotonic()
                cl = classical.measure(img)
                t_stage["classical"] += time.monotonic() - t
                sub = {"technical_score": tech, "aesthetic_score": aes, **cl}
                subjects = UNKNOWN
                t = time.monotonic()
                if tagger is not None:
                    subjects, margin = tagger.tag(clip_emb)
                    sub["subjects_margin"] = margin
                sub["clip_parent"] = parent_tagger.tag(clip_emb)
                t_stage["tag"] += time.monotonic() - t
                rows.append(PhotoAnalysis(photo_id=ref.photo_id, subjects=subjects,
                                          sub_scores=sub, model_version=MODEL_VERSION))
                clips[ref.photo_id] = clip_emb
                result.processed += 1
                t_photo += time.monotonic() - t0 + t_shared
                if i % 20 == 0 or i == len(todo):
                    log.info("  %d/%d  (%.2fs/장)  %s", i, len(todo), t_photo / i,
                             " ".join(f"{k}={v / i:.2f}" for k, v in t_stage.items()))
            except Exception as exc:  # noqa: BLE001 — 한 장 실패가 잡을 죽이면 안 된다
                log.exception("사진 실패 %s: %s", ref.photo_id, exc)
                result.failed.append(ref.photo_id)
            done = i

            if len(rows) >= k.write_batch or i == len(todo):
                t = time.monotonic()
                flush(rows, clips)
                t_stage["write"] += time.monotonic() - t
                rows, clips = [], {}
                longest_batch = max(longest_batch, time.monotonic() - batch_started)
                batch_started = time.monotonic()
                if remaining_seconds is not None and i < len(todo):
                    budget = longest_batch + settings.stop_margin_seconds
                    left = remaining_seconds()
                    if left < budget:
                        result.stopped = True
                        log.warning("[score] 갤러리 %s: 남은 시간 %.0fs < 배치 예산 %.0fs — 배치 경계에서 멈춤 (남은 %d장)",
                                    gallery, left, budget, len(todo) - i)
                        break
        if result.stopped:
            break

    result.remaining = len(todo) - done
    stage["photos"] = t_photo
    stage.update(t_stage)
    result.per_stage_seconds = stage
    result.elapsed_seconds = time.monotonic() - started
    return result.to_dict()
