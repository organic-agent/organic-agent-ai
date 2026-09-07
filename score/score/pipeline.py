"""SCORE 본체 — 사진별 점수. 갤러리 전수, CPU, torch. VLM 없음.

    ARNIQA(spaq)                → technical_score
    CLIP ViT-L/14 + LAION MLP   → aesthetic_score
        CLIP 벡터는 저장한다 (photo_analysis.clip_embedding) — categorize 가 재계산 없이 읽는다
        같은 벡터에 텍스트 프롬프트를 대어 subjects(피사체)·clip_parent(부모 검증 라벨)도 여기서
    고전 지표                    → sharpness · highlight_clip · shadow_clip · mean_luma (sub_scores)
    → store.write_scores  (subjects · sub_scores · clip_embedding · model_version 만)

한 장은 한 번만 디코드해서(1024px PIL) 세 러너에 넘기고, CLIP 은 `clip_batch` 장씩, ARNIQA 는 `arniqa_batch` 장씩 한 forward 로
묶는다(#51 · #68). 배치가 실패하면 그 묶음만 한 장씩으로 물러난다 — 한 장 실패가 묶음·잡을 죽이지 않는다.
GPU(#68)에서는 디코드·classical 을 `decode_workers` 스레드가 두 묶음 앞서 준비한다 — 그러지 않으면 GPU 가 4 vCPU 의 JPEG
디코드를 기다리며 논다. Lambda(CPU) 는 0 이라 예전처럼 한 스레드에서 순서대로.
사진마다 독립이라 재개가 쉽다: 같은 MODEL_VERSION 이고 CLIP 벡터가 저장된 사진은 건너뛴다.
`write_batch` 장마다 commit 하고, `remaining_seconds` 가 있으면(Lambda) 배치 경계에서 데드라인을 보고 멈춘다 —
결과의 `stopped`·`remaining` 으로 드러나고, 재호출은 handler 의 몫이다(embedder #24 와 같은 규칙).
임베더(DINOv3) 벡터 유무는 보지 않는다 — 그건 categorize 의 입력 조건이다.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
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


#: 디코드 스레드가 앞서 준비해 두는 묶음 수. 묶음 = clip_batch 장. 2 면 메모리의 PIL 이미지는 세 묶음을 넘지 않는다.
PREFETCH_CHUNKS = 2


def _compute_env(device: str | None = None) -> str:
    """torch 스레드 · CPU 수 · 장치 — Lambda 에서 실제로 몇 코어를 쓰는지, GPU 면 어떤 GPU 인지 로그로 남긴다(#51 · #68)."""
    import os

    try:
        import torch

        from score.device import describe

        dev = f" device={describe(device)}" if device else ""
        return f"torch_threads={torch.get_num_threads()} interop={torch.get_num_interop_threads()} cpu_count={os.cpu_count()}{dev}"
    except Exception:  # noqa: BLE001 — 테스트의 가짜 러너 환경
        return f"cpu_count={os.cpu_count()} device={device}"


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


class Scorer:
    """러너·태거를 한 번 올려 여러 배치에 재사용한다(#75). Lambda 는 호출마다 하나, GPU 워커는 프로세스에 하나.

    `run()` 이 대상 선정(재개 판정)을 하고 `score()` 가 계산·적재를 한다 — 워커는 집기(claim)로 대상이 이미 정해져 있어
    `run(force=True, scorer=…)` 로 부른다."""

    def __init__(self, settings: Settings) -> None:
        k = settings.knobs
        t0 = time.monotonic()
        classical, ArniqaRunner, LaionRunner, ParentTagger, SubjectsTagger = _load_runners()
        self.classical = classical
        self.laion = LaionRunner(device=k.device, fp16=k.fp16)
        self.arniqa = ArniqaRunner(long_edge=k.arniqa_long_edge, device=k.device, fp16=k.fp16)
        self.device = getattr(self.laion, "device", None)
        self.tagger = SubjectsTagger(self.laion) if k.subjects_zero_shot else None
        self.parent_tagger = ParentTagger(self.laion)
        self.load_seconds = time.monotonic() - t0
        log.info("[score] 러너 로드 %.1fs · %s · clip_batch=%d arniqa_batch=%d fp16=%s decode_workers=%d",
                 self.load_seconds, _compute_env(self.device), k.clip_batch, k.arniqa_batch, k.fp16, k.decode_workers)

    def warm_up(self) -> float:
        """cuda 일 때 더미 1장을 한 번 돌린다(#79) — cuDNN 자동 튜닝·가중치 페이지 캐시를 첫 실제 배치 앞에서 치른다.
        cpu·mps 는 건너뛴다. 실패해도 예외를 내지 않는다(워밍은 최적화일 뿐)."""
        if self.device != "cuda":
            return 0.0
        t0 = time.monotonic()
        try:
            from PIL import Image

            dummy = Image.new("RGB", (683, 1024), (128, 128, 128))
            self.laion.embed_batch([dummy])
            self.arniqa.score_batch([dummy])
        except Exception as exc:  # noqa: BLE001
            log.warning("[score] 워밍 실패 (무시): %s", exc)
        took = time.monotonic() - t0
        log.info("[score] 워밍 %.1fs", took)
        return took

    def score(self, store: Store, gallery: str, todo: list[PhotoRef], settings: Settings, result: "ScoreResult",
              stage: dict[str, float], remaining_seconds: Callable[[], float] | None = None) -> None:
        """`todo` 를 계산해 store 에 쓴다. result·stage 를 채운다(호출자가 만든 것)."""
        k = settings.knobs
        classical, laion, arniqa, tagger, parent_tagger = self.classical, self.laion, self.arniqa, self.tagger, self.parent_tagger
        result.subjects_used = tagger is not None
        stage["load"] = self.load_seconds
        #: 장별 누적 시간 — 어디서 시간이 가는지 로그로 본다(#51). decode·clip·arniqa 는 묶음 단위라 묶음 시간을 장수로 나눈다.
        #: 프리페치(decode_workers > 0)면 decode 는 "묶음이 준비되길 기다린 벽시계", classical 은 스레드 안에서 잰 CPU 시간이라
        #: 추론과 겹친다 — 둘을 더해 장당 시간으로 읽으면 안 된다.
        t_stage = {"decode": 0.0, "clip": 0.0, "arniqa": 0.0, "classical": 0.0, "tag": 0.0, "write": 0.0}

        def flush(rows: list[PhotoAnalysis], clips: dict[str, np.ndarray]) -> None:
            ids = [r.photo_id for r in rows]
            C = np.stack([clips[i] for i in ids]) if ids else np.zeros((0, 768))
            store.write_scores(gallery, rows, (ids, C))

        def prepare(ref: PhotoRef) -> tuple[object, dict, float, object, object]:
            """CPU 만 쓰는 준비 — 디코드 1회 + classical + 러너별 전처리(CLIP 224 · ARNIQA uint8). 스레드에서 돈다 —
            GPU forward 를 제외한 장당 CPU 일이 전부 여기 있어야 메인 스레드(GPU)가 기다리지 않는다(#68)."""
            img = load_image(ref.path)
            t = time.monotonic()
            cl = classical.measure(img)
            t_cl = time.monotonic() - t
            return img, cl, t_cl, laion.prepare(img), arniqa.prepare(img)

        def embed_chunk(chunk: list[PhotoRef], futures: list[Future] | None) -> list[tuple]:
            """[(ref, 이미지, CLIP 벡터, classical, ARNIQA 점수, 오류)] — 디코드는 한 번, CLIP·ARNIQA 는 묶어서.
            실패한 장은 오류를 들고 나온다."""
            loaded: list[tuple[PhotoRef, object, dict, object]] = []
            arniqa_in: dict[str, object] = {}
            out: dict[str, list] = {}
            t = time.monotonic()
            for j, ref in enumerate(chunk):
                try:
                    img, cl, t_cl, clip_x, arniqa_x = futures[j].result() if futures is not None else prepare(ref)
                    t_stage["classical"] += t_cl
                    loaded.append((ref, img, cl, clip_x))
                    arniqa_in[ref.photo_id] = arniqa_x
                except Exception as exc:  # noqa: BLE001
                    out[ref.photo_id] = [ref, None, None, None, None, exc]
            t_stage["decode"] += time.monotonic() - t
            t = time.monotonic()
            if loaded:
                try:
                    embs = laion.embed_prepared([x for _, _, _, x in loaded])
                    for (ref, img, cl, _), emb in zip(loaded, embs):
                        out[ref.photo_id] = [ref, img, emb, cl, None, None]
                except Exception as exc:  # noqa: BLE001 — 묶음이 죽으면 한 장씩 물러난다
                    log.warning("CLIP 배치 %d장 실패(%s) — 한 장씩 재시도", len(loaded), exc)
                    for ref, img, cl, _ in loaded:
                        try:
                            out[ref.photo_id] = [ref, img, laion.embed(img), cl, None, None]
                        except Exception as exc1:  # noqa: BLE001
                            out[ref.photo_id] = [ref, img, None, cl, None, exc1]
            t_stage["clip"] += time.monotonic() - t
            t = time.monotonic()
            alive = [out[ref.photo_id] for ref in chunk if out[ref.photo_id][5] is None]
            step = max(1, k.arniqa_batch)
            for start in range(0, len(alive), step):
                group = alive[start:start + step]
                try:
                    for entry, tech in zip(group, arniqa.score_prepared([arniqa_in[e[0].photo_id] for e in group])):
                        entry[4] = tech["technical_score"]
                except Exception as exc:  # noqa: BLE001 — 묶음이 죽으면 한 장씩 물러난다
                    log.warning("ARNIQA 배치 %d장 실패(%s) — 한 장씩 재시도", len(group), exc)
                    for entry in group:
                        try:
                            entry[4] = arniqa.score(entry[1])["technical_score"]
                        except Exception as exc1:  # noqa: BLE001
                            entry[5] = exc1
            t_stage["arniqa"] += time.monotonic() - t
            return [tuple(out[ref.photo_id]) for ref in chunk]

        rows: list[PhotoAnalysis] = []
        clips: dict[str, np.ndarray] = {}
        t_photo = 0.0
        longest_batch = 0.0
        batch_started = time.monotonic()
        done = 0
        i = 0
        chunk_size = max(1, k.clip_batch)
        chunks = [todo[start:start + chunk_size] for start in range(0, len(todo), chunk_size)]
        executor = ThreadPoolExecutor(max_workers=k.decode_workers) if k.decode_workers > 0 else None
        pending: dict[int, list[Future]] = {}

        def prefetch(ci: int) -> None:
            if executor is not None and ci < len(chunks) and ci not in pending:
                pending[ci] = [executor.submit(prepare, ref) for ref in chunks[ci]]

        try:
            for ci, chunk in enumerate(chunks):
                for ahead in range(ci, ci + PREFETCH_CHUNKS + 1):
                    prefetch(ahead)
                t_chunk = time.monotonic()
                prepared = embed_chunk(chunk, pending.pop(ci, None))
                t_shared = (time.monotonic() - t_chunk) / max(1, len(prepared))   # 디코드+CLIP+ARNIQA 몫을 장별로 나눈다
                for ref, img, clip_emb, cl, tech, err in prepared:
                    i += 1
                    t0 = time.monotonic()
                    try:
                        if err is not None:
                            raise err
                        aes = laion.score_from_embedding(clip_emb)
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
        finally:
            if executor is not None:
                executor.shutdown(wait=False, cancel_futures=True)

        result.remaining = len(todo) - done
        stage["photos"] = t_photo
        stage.update(t_stage)


def run(store: Store, gallery: str, refs: list[PhotoRef], settings: Settings, force: bool = False,
        remaining_seconds: Callable[[], float] | None = None, since: datetime | None = None,
        scorer: Scorer | None = None) -> dict:
    """`since` 가 있으면 그 시각 이후에 쓴 점수만 "있음"으로 친다 — force 실행의 시작 시각을 재호출·샤드에 넘겨,
    force 를 잃어도 이번 실행 전 점수는 다시 계산한다(#54). force 는 since 없는 로컬 전체 재계산.
    `scorer` 를 주면 러너를 다시 올리지 않는다(#75, GPU 워커) — 없으면 여기서 하나 만든다(대상이 있을 때만)."""
    started = time.monotonic()
    result = ScoreResult(gallery=gallery, targets=len(refs))
    stage: dict[str, float] = {}

    def fresh(r: PhotoAnalysis) -> bool:
        if r.model_version != MODEL_VERSION:
            return False
        if since is None:
            return True
        at = _as_dt(r.analyzed_at)
        return at is not None and at >= since

    if force:
        # 재개 판정을 건너뛴다 — store 를 읽지 않는다(#81: 워커는 gallery 자리에 라벨을 넘기므로 갤러리 조회가 있어선 안 된다)
        previous, prev_clip = {}, set()
    else:
        previous = {r.photo_id: r for r in store.read_analysis(gallery) if fresh(r)}
        prev_clip = set(store.read_clip_embeddings(gallery)[0])
    todo = [r for r in refs if r.photo_id not in previous or r.photo_id not in prev_clip]
    result.skipped = len(refs) - len(todo)
    log.info("[score] 갤러리 %s: 대상 %d장, 이미 점수 있음 %d장", gallery, len(refs), result.skipped)
    if not todo:
        result.elapsed_seconds = time.monotonic() - started
        return result.to_dict()

    scorer = scorer or Scorer(settings)
    scorer.score(store, gallery, todo, settings, result, stage, remaining_seconds=remaining_seconds)
    result.per_stage_seconds = stage
    result.elapsed_seconds = time.monotonic() - started
    return result.to_dict()
