"""테스트 공용 — 합성 갤러리(world) · 가짜 LLM · Knobs 바꿔 끼우기. 모델 없음, torch 없음, DB 없음."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
from PIL import Image

from categorize.config.settings import PIPELINE_VERSION, Knobs, Settings
from categorize.domain.analysis import PhotoAnalysis
from categorize.domain.photo import PhotoRef
from categorize.repository.local import LocalStore
from categorize.service.naming import CONCEPT_SYSTEM, DETAIL_SYSTEM
from categorize.service.pipeline import percentile

T0 = dt.datetime(2026, 4, 15, 12, 0, 0)


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@dataclass
class World:
    store: LocalStore
    rows: list[PhotoAnalysis]      # 촬영 시각순
    refs: list[PhotoRef]
    E: np.ndarray
    C: np.ndarray
    settings: Settings
    concept_of: list[int]          # 사진별 정답 컨셉
    set_of: list[int]              # 사진별 정답 세트(갤러리 전체에서 유일)


def world(tmp_path, order=(0, 1, 2), sets=1, per_set=10, seed=0, timed=True, gap_s=900, shot_s=5) -> World:
    """컨셉마다 중심 + 세트마다 중심에서 벗어난 방향 + 작은 노이즈. E·C 는 같은 구조를 공유한다.

    order = 촬영 순서의 컨셉 번호 — (0, 1, 0) 이면 컨셉 0 을 나중에 다시 찍는다. 블록 사이 공백 gap_s, 블록 안 간격 shot_s.
    timed=False 면 taken_at 이 없다(이미지 모드).
    """
    rng = np.random.default_rng(seed)
    dim = 32
    n_concepts = max(order) + 1
    concept_c = [unit(rng.normal(size=dim)) for _ in range(n_concepts)]
    set_c = {(c, s): unit(concept_c[c] + 0.8 * unit(rng.normal(size=dim))) for c in range(n_concepts) for s in range(sets)}
    rows, refs, E, C, concept_of, set_of = [], [], [], [], [], []
    t = T0
    for b, c in enumerate(order):
        if b:
            t += dt.timedelta(seconds=gap_s)
        for s in range(sets):
            for j in range(per_set):
                center = set_c[(c, s)]
                E.append(unit(center + 0.02 * rng.normal(size=dim)))
                C.append(unit(center + 0.02 * rng.normal(size=dim)))
                pid = f"b{b}-c{c}-s{s}-{j:02d}.jpg"
                sub = {"technical_score": rng.uniform(0.3, 0.8), "aesthetic_score": rng.uniform(5, 6.5),
                       "sharpness": rng.uniform(50, 500)}
                rows.append(PhotoAnalysis(photo_id=pid, subjects="couple", sub_scores=sub, pipeline_version=PIPELINE_VERSION))
                refs.append(PhotoRef(photo_id=pid, path=None, taken_at=t if timed else None, camera="A"))
                concept_of.append(c)
                set_of.append(c * sets + s)
                t += dt.timedelta(seconds=shot_s)
    E, C = np.stack(E), np.stack(C)
    for r, v in zip(rows, percentile([r.sub_scores["technical_score"] for r in rows])):
        r.technical_pct = v

    img_root = tmp_path / "dataset"
    img_root.mkdir(exist_ok=True)
    for r in rows:
        Image.new("RGB", (32, 32), (200, 180, 160)).save(img_root / r.photo_id)
    ids = [r.photo_id for r in rows]
    store = LocalStore(tmp_path / "out", dataset_root=img_root)
    store.write_analysis("g", rows, (ids, E), (ids, C))
    settings = Settings(out_root=tmp_path / "out", dataset_root=img_root)
    return World(store, rows, refs, E, C, settings, concept_of, set_of)


def with_knobs(settings, **kw):
    return Settings(out_root=settings.out_root, dataset_root=settings.dataset_root, knobs=Knobs(**kw), llm=settings.llm)


def _labels(user, prefix: str) -> list[int]:
    return [int(t.split("]")[0].split()[1]) for k, t in user if k == "text" and t.startswith(prefix)]


class FakeLlm:
    """1층 호출: 구간 → merge(구간) 번호의 컨셉. 2층 호출: 그룹마다 `세트{id}`.

    merge 기본값은 구간마다 자기 컨셉. names 로 컨셉 이름을 바꿀 수 있다(중복 이름 시험용)."""

    def __init__(self, merge=None, names=None, concept_conf=0.95, drop_segments=()):
        self.merge = merge or (lambda u: u)
        self.names = names or (lambda c: f"컨셉{c}")
        self.concept_conf = concept_conf
        self.drop_segments = set(drop_segments)
        self.calls: list[tuple[str, str]] = []      # (kind, 첫 텍스트)

    def complete_json(self, system, user, schema, max_tokens):
        head = next(t for k, t in user if k == "text")
        if system == CONCEPT_SYSTEM:
            self.calls.append(("concept", head))
            by: dict[int, list[int]] = {}
            for u in _labels(user, "[구간"):
                if u not in self.drop_segments:
                    by.setdefault(self.merge(u), []).append(u)
            return {"concepts": [{"name": self.names(c), "segments": us, "confidence": self.concept_conf}
                                 for c, us in sorted(by.items())]}
        assert system == DETAIL_SYSTEM
        self.calls.append(("detail", head))
        return {"groups": [{"group_id": g, "detail": f"세트{g}", "confidence": 0.95}
                           for g in _labels(user, "[그룹")]}

    def kinds(self) -> list[str]:
        return [k for k, _ in self.calls]


class CountingStore:
    """LocalStore 를 감싸 read_gallery · preview_paths 호출을 센다."""

    def __init__(self, inner):
        self.inner = inner
        self.reads = 0
        self.preview_calls: list[list[str]] = []

    def read_gallery(self, gallery):
        self.reads += 1
        return self.inner.read_gallery(gallery)

    def preview_paths(self, gallery, photo_ids):
        self.preview_calls.append(list(photo_ids))
        return self.inner.preview_paths(gallery, photo_ids)

    def __getattr__(self, name):
        return getattr(self.inner, name)
