"""테스트 공용 — 합성 갤러리(_world) · 가짜 LLM · Knobs 바꿔 끼우기. 모델 없음, torch 없음, DB 없음."""

from __future__ import annotations

import numpy as np
from PIL import Image

from categorize.config.settings import MODEL_VERSION, Knobs, Settings
from categorize.domain.analysis import PhotoAnalysis
from categorize.repository.local import LocalStore
from categorize.service import concept
from categorize.service.pipeline import assign_ranks, concat_space, percentile


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def world(tmp_path, n_groups=3, per_group=10, seed=0, clip_parent="실내 스튜디오", bg=None):
    """그룹마다 중심 벡터 + 노이즈. E(임베더)·C(CLIP)는 같은 그룹 구조를 공유한다.
    clip_parent 는 score 가 사진마다 저장하는 부모 검증 라벨(sub_scores.clip_parent)이다.
    bg(group, j) 를 주면 그 값을 sub_scores.bg_luma 로 싣는다(score #117) — None 이면 아예 없는 갤러리다."""
    rng = np.random.default_rng(seed)
    dim = 32
    centers = [unit(rng.normal(size=dim)) for _ in range(n_groups)]
    rows, E, C = [], [], []
    for g, c in enumerate(centers):
        for j in range(per_group):
            E.append(unit(c + 0.08 * rng.normal(size=dim)))
            C.append(unit(c + 0.08 * rng.normal(size=dim)))
            pid = f"g{g}-{j:02d}"
            rows.append(PhotoAnalysis(
                photo_id=pid, subjects="couple",
                sub_scores={"technical_score": rng.uniform(0.3, 0.8),
                            "aesthetic_score": rng.uniform(5, 6.5),
                            "sharpness": rng.uniform(50, 500),
                            "clip_parent": clip_parent},
                model_version=MODEL_VERSION))
            if bg is not None:
                rows[-1].sub_scores["bg_luma"] = float(bg(g, j))
    E, C = np.stack(E), np.stack(C)
    for r, t in zip(rows, percentile([r.sub_scores["technical_score"] for r in rows])):
        r.technical_pct = t
    X = concat_space(E, C)
    gids, _ = concept.concept_groups(X, 0.4, min_groups=2, max_share=0.6)
    for i, (r, g) in enumerate(zip(rows, gids)):
        r.embed_group_id = int(g)
        r.cluster_id = i          # 연사 없음 — 전부 단독 클러스터
    assign_ranks(rows)

    # 대표 사진용 실제 JPEG (naming 이 jpeg_bytes 를 부른다)
    img_root = tmp_path / "dataset"
    img_root.mkdir(exist_ok=True)
    for r in rows:
        Image.new("RGB", (32, 32), (200, 180, 160)).save(img_root / f"{r.photo_id}.jpg")
        r.photo_id = r.photo_id + ".jpg"    # LocalStore.preview_path 는 dataset_root/photo_id
    ids = [r.photo_id for r in rows]

    store = LocalStore(tmp_path / "out", dataset_root=img_root)
    store.write_analysis("g", rows, (ids, E), (ids, C))
    settings = Settings(out_root=tmp_path / "out", dataset_root=img_root)
    return store, rows, E, C, settings


def with_knobs(settings, **kw):
    return Settings(out_root=settings.out_root, dataset_root=settings.dataset_root, knobs=Knobs(**kw), llm=settings.llm)


class FakeLlm:
    """청크 vision 호출은 그룹마다 이름을, 통합 호출은 concept 표기를 바꿔 돌려준다."""

    def __init__(self, parent="실내 스튜디오", low_conf_gid=None):
        self.calls: list[tuple[str, bool]] = []
        self.parent = parent
        self.low_conf_gid = low_conf_gid

    def complete_json(self, system, user, schema, max_tokens):
        has_image = not isinstance(user, str) and any(k == "image" for k, _ in user)
        kind = "vision" if has_image else "merge"
        self.calls.append((kind, has_image))
        if kind == "vision":
            gids = [int(t.split("]")[0].split()[1]) for k, t in user
                    if k == "text" and t.startswith("[그룹")]
            return {"groups": [
                {"group_id": g, "parent": self.parent, "proposed_parent": None,
                 "concept": f"세트{g}",
                 "confidence": 0.3 if g == self.low_conf_gid else 0.95}
                for g in gids]}
        out = []
        for ln in user.splitlines():
            gid = int(ln.split(":")[0].split()[1])
            out.append({"group_id": gid, "parent": self.parent, "proposed_parent": None,
                        "concept": f"세트{gid}(통일)"})
        return {"groups": out}


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
