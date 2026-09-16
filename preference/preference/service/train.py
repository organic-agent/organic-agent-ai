"""학습 — 클러스터 단위 양성·음성 → 두 강도 L2 로지스틱 회귀 (풀이는 `infrastructure/solver.py`).

라벨 규칙 (plan §1):
  양성   부부가 고른 사진. 한 클러스터에 둘 이상이면 가중치 1/count — 클러스터당 양성 1
  음성   양성이 없는 클러스터의 대표 1장 (cluster_rank 0, 없으면 첫 행)
  제외   양성의 연사 형제 — 거의 같은 벡터에 반대 라벨을 주면 없는 경계를 그리게 된다
클래스 가중치는 양성·음성 합이 같게(balanced).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from preference.config.settings import Knobs
from preference.domain.features import N_EMB, N_SCALAR, Features
from preference.domain.gallery import LabeledGallery
from preference.domain.model import PreferenceModel, lam
from preference.infrastructure.solver import fit
from preference.service.features import build

log = logging.getLogger(__name__)


@dataclass
class Sample:
    idx: np.ndarray     # 쓰는 행
    y: np.ndarray       # 1/0
    w: np.ndarray       # 표본 가중치 (양성·음성 합 각각 0.5)


def make_sample(lg: LabeledGallery) -> Sample:
    gd = lg.data
    pos = lg.positive_idx
    if pos.size == 0:
        raise ValueError(f"갤러리 {gd.gallery_id}: 선택 사진이 재료 안에 없다")
    cid = gd.cluster_id.copy()
    solo = cid < 0
    cid[solo] = -(np.arange(solo.sum()) + 1)            # 단독은 자기만의 클러스터
    pos_clusters = set(cid[pos].tolist())

    # 양성: 클러스터당 합 1
    pos_w = np.zeros(len(pos))
    for c in pos_clusters:
        members = [k for k, i in enumerate(pos) if cid[i] == c]
        for k in members:
            pos_w[k] = 1.0 / len(members)

    # 음성: 양성 클러스터 밖의 대표 1장
    neg: list[int] = []
    order = np.lexsort((np.arange(gd.n), gd.cluster_rank))   # rank 0 우선, 같으면 행 순
    seen: set[int] = set()
    for i in order:
        c = int(cid[i])
        if c in pos_clusters or c in seen:
            continue
        seen.add(c)
        neg.append(int(i))
    neg_a = np.array(neg, dtype=int)
    if neg_a.size == 0:
        raise ValueError(f"갤러리 {gd.gallery_id}: 음성이 없다 (모든 클러스터에 선택이 있다)")

    idx = np.concatenate([pos, neg_a])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg_a))])
    w = np.concatenate([pos_w / pos_w.sum() * 0.5, np.full(len(neg_a), 0.5 / len(neg_a))])
    return Sample(idx=idx, y=y, w=w)


def train(galleries: list[LabeledGallery], knobs: Knobs,
          features: dict[str, Features] | None = None) -> PreferenceModel:
    """여러 갤러리를 합쳐 w 하나 — 부부 평균 취향. 갤러리마다 표본 가중치 합이 1 이라 큰 갤러리가 지배하지 않는다."""
    if not galleries:
        raise ValueError("학습 갤러리가 없다")
    feats = features or {}
    xs_l, xe_l, y_l, w_l = [], [], [], []
    n_pos = 0
    for lg in galleries:
        f = feats.get(lg.data.gallery_id) or build(lg.data)
        s = make_sample(lg)
        xs_l.append(f.scalar[s.idx]); xe_l.append(f.emb[s.idx]); y_l.append(s.y); w_l.append(s.w)
        n_pos += int(s.y.sum())
    xs, xe, y, w = (np.concatenate(a) for a in (xs_l, xe_l, y_l, w_l))
    assert xs.shape[1] == N_SCALAR and xe.shape[1] == N_EMB
    w_s, w_e, b = fit(xs, xe, y, w, knobs)
    first = galleries[0].data
    return PreferenceModel(
        w_scalar=w_s, w_emb=w_e, bias=b,
        n_galleries=len(galleries), n_positives=n_pos,
        train_gallery_ids=[lg.data.gallery_id for lg in galleries],
        embedding_model=first.embedding_model, model_version=first.model_version,
        lam=lam(len(galleries), knobs.n0),
    )
