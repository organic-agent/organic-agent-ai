"""학습 — 클러스터 단위 양성·음성 → 두 강도 L2 로지스틱 회귀 (scipy L-BFGS).

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
from scipy.optimize import minimize

from preference.config import Knobs
from preference.features import Features, N_EMB, N_SCALAR, build
from preference.model import PreferenceModel, lam
from preference.store import LabeledGallery

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


def _block_scale(x: np.ndarray) -> float:
    """좌표당 평균 분산 = trace(Cov)/d. 0 이면 1(상수 블록)."""
    v = float(x.var(axis=0).mean())
    return v if v > 1e-12 else 1.0


def fit(xs: np.ndarray, xe: np.ndarray, y: np.ndarray, w: np.ndarray, knobs: Knobs) -> tuple[np.ndarray, np.ndarray, float]:
    """가중 로지스틱 손실 + λ_s‖w_s‖² + λ_e‖w_e‖². bias 는 정규화하지 않는다."""
    ns, ne = xs.shape[1], xe.shape[1]
    X = np.concatenate([xs, xe, np.ones((len(y), 1))], axis=1)
    # 정규화 강도를 블록의 분산 규모로 보정한다 — 임의 방향의 로짓 표준편차가 1 이 되는 ‖w‖ 를 기준 단위로.
    # 임베딩 1536차원은 좌표당 분산이 1/1536 수준이라 보정 없이 l2·‖w‖² 을 쓰면 값의 뜻이 차원 수에 묻힌다.
    reg = np.concatenate([np.full(ns, knobs.l2_scalar * _block_scale(xs)),
                          np.full(ne, knobs.l2_emb * _block_scale(xe)), [0.0]])
    wsum = w.sum()

    def f(theta: np.ndarray) -> tuple[float, np.ndarray]:
        m = X @ theta
        # log(1+e^m) − y·m, 수치 안정
        loss = np.logaddexp(0.0, m) - y * m
        p = 1.0 / (1.0 + np.exp(-m))
        total = (w * loss).sum() / wsum + (reg * theta * theta).sum()
        grad = X.T @ (w * (p - y)) / wsum + 2.0 * reg * theta
        return float(total), grad

    res = minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B", options={"maxiter": knobs.max_iter})
    if not res.success:
        log.warning("L-BFGS 가 수렴 조건을 못 채웠다: %s (iter %d)", res.message, res.nit)
    theta = res.x
    return theta[:ns], theta[ns:ns + ne], float(theta[-1])


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
