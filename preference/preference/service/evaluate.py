"""평가 — recall@K(strict·cluster) · AUC · 세 점수식(prior / pref / 융합) · leave-one-gallery-out · 게이트.

K = k_multiplier × 양성 수. "추천 후보 범위"의 정의다(plan §5). 순위는 **연사 클러스터당 최고점 1장으로 dedup 한 뒤**
매긴다 — wes 추천이 그렇게 하기 때문이다(갤러리 8 실측: 양성 30장의 클러스터가 1,109장이라 dedup 없이는 형제가 상위를 채운다).
strict 는 양성 그 사진이 클러스터 대표로 올라온 경우, cluster 는 그 클러스터가 올라온 경우.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy.stats import binomtest

from preference.config.settings import Knobs
from preference.domain.features import Features
from preference.domain.gallery import LabeledGallery
from preference.domain.model import PreferenceModel, lam, prior, z
from preference.service.features import build
from preference.service.train import make_sample, train

log = logging.getLogger(__name__)

SCORES = ("prior", "pref", "fused")


def dedup_order(scores: np.ndarray, cluster_id: np.ndarray | None) -> np.ndarray:
    """추천이 실제로 보는 순위 — 연사 클러스터당 최고점 1장만 남기고 점수 내림차순 (wes MmrSelector 의 bestPerCluster 와 같다).
    cluster_id 가 없으면 전체 순위."""
    if cluster_id is None:
        return np.argsort(-scores, kind="stable")
    best: dict[int, int] = {}
    solo = 0
    for i in np.argsort(-scores, kind="stable"):
        c = int(cluster_id[i])
        if c < 0:
            c = -1 - solo
            solo += 1
        if c not in best:
            best[c] = int(i)
    return np.array(list(best.values()), dtype=int)


def recalls(scores: np.ndarray, pos: np.ndarray, k: int, cluster_id: np.ndarray | None) -> tuple[float, float]:
    """(strict, cluster). dedup 순위 상위 K 안에 — strict: 양성 그 사진이 대표로 올라왔는가 / cluster: 양성의 클러스터가 올라왔는가."""
    if pos.size == 0:
        return float("nan"), float("nan")
    top = dedup_order(scores, cluster_id)[:k]
    top_set = set(top.tolist())
    strict = sum(1 for p in pos if p in top_set) / pos.size
    if cluster_id is None:
        return float(strict), float(strict)
    top_clusters = {int(cluster_id[i]) for i in top if cluster_id[i] >= 0}
    hit = sum(1 for p in pos if p in top_set or (cluster_id[p] >= 0 and int(cluster_id[p]) in top_clusters))
    return float(strict), float(hit / pos.size)


def recall_at_k(scores: np.ndarray, pos: np.ndarray, k: int, cluster_id: np.ndarray | None = None) -> float:
    """cluster 기준 recall@K (cluster_id 없으면 전체 순위)."""
    return recalls(scores, pos, k, cluster_id)[1]


def auc(scores: np.ndarray, pos: np.ndarray, neg: np.ndarray) -> float:
    """(양성, 음성) 쌍 중 양성 점수가 큰 비율 — 쌍 예측 정확도. 동점은 0.5."""
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    sp, sn = scores[pos][:, None], scores[neg][None, :]
    return float(((sp > sn).sum() + 0.5 * (sp == sn).sum()) / (pos.size * neg.size))


@dataclass
class GalleryEval:
    gallery_id: str
    n_photos: int
    n_positives: int
    k: int
    shoot_type: str | None
    metrics: dict[str, dict[str, float]]   # score name → {recall_strict, recall_cluster, auc}

    def to_dict(self) -> dict:
        return {"gallery_id": self.gallery_id, "n_photos": self.n_photos, "n_positives": self.n_positives,
                "k": self.k, "shoot_type": self.shoot_type, "metrics": self.metrics}


def evaluate_gallery(lg: LabeledGallery, model: PreferenceModel | None, knobs: Knobs,
                     features: Features | None = None, lam_override: float | None = None) -> GalleryEval:
    gd = lg.data
    f = features or build(gd)
    pos = lg.positive_idx
    s = make_sample(lg)
    neg = s.idx[s.y == 0]
    k = knobs.k_multiplier * max(int(pos.size), 1)
    pr = prior(gd.technical_pct, gd.aesthetic_pct)
    cand = {"prior": z(pr)}
    if model is not None:
        cand["pref"] = z(model.raw(f))
        cand["fused"] = model.fuse(pr, f, lam_override)
    metrics = {}
    for name, sc in cand.items():
        strict, cluster = recalls(sc, pos, k, gd.cluster_id)
        metrics[name] = {"recall_strict": strict, "recall_cluster": cluster, "auc": auc(sc, pos, neg)}
    return GalleryEval(gd.gallery_id, gd.n, int(pos.size), k, gd.shoot_type, metrics)


# ── leave-one-gallery-out ─────────────────────────────────────────────────────
def logo(galleries: list[LabeledGallery], knobs: Knobs, features: dict[str, Features] | None = None) -> list[GalleryEval]:
    """갤러리 하나를 빼고 나머지로 학습 → 뺀 갤러리에서 평가. 2개 미만이면 빈 목록(측정 불가)."""
    if len(galleries) < 2:
        return []
    feats = features or {lg.data.gallery_id: build(lg.data) for lg in galleries}
    out = []
    for i, held in enumerate(galleries):
        rest = galleries[:i] + galleries[i + 1:]
        model = train(rest, knobs, feats)
        out.append(evaluate_gallery(held, model, knobs, feats[held.data.gallery_id]))
    return out


def paired_diff(rows: list[GalleryEval], metric: str = "recall_cluster") -> np.ndarray:
    return np.array([r.metrics["fused"][metric] - r.metrics["prior"][metric] for r in rows])


def sign_test(diff: np.ndarray) -> float:
    """쌍 차이의 부호 검정 p (단측: 개선). 동점은 제외. 표본 0 이면 1.0."""
    pos, neg = int((diff > 0).sum()), int((diff < 0).sum())
    if pos + neg == 0:
        return 1.0
    return float(binomtest(pos, pos + neg, alternative="greater").pvalue)


def gate(galleries: list[LabeledGallery], knobs: Knobs, features: dict[str, Features] | None = None,
         metric: str = "recall_cluster") -> dict:
    """연결 조건 세 가지(plan §5). 갤러리 수가 모자라면 이유를 적고 통과시키지 않는다."""
    n = len(galleries)
    result: dict = {"metric": metric, "n_galleries": n, "passed": False, "reasons": [], "rows": [], "curve": []}
    if n < 2:
        result["reasons"].append("홀드아웃 측정 불가 — 라벨 갤러리 2개 미만")
        return result
    feats = features or {lg.data.gallery_id: build(lg.data) for lg in galleries}
    rows = logo(galleries, knobs, feats)
    result["rows"] = [r.to_dict() for r in rows]
    diff = paired_diff(rows, metric)
    p = sign_test(diff)
    worse = float((diff < 0).mean())
    result.update({"mean_diff": float(diff.mean()), "p_value": p, "worse_ratio": worse,
                   "prior_mean": float(np.mean([r.metrics["prior"][metric] for r in rows])),
                   "fused_mean": float(np.mean([r.metrics["fused"][metric] for r in rows]))})
    ok = True
    if p >= knobs.gate_p:
        ok = False
        result["reasons"].append(f"부호 검정 p={p:.3f} ≥ {knobs.gate_p}")
    if worse > knobs.gate_worse_ratio:
        ok = False
        result["reasons"].append(f"악화 비율 {worse:.2f} > {knobs.gate_worse_ratio}")
    # 안정성: 최근 window 회 갤러리 추가에서 평균 차이가 줄지 않았는가 (접두 크기별 LOGO)
    w = knobs.gate_stability_window
    if n < w + 2:
        ok = False
        result["reasons"].append(f"안정성 판단 불가 — 갤러리 {w + 2}개 미만")
    else:
        curve = []
        for m in range(n - w, n + 1):
            sub = galleries[:m]
            curve.append({"n": m, "mean_diff": float(paired_diff(logo(sub, knobs, feats), metric).mean())})
        result["curve"] = curve
        drops = [curve[i]["mean_diff"] - curve[i - 1]["mean_diff"] for i in range(1, len(curve))]
        if any(d < -knobs.gate_stability_tol for d in drops):
            ok = False
            result["reasons"].append("최근 갤러리 추가에서 평균 차이가 줄었다")
    result["passed"] = ok
    return result


def lam_for(n: int, knobs: Knobs) -> float:
    return lam(n, knobs.n0)
