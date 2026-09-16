"""PreferenceModel — 가중치 벡터 한 행. `preference_models` 의 열과 1:1.

    pref(x) = w_scalar·x_scalar + w_emb·x_emb + bias
    score   = z(prior) + λ·z(pref)                  λ = n / (n + n0)

λ · z · prior 는 wes 추천 쪽과 같은 정의여야 하는 계약식이라 여기(domain)에 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from preference.domain.features import FEATURE_SPEC, Features


def lam(n_galleries: int, n0: float) -> float:
    """증거량 감쇠. n=0 → 0 (prior 그대로), n=n0 → 0.5, n→∞ → 1."""
    n = max(int(n_galleries), 0)
    return n / (n + n0) if n0 > 0 else 1.0


def z(values: np.ndarray) -> np.ndarray:
    """wes RecommendationScoring.z 와 같은 정의(모집단 표준편차, 0 이면 나누지 않는다)."""
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return v
    std = v.std()
    return (v - v.mean()) / (std if std > 1e-12 else 1.0)


def prior(technical_pct: np.ndarray, aesthetic_pct: np.ndarray) -> np.ndarray:
    """wes 와 같은 prior — 0.5·tech + 0.5·aes (백분위)."""
    return 0.5 * np.asarray(technical_pct, dtype=float) + 0.5 * np.asarray(aesthetic_pct, dtype=float)


@dataclass
class PreferenceModel:
    w_scalar: np.ndarray
    w_emb: np.ndarray
    bias: float
    n_galleries: int
    n_positives: int
    train_gallery_ids: list[str]
    embedding_model: str = ""
    model_version: str = ""
    feature_spec: str = FEATURE_SPEC
    lam: float = 0.0
    holdout: dict = field(default_factory=dict)
    active: bool = False

    def raw(self, f: Features) -> np.ndarray:
        return f.scalar @ self.w_scalar + f.emb @ self.w_emb + self.bias

    def fuse(self, prior_raw: np.ndarray, f: Features, lam_override: float | None = None) -> np.ndarray:
        """추천 점수 — wes 가 하게 될 계산과 같다."""
        l = self.lam if lam_override is None else lam_override
        return z(prior_raw) + l * z(self.raw(f))

    def to_row(self) -> dict:
        return {
            "feature_spec": self.feature_spec,
            "embedding_model": self.embedding_model,
            "model_version": self.model_version,
            "w_scalar": [float(v) for v in self.w_scalar],
            "w_emb": [float(v) for v in self.w_emb],
            "bias": float(self.bias),
            "lambda": float(self.lam),
            "n_galleries": int(self.n_galleries),
            "n_positives": int(self.n_positives),
            "train_gallery_ids": list(self.train_gallery_ids),
            "holdout": self.holdout,
            "active": bool(self.active),
        }

    @classmethod
    def from_row(cls, row: dict) -> "PreferenceModel":
        return cls(
            w_scalar=np.asarray(row["w_scalar"], dtype=float), w_emb=np.asarray(row["w_emb"], dtype=float),
            bias=float(row["bias"]), n_galleries=int(row["n_galleries"]), n_positives=int(row["n_positives"]),
            train_gallery_ids=[str(g) for g in row["train_gallery_ids"]],
            embedding_model=row.get("embedding_model", ""), model_version=row.get("model_version", ""),
            feature_spec=row.get("feature_spec", FEATURE_SPEC), lam=float(row.get("lambda", 0.0)),
            holdout=row.get("holdout", {}), active=bool(row.get("active", False)),
        )
