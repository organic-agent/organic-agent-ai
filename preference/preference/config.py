"""설정 — categorize 와 같은 방식. `Settings.from_env()` 하나로 읽고 코드 어디서도 `os.environ` 을 만지지 않는다."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

MODULE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Knobs:
    """학습·평가 손잡이. 값의 근거는 plan-preference-layer.md §3~§5."""

    #: L2 — 스칼라(11) 약하게, 임베딩(1536) 강하게. 갤러리당 양성 수십 개라 임베딩 쪽은 강한 정규화 없이는 외운다.
    l2_scalar: float = 1e-2
    l2_emb: float = 0.1
    #: λ(n) = n / (n + n0). n = 학습 갤러리 수.
    n0: float = 5.0
    #: recall@K 의 K = k_multiplier × 양성 수.
    k_multiplier: int = 3
    #: 게이트 — 부호 검정 p, 악화 갤러리 비율 상한, 안정성에 쓰는 최근 갤러리 수.
    gate_p: float = 0.05
    gate_worse_ratio: float = 0.30
    gate_stability_window: int = 3
    #: 안정성 — 최근 추가에서 평균 차이가 이만큼 넘게 줄면 실패. 갤러리 하나가 바꾸는 recall 폭이 1/m 이라 0 으로 두면 잡음에 걸린다.
    gate_stability_tol: float = 0.05
    #: L-BFGS 반복 상한.
    max_iter: int = 500


@dataclass(frozen=True)
class Settings:
    #: 로컬 모드 출력 루트 — out/preference/<gallery>/ 에 npz 캐시와 결과가 놓인다.
    out_root: Path

    db_host: str | None = None
    db_port: int = 5432
    db_name: str | None = None
    db_user: str | None = None
    db_password: str | None = None
    db_sslmode: str = "require"
    db_sslrootcert: str | None = None

    knobs: Knobs = field(default_factory=Knobs)

    @property
    def db_enabled(self) -> bool:
        return bool(self.db_host and self.db_name and self.db_user)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            out_root=Path(os.environ.get("PREFERENCE_OUT", MODULE_ROOT / "out")),
            db_host=os.environ.get("DB_HOST"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=os.environ.get("DB_NAME"),
            db_user=os.environ.get("DB_USER"),
            db_password=os.environ.get("DB_PASSWORD"),
            db_sslmode=os.environ.get("DB_SSLMODE", "require"),
            db_sslrootcert=os.environ.get("DB_SSLROOTCERT"),
            knobs=Knobs(
                l2_scalar=float(os.environ.get("PREF_L2_SCALAR", Knobs.l2_scalar)),
                l2_emb=float(os.environ.get("PREF_L2_EMB", Knobs.l2_emb)),
                n0=float(os.environ.get("PREF_N0", Knobs.n0)),
            ),
        )
