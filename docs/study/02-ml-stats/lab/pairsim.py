"""02 실습의 공용 기반 — 01의 갤러리·학습기를 그대로 빌려 쓴다.

────────────────────────────────────────────────────────────────────────────
왜 새 데이터를 안 만드는가

02는 새 문제를 배우는 영역이 아니라 **01에서 만든 숫자를 판정하는 법**을 배우는
영역이다. 그래서 데이터도 01의 것을 그대로 쓴다 — 같은 합성 갤러리, 같은 쌍 규칙,
같은 Bradley-Terry 학습기.

01에서 우리는 이런 숫자들을 만들었다.

    0.484 ± 0.09   12쌍 × 태그 33차원의 홀드아웃 정확도      (01 step2)
    λ = 0.4        12쌍 구간의 '최적' λ                      (01 step3 [B])
    λ_mmr = 0.3    다양성 다이얼의 '최적'값                    (01 step4 [B])
    0.606          4라운드 뒤의 홀드아웃 정확도                (01 step6 [B])

**이 숫자들은 전부 통계적 주장이다.** 02는 그 주장이 성립하는지, 어디가 새는지를 본다.
그중 몇 개는 실제로 샌다 — step4(승자의 저주)와 step5(다중 비교)가 그것이다.

이 모듈이 하는 일은 하나다: **"사람 한 명에게 실험을 시켜 (맞힌 수, 전체 수)를 얻는다"**
를 한 줄로 만들어 주는 것. 통계 실습에서 매번 파이프라인을 다시 짜지 않기 위해서다.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

# 01의 lab을 import 경로에 올린다. 폴더를 옮기면 이 한 줄만 고치면 된다.
_RECSYS_LAB = (Path(__file__).resolve().parent / ".." / ".." / "01-recsys" / "lab").resolve()
if not _RECSYS_LAB.is_dir():                     # pragma: no cover
    raise SystemExit(f"01-recsys/lab을 찾을 수 없다: {_RECSYS_LAB}")
sys.path.insert(0, str(_RECSYS_LAB))

import gallery as G                # noqa: E402  (경로 삽입 후여야 한다)
from step1_bt import fit_bt        # noqa: E402

STRENGTH, TEMP = 1.5, 0.7          # 01과 같은 값. 바꾸면 01의 모든 숫자와 대응이 깨진다


@lru_cache(maxsize=4)
def world(n: int = 1000, seed: int = 0, tag_error: float = 0.0):
    """갤러리 + 학습/홀드아웃 쌍 풀. 캐시하므로 여러 번 불러도 한 번만 만든다.

    풀을 학습용과 홀드아웃용으로 **먼저** 가른다. 같은 쌍이 양쪽에 들어가면
    정확도가 부풀려진다(누수) — 01 step2에서 이미 본 규칙이다.
    """
    g = G.make_gallery(n, seed=seed, tag_error=tag_error)
    train_pool, test_pool = G.split_pool(G.design_pair_pool(g), test_frac=0.5, seed=7)
    return g, train_pool, test_pool


@dataclass
class Trial:
    """한 사람의 실험 결과. 통계 실습이 다루는 최소 단위."""

    k: int          # 홀드아웃에서 맞힌 쌍 수
    n: int          # 홀드아웃 쌍 수
    w_hat: np.ndarray
    margin: np.ndarray   # (n,) 각 홀드아웃 쌍의 예측 마진. 부호가 맞으면 정답

    @property
    def acc(self) -> float:
        return self.k / self.n if self.n else float("nan")


def run_person(
    person_seed: int,
    n_train: int,
    n_hold: int = 60,
    reg: float = 1.0,
    tag_error: float = 0.0,
    n: int = 1000,
) -> Trial:
    """사람 한 명: 학습 쌍으로 취향을 배우고 홀드아웃 쌍을 맞힌다.

    01의 파이프라인을 그대로 축약한 것이다 —
    쌍 뽑기 → 사람이 답함(확률적) → BT 학습 → 홀드아웃 예측.
    """
    g, train_pool, test_pool = world(n=n, tag_error=tag_error)
    w_true = G.make_person(seed=person_seed, strength=STRENGTH)

    tr = G.sample_pairs(train_pool, n_train, seed=100 + person_seed)
    te = G.sample_pairs(test_pool, n_hold, seed=200 + person_seed)
    c, r = G.answer_pairs(g, w_true, tr, temp=TEMP, seed=300 + person_seed)
    ct, rt = G.answer_pairs(g, w_true, te, temp=TEMP, seed=400 + person_seed)

    w_hat = fit_bt(g.X[c] - g.X[r], reg=reg)
    margin = (g.X[ct] - g.X[rt]) @ w_hat
    return Trial(k=int((margin > 0).sum()), n=len(margin), w_hat=w_hat, margin=margin)


def run_group(
    n_people: int, n_train: int, n_hold: int = 60, reg: float = 1.0,
    tag_error: float = 0.0, seed_base: int = 0,
) -> list[Trial]:
    """사람 여럿. 표본 단위가 '쌍'이 아니라 '사람'이라는 것을 잊지 않기 위해 리스트로 돌려준다."""
    return [
        run_person(seed_base + i, n_train, n_hold, reg=reg, tag_error=tag_error)
        for i in range(n_people)
    ]


def pooled(trials: list[Trial]) -> tuple[int, int]:
    """사람들의 결과를 하나의 이항으로 합친다. **편할 뿐 옳지는 않다** — step1·step5 참조."""
    return sum(t.k for t in trials), sum(t.n for t in trials)


def true_accuracy(person_seed: int, n_hold: int = 2000, tag_error: float = 0.0) -> float:
    """그 사람의 '진짜' 정확도 — 정답 취향 벡터로 아주 많은 쌍을 맞혀 본 값.

    현실에서는 절대 알 수 없다. 시뮬레이션이라서 볼 수 있는 값이고,
    추정량이 이 값을 제대로 덮는지(커버리지) 재는 데 쓴다.
    """
    g, _, test_pool = world(tag_error=tag_error)
    w_true = G.make_person(seed=person_seed, strength=STRENGTH)
    te = G.sample_pairs(test_pool, min(n_hold, len(test_pool)), seed=777)
    ct, rt = G.answer_pairs(g, w_true, te, temp=TEMP, seed=778)
    return float(((g.X[ct] - g.X[rt]) @ w_true > 0).mean())
