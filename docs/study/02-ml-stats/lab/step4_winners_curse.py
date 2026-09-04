"""STEP 4 — 승자의 저주: 같은 데이터로 고르고 같은 데이터로 보고하면 안 되는 이유.

────────────────────────────────────────────────────────────────────────────
이론

01 step3 [B]에서 우리는 이렇게 했다.

    λ 격자 6개에서 각각 만족도를 재고 → **최댓값이 나온 λ를 '최적'이라 부르고**
    → 그 값을 표에 적었다.

01 step4 [B]도 λ_mmr 7개로 같은 일을 했고, 01 step2 [B]는 reg 6개로 했다.
**세 번 다 같은 함정을 밟았다.**

1) 승자의 저주(winner's curse). 여러 후보의 측정값에는 잡음이 섞여 있다.
   그중 **최댓값을 고르면 잡음이 큰 쪽이 뽑힐 확률이 높다.**
   따라서 "고른 후보의 측정값"은 그 후보의 진짜 성능보다 **위로 치우친다.**

       E[측정된 최댓값]  >  실제 그 후보의 성능

   후보가 많을수록, 잡음이 클수록 편향이 크다. 데이터를 늘리는 것만으로는 안 사라진다 —
   **선택과 보고에 같은 데이터를 쓰는 한** 구조적으로 남는다.

2) 대책은 하나다. **고르는 데이터와 보고하는 데이터를 나눈다.**

       학습(train)      모델 파라미터 w를 적합
       검증(validation) 하이퍼파라미터 λ·reg·λ_mmr을 고름
       시험(test)       **한 번만** 보고 그대로 보고

   시험 세트를 보고 λ를 바꾸는 순간 그것은 검증 세트가 된다. 그리고 다시는
   그 숫자를 '깨끗한 성능'이라고 부를 수 없다.

3) 우리 설계에 그대로 걸린다. `plan.md` §6은 이렇게 적혀 있다 —
   *"λ 곡선·w·β는 §6 홀드아웃 정확도로 튜닝"*. 그런데 §6의 홀드아웃 정확도는
   **주지표이기도 하다.** 튜닝용과 보고용이 같은 데이터다.

실행: ../../.venv/bin/python step4_winners_curse.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

import pairsim as P              # ★ 먼저 import

import gallery as G              # noqa: E402
from step1_bt import fit_bt      # noqa: E402
from step3_lambda import axis_confidence, pref_scores, satisfaction, z   # noqa: E402

N_PEOPLE = 24
TOP_K = 30
N_TRAIN = 30
LAM_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
SPLITS = 400


def satisfaction_matrix(lam_grid: tuple[float, ...], n_people: int = N_PEOPLE) -> np.ndarray:
    """M[i, j] = 사람 i에게 λ_j로 30장을 뽑았을 때의 평균 만족도.

    01 step3 [B]의 표를 **사람별로 펼친 것**이다. 그 표는 이 행렬의 열 평균이었다.
    """
    g, train_pool, _ = P.world()
    prior = z((g.tech + g.aes) / 2)
    M = np.zeros((n_people, len(lam_grid)))
    for i in range(n_people):
        w_true = G.make_person(seed=i, strength=P.STRENGTH)
        sat = satisfaction(g, w_true)
        tr = G.sample_pairs(train_pool, N_TRAIN, seed=100 + i)
        c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=300 + i)
        w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
        pref = z(pref_scores(g, w_hat, axis_confidence(g, w_hat, c, r)))
        for j, lam in enumerate(lam_grid):
            score = (1 - lam) * prior + lam * pref
            M[i, j] = sat[np.argsort(-score)[:TOP_K]].mean()
    return M


def curse(M: np.ndarray, rng, splits: int = SPLITS) -> dict:
    """사람을 절반씩 검증/시험으로 갈라 λ를 고르고, 두 곳에서의 값을 비교한다."""
    n = M.shape[0]
    half = n // 2
    val_best, test_at_pick, test_best, picked = [], [], [], []
    for _ in range(splits):
        perm = rng.permutation(n)
        v, t = perm[:half], perm[half:]
        vm, tm = M[v].mean(axis=0), M[t].mean(axis=0)
        j = int(np.argmax(vm))
        val_best.append(vm[j])
        test_at_pick.append(tm[j])
        test_best.append(tm.max())
        picked.append(j)
    return {
        "val_best": float(np.mean(val_best)),
        "test_at_pick": float(np.mean(test_at_pick)),
        "test_best": float(np.mean(test_best)),
        "picked": np.array(picked),
    }


def main() -> None:
    rng = np.random.default_rng(0)

    # ── [A] 승자의 저주를 직접 잰다 ──────────────────────────────────────
    print("=" * 88)
    print("[A] 01 step3 [B]의 '최적 λ'는 얼마나 낙관적인가")
    print("=" * 88)
    M = satisfaction_matrix(LAM_GRID)
    print(f"  사람 {N_PEOPLE}명 × λ 격자 {len(LAM_GRID)}개. 절반으로 갈라 한쪽에서 고르고 다른 쪽에서 확인.")
    print(f"  (참고: 전체 {N_PEOPLE}명의 열 평균 = 01 step3 [B]가 보고한 그 줄)")
    print("  " + "  ".join(f"λ={l:g}:{M[:, j].mean():.3f}" for j, l in enumerate(LAM_GRID)))
    print()
    res = curse(M, rng)
    print(f"{'검증셋에서 본 최댓값':<28}{res['val_best']:>9.3f}   ← 리포트에 쓰고 싶은 숫자")
    print(f"{'같은 λ의 시험셋 성능':<28}{res['test_at_pick']:>9.3f}   ← 실제로 얻는 성능")
    print(f"{'낙관 편향':<28}{res['val_best'] - res['test_at_pick']:>9.3f}")
    print(f"{'시험셋의 진짜 최댓값':<28}{res['test_best']:>9.3f}   ← 사후적으로만 알 수 있는 값")
    counts = np.bincount(res["picked"], minlength=len(LAM_GRID))
    print("\n  고른 λ의 분포: " + "  ".join(f"λ={l:g}:{c / SPLITS:.0%}" for l, c in zip(LAM_GRID, counts)))
    print("  → **뽑히는 λ가 갈린다.** 절반씩 갈랐을 뿐인데 0.4와 0.6 사이를 오간다.")
    print("     '최적 λ는 0.4'라는 문장은 이 분포에서 한 번 뽑은 결과일 뿐이다 —")
    print("     01 step3 [B]의 '최적 λ' 열은 그 정도 해상도로만 읽어야 한다.")
    print("  ※ 여기서 낙관 편향이 작게 나오는 데는 이유가 있다. λ 곡선이 최댓값 근처에서")
    print("     평평해서 어느 쪽을 골라도 성능이 비슷하기 때문이다. **편향의 크기는")
    print("     후보들이 얼마나 비슷한가에 달려 있다** — [B]에서 그 의존성을 본다.")

    # ── [B] 후보가 많을수록 저주가 깊다 ──────────────────────────────────
    print("\n" + "=" * 88)
    print("[B] 후보 개수와 낙관 편향 — 격자를 촘촘히 할수록 나빠진다")
    print("=" * 88)
    print(f"{'격자':<28}{'후보 수':>8}{'검증 최댓값':>12}{'시험 성능':>11}{'낙관 편향':>11}")
    grids = {
        "λ ∈ {0.2, 0.6}": (0.2, 0.6),
        "λ ∈ {0, .2, .4, .6, .8, 1}": LAM_GRID,
        "λ 0~1을 0.05 간격": tuple(round(x, 2) for x in np.arange(0, 1.001, 0.05)),
    }
    for name, grid in grids.items():
        Mg = satisfaction_matrix(grid)
        r = curse(Mg, np.random.default_rng(1))
        print(f"{name:<28}{len(grid):>8}{r['val_best']:>12.3f}{r['test_at_pick']:>11.3f}"
              f"{r['val_best'] - r['test_at_pick']:>11.3f}")
    print("  → 후보를 늘리면 검증 최댓값은 오르는데 **시험 성능은 안 오른다.** 편향만 커진다.")
    print("     '더 촘촘히 탐색했다'가 '더 좋은 값을 찾았다'가 아니라는 뜻이다.")
    print("  ※ 이건 λ 하나의 이야기가 아니다. 우리는 λ·λ_mmr·reg·a·K·w_t·w_a를 고른다.")
    print("     조합으로 세면 후보가 수백 개다. 전부 같은 홀드아웃에서 고르면 편향이 쌓인다.")

    # ── [C] 3분할이 답이다 ──────────────────────────────────────────────
    print("\n" + "=" * 88)
    print("[C] 2분할 vs 3분할 — 보고하는 숫자가 얼마나 달라지나")
    print("=" * 88)
    M20 = satisfaction_matrix(tuple(round(x, 2) for x in np.arange(0, 1.001, 0.05)))
    n = M20.shape[0]
    rng2 = np.random.default_rng(2)
    two, three = [], []
    for _ in range(SPLITS):
        perm = rng2.permutation(n)
        a, b, c = perm[:8], perm[8:16], perm[16:]
        # 2분할: 같은 데이터(a+b)에서 고르고 그 값을 보고
        vm = M20[np.concatenate([a, b])].mean(axis=0)
        two.append(vm.max())
        # 3분할: a+b에서 고르고 c에서 보고
        three.append(M20[c].mean(axis=0)[int(np.argmax(vm))])
    print(f"{'2분할 (고른 곳에서 보고)':<30}{np.mean(two):>9.3f}")
    print(f"{'3분할 (따로 둔 시험셋에서 보고)':<30}{np.mean(three):>9.3f}")
    print(f"{'차이':<30}{np.mean(two) - np.mean(three):>9.3f}")
    print("  → 3분할의 숫자가 낮다. **그게 정직한 숫자다.** 2분할의 숫자는 우리가 최댓값을")
    print("     골랐다는 사실 때문에 부풀려져 있고, 실제 고객에게서는 재현되지 않는다.")
    print("  → 대가: 시험셋으로 뺀 만큼 검증에 쓸 사람이 줄어 λ 선택이 더 흔들린다.")
    print("     사람이 귀할 때는 **중첩 교차검증**(바깥 루프=시험, 안쪽 루프=선택)으로 아낀다.")

    # ── [D] 우리 문서에 그대로 걸려 있다 ────────────────────────────────
    print("\n" + "=" * 88)
    print("[D] plan.md §6에 그대로 걸려 있는 문장")
    print("=" * 88)
    print('  현행: "λ 곡선·w·β는 §6 홀드아웃 정확도로 튜닝"')
    print("        그런데 §6의 홀드아웃 정확도는 **주지표**이기도 하다 — 같은 데이터다.")
    print()
    print("  W1–2 실험(사람 10명 × 120쌍)에 적용하면:")
    print(f"{'용도':<22}{'배분':<16}{'무엇을 하나'}")
    print(f"{'학습':<22}{'사람당 60쌍':<16}취향 벡터 w 적합")
    print(f"{'검증':<22}{'사람 6명':<16}λ·reg·λ_mmr 선택 — 몇 번이든 본다")
    print(f"{'시험':<22}{'사람 4명':<16}**한 번만** 본다. 리포트의 주지표")
    print("  → 사람 단위로 갈라야 한다. 쌍 단위로 가르면 같은 사람이 양쪽에 들어가")
    print("     누수가 생긴다(01 step5 [D]에서 본 군집 구조와 같은 이유).")
    print("  → 10명이 적어 보이면 **중첩 교차검증**: 10명을 5겹으로 나눠 매번 2명을 시험으로,")
    print("     나머지 8명 안에서 다시 갈라 λ를 고른다. 모든 사람이 한 번씩 시험셋이 된다.")
    print("  ※ 그리고 어느 쪽이든 **λ 선택은 실험 전에 정해 둘수록 좋다.** 이번 실험에서")
    print("     λ까지 고르려면 표본이 두 배 필요하다 — step3 [D]처럼 k를 따로 추정해")
    print("     λ 곡선을 **먼저 확정**하고 실험에 들어가는 편이 싸다.")


if __name__ == "__main__":
    main()
