"""STEP 3 — 경험적 베이즈: λ의 k와 conf의 a를 데이터에서 정하기.

────────────────────────────────────────────────────────────────────────────
이론

01 step3에서 우리는 두 개의 상수를 **손으로 정했다.**

    λ(n) = 0.2 + 0.65 · n/(n + k)      k = 10      ← 설계안 명세에 맞춰 역산
    conf = max(0, 2·p̂ − 1),  p̂ = (승 + a)/(쌍 + 2a)      a = 1      ← 그냥 1

그리고 01 step3 [B]에서 이런 관찰이 있었다 — **"설계 곡선이 쓴 λ가 매번 최적 λ보다 위에 있다."**
02는 그 상수들을 **데이터에서 추정**해서 그 관찰의 원인을 찾는다.

1) 야구 타율 문제(David Robinson, empirical Bayes). 10타수 4안타(0.400)와
   1000타수 300안타(0.300) 중 누가 더 좋은 타자인가. 0.400을 그대로 믿으면 안 된다 —
   표본이 작으면 극단값이 쉽게 나온다. 해법: **모든 타자의 타율 분포를 사전분포로 삼아**
   개별 추정치를 그쪽으로 끌어당긴다(shrinkage).

       p̂_EB = (안타 + a) / (타수 + a + b)      Beta(a, b)가 사전분포

   a, b를 **다른 타자들의 데이터에서 추정**하기 때문에 '경험적(empirical)' 베이즈다.

2) **우리 conf(axis)가 정확히 이 식이다.** 타자 → (사람, 축), 타수 → 그 축이 갈린 쌍 수,
   안타 → 추정 방향이 맞은 쌍 수. 01은 a=b=1을 그냥 썼다. 데이터가 말하는 값은 얼마인가?

3) 신뢰도(reliability)와 shrinkage 계수. 계층 모형에서 사후평균의 가중치는

       n / (n + k),      k = (개인 내 잡음 분산) / (사람 간 참값 분산)

   **k는 임의로 고르는 값이 아니라 두 분산의 비다.** 사람마다 취향이 크게 다르면(τ² 큼)
   k가 작아 데이터를 빨리 믿고, 한 쌍의 답이 잡음투성이면(σ² 큼) k가 커서 천천히 믿는다.
   λ의 k도 같은 뜻이므로 **측정할 수 있다.**

4) 재는 법: 학습 쌍 n개로 추정한 ŵ가 참값 w와 얼마나 닮았는지(설명력 r²)를 n마다 재고,
   그 곡선에 n/(n+k)를 맞춘다. r²는 "추정이 참값 분산의 몇 %를 설명하는가"라
   신뢰도의 정의와 그대로 맞아떨어진다.

실행: ../../.venv/bin/python step3_shrinkage.py   (실측 약 1초)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import betaln

import pairsim as P              # ★ 먼저 import

import gallery as G              # noqa: E402
from step1_bt import fit_bt      # noqa: E402
from step3_lambda import LAMBDA_MAX, LAMBDA_MIN, lambda_of   # noqa: E402

N_PEOPLE = 40
N_LIST = (6, 12, 24, 48, 96, 192)


def center_by_axis(w: np.ndarray) -> np.ndarray:
    """축 내부 평균을 뺀다. 원핫은 축 내부 상대값만 식별되므로(01 step2 §5) 필수."""
    w = w.copy()
    for ax, vals in G.AXES.items():
        cols = [G.FEATURE_INDEX[f"{ax}={v}"] for v in vals]
        w[cols] -= w[cols].mean()
    return w


def learnable_mask() -> np.ndarray:
    """scene은 쌍 규칙상 신호가 0이라 제외한다 — 안 빼면 신뢰도가 인위적으로 낮아진다."""
    m = np.ones(G.D_TAG, dtype=bool)
    for v in G.AXES["scene"]:
        m[G.FEATURE_INDEX[f"scene={v}"]] = False
    return m


# ── 베타-이항 경험적 베이즈 ─────────────────────────────────────────────────
def beta_binom_nll(params: np.ndarray, k: np.ndarray, n: np.ndarray) -> float:
    """Beta(a,b) 사전 + 이항 관측의 음의 로그우도. a,b는 로그 스케일로 받는다."""
    a, b = np.exp(params)
    return float(-(betaln(k + a, n - k + b) - betaln(a, b)).sum())


def fit_beta_prior(k: np.ndarray, n: np.ndarray) -> tuple[float, float]:
    """관측 (승, 시도)들에서 사전분포 Beta(a,b)를 최대우도로 추정한다."""
    res = minimize(beta_binom_nll, np.array([0.0, 0.0]), args=(k, n), method="Nelder-Mead")
    a, b = np.exp(res.x)
    return float(a), float(b)


def collect_axis_records(
    g, train_pool, n_train: int, n_people: int, loo: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """(사람 × 축)마다 '그 축이 갈린 쌍 수'와 '추정 방향이 맞은 쌍 수'를 모은다.

    01 step3의 `axis_confidence`가 세는 것과 완전히 같은 두 숫자다.

    loo=False — 01과 같다. **학습에 쓴 그 쌍으로** 방향이 맞았는지 센다(in-sample).
    loo=True  — 그 쌍을 빼고 다시 학습한 ŵ로 센다(leave-one-out). 낙관 편향이 사라진다.
    """
    wins, tries = [], []
    for ps in range(n_people):
        w_true = G.make_person(seed=ps, strength=P.STRENGTH)
        tr = G.sample_pairs(train_pool, n_train, seed=100 + ps)
        c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=300 + ps)
        d_all = g.X[c] - g.X[r]
        w_full = center_by_axis(fit_bt(d_all, reg=1.0))
        w_minus = None
        if loo:
            w_minus = []
            for t in range(len(c)):
                m = np.ones(len(c), dtype=bool); m[t] = False
                w_minus.append(center_by_axis(fit_bt(d_all[m], reg=1.0)))
        for ax, vals in G.AXES.items():
            cols = [G.FEATURE_INDEX[f"{ax}={v}"] for v in vals]
            n_ax = win = 0
            for t, (ci, ri) in enumerate(zip(c, r)):
                tc, tr_ = g.tags[ci][ax], g.tags[ri][ax]
                if tc == tr_:
                    continue
                w_ax = (w_minus[t] if loo else w_full)[cols]
                n_ax += 1
                win += int(w_ax[vals.index(tc)] > w_ax[vals.index(tr_)])
            if n_ax:
                wins.append(win); tries.append(n_ax)
    return np.array(wins), np.array(tries)


def main() -> None:
    g, train_pool, test_pool = P.world()
    mask = learnable_mask()

    # ── [A] 소표본의 극단값 — 야구 타율과 같은 문제 ──────────────────────
    print("=" * 86)
    print("[A] 축별 승률의 분포 — 쌍이 적은 축일수록 0과 1로 튄다 (학습 30쌍, 40명)")
    print("=" * 86)
    wins, tries = collect_axis_records(g, train_pool, 30, N_PEOPLE)
    print(f"{'그 축이 갈린 쌍 수':<18}{'표본 수':>8}{'평균 승률':>10}{'표준편차':>10}{'0 또는 1인 비율':>16}")
    for lo, hi, label in ((1, 3, "1~3쌍"), (4, 7, "4~7쌍"), (8, 15, "8~15쌍"), (16, 999, "16쌍 이상")):
        m = (tries >= lo) & (tries <= hi)
        if not m.any():
            continue
        p = wins[m] / tries[m]
        extreme = float(((p == 0) | (p == 1)).mean())
        print(f"{label:<18}{m.sum():>8}{p.mean():>10.3f}{p.std():>10.3f}{extreme:>16.2f}")
    print("  → 1~3쌍짜리 축은 승률이 0이나 1로 나오는 일이 흔하다. **10타수 4안타와 같은 상황**이다.")
    print("     conf = 2p̂ − 1을 날것으로 쓰면 '이 고객은 클로즈업을 100% 선호'가 되어 버린다.")

    # ── [B] 사전분포를 데이터에서 추정 ───────────────────────────────────
    print("\n" + "=" * 86)
    print("[B] 사전분포 Beta(a,b)를 데이터에서 — 01이 쓴 a=b=1은 맞는 값인가")
    print("=" * 86)
    a_hat, b_hat = fit_beta_prior(wins, tries)
    print(f"  최대우도 추정:  a = {a_hat:.2f}   b = {b_hat:.2f}")
    print(f"  사전 평균 a/(a+b) = {a_hat / (a_hat + b_hat):.3f}   사전 강도 a+b = {a_hat + b_hat:.2f}")
    print(f"  01의 가정:      a = b = 1  →  사전 평균 0.500, 사전 강도 2.00")
    print()
    print(f"{'갈린 쌍':<10}{'승':<6}{'날것 p̂':>10}{'01 (a=1)':>11}{'EB 추정':>10}")
    for n_ax, win in ((1, 1), (2, 2), (3, 2), (6, 4), (12, 8)):
        raw = win / n_ax
        old = (win + 1) / (n_ax + 2)
        eb = (win + a_hat) / (n_ax + a_hat + b_hat)
        print(f"{n_ax:<10}{win:<6}{raw:>10.3f}{old:>11.3f}{eb:>10.3f}")
    print(f"  → 사전 평균이 0.5가 아니라 {a_hat / (a_hat + b_hat):.2f}다. 우연이 아니다.")
    print()
    print("  ★ 01의 `axis_confidence`는 **학습에 쓴 그 쌍으로** 방향이 맞았는지 센다(in-sample).")
    print("     학습 쌍은 ŵ가 이미 맞추도록 적합된 쌍이다 — 당연히 잘 맞는다.")
    print("     같은 계산을 leave-one-out으로 다시 하면:")
    wins_loo, tries_loo = collect_axis_records(g, train_pool, 30, N_PEOPLE, loo=True)
    a_loo, b_loo = fit_beta_prior(wins_loo, tries_loo)
    print(f"       in-sample (01과 동일) : 평균 승률 {np.mean(wins / tries):.3f}   사전 {a_hat:.1f}/{b_hat:.1f} → 평균 {a_hat / (a_hat + b_hat):.3f}")
    print(f"       leave-one-out        : 평균 승률 {np.mean(wins_loo / tries_loo):.3f}   사전 {a_loo:.1f}/{b_loo:.1f} → 평균 {a_loo / (a_loo + b_loo):.3f}")
    print("  → 낙관 편향이 실제로 있다. **conf는 '이 축에 의견이 있는가'를 재려 했는데,")
    print("     '학습 쌍을 얼마나 잘 외웠나'를 함께 재고 있었다.**")
    print("     증거가 적은 축일수록 외우기 쉬우므로(1쌍이면 100% 외운다) 편향이 크다 —")
    print("     하필 conf가 가장 조심해야 할 구간이다.")
    print("  → 되먹임: conf를 LOO로 재거나, 사전 평균을 0.5가 아니라 **그 in-sample 기준선**으로")
    print("     잡아야 한다. 지금 구현은 '아무 의견 없는 축'도 conf > 0을 받는다.")

    # ── [C] shrinkage가 정말 오차를 줄이는가 ─────────────────────────────
    print("\n" + "=" * 86)
    print("[C] shrinkage의 값어치 — 진짜 값을 아는 세계에서 제곱오차를 비교")
    print("=" * 86)
    rng = np.random.default_rng(0)
    reps = 4000
    n_draw = rng.choice(tries, size=reps)                    # 실제 관측된 '갈린 쌍 수' 분포 그대로
    p_true = rng.beta(a_hat, b_hat, size=reps)               # 진짜 승률
    k_draw = rng.binomial(n_draw, p_true)
    raw = k_draw / n_draw
    old = (k_draw + 1) / (n_draw + 2)
    eb = (k_draw + a_hat) / (n_draw + a_hat + b_hat)
    print(f"{'추정량':<22}{'전체 MSE':>12}{'쌍 1~3개인 축의 MSE':>22}")
    small = n_draw <= 3
    for name, est in (("날것 p̂", raw), ("01 (a=b=1)", old), ("EB (추정된 a,b)", eb)):
        print(f"{name:<22}{np.mean((est - p_true) ** 2):>12.4f}{np.mean((est[small] - p_true[small]) ** 2):>22.4f}")
    print("  → shrinkage가 전 구간에서 이긴다. 특히 **쌍이 1~3개인 축에서 차이가 크다.**")
    print("     a=b=1도 날것보다는 훨씬 낫다 — 01의 선택이 틀린 게 아니라 **덜 눌렀을** 뿐이다.")
    print("  ※ 위 표는 **사전분포가 맞다는 가정** 위에 있다 — 진짜 p를 바로 그 Beta(a,b)에서 뽑았다.")
    print("     경험적 베이즈에 유리하게 짜인 시험이라는 뜻이다. 사전분포가 틀리면 어떻게 되나:")
    p_wrong = rng.beta(2.0, 2.0, size=reps)          # 훨씬 퍼진 진짜 분포
    k_wrong = rng.binomial(n_draw, p_wrong)
    print(f"\n  진짜 분포가 Beta(2,2)인데 EB는 Beta({a_hat:.0f},{b_hat:.0f})을 쓴 경우")
    print(f"{'추정량':<22}{'전체 MSE':>12}")
    for name, est in (
        ("날것 p̂", k_wrong / n_draw),
        ("01 (a=b=1)", (k_wrong + 1) / (n_draw + 2)),
        ("EB (틀린 사전)", (k_wrong + a_hat) / (n_draw + a_hat + b_hat)),
    ):
        print(f"{name:<22}{np.mean((est - p_wrong) ** 2):>12.4f}")
    print("  → 사전분포가 틀리면 **EB가 날것보다 나쁠 수 있다.** 세게 누를수록 틀렸을 때 크게 틀린다.")
    print("     a=b=1은 약하게 누르므로 덜 다친다 — 01의 보수적 선택에도 이유가 있는 셈이다.")
    print("     교훈: shrinkage의 세기는 **사전분포를 얼마나 믿느냐**의 함수다. 사람 데이터가")
    print("     쌓이기 전까지는 약한 사전(a=b=1 근처)이 안전하다.")

    # ── [D] λ의 k를 데이터에서 ───────────────────────────────────────────
    print("\n" + "=" * 86)
    print("[D] λ 곡선의 k — 손으로 정한 10이 맞는 값인가")
    print("=" * 86)
    print("  신뢰도 r²(n) = 학습 n쌍으로 만든 ŵ가 참값 w의 분산을 몇 %나 설명하는가")
    print(f"  (축 내부 중심화 후, scene 제외, 사람 {N_PEOPLE}명 평균)\n")
    rel = {}
    for n_train in N_LIST:
        rs = []
        for ps in range(N_PEOPLE):
            w_true = center_by_axis(G.make_person(seed=ps, strength=P.STRENGTH))
            tr = G.sample_pairs(train_pool, n_train, seed=100 + ps)
            c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=300 + ps)
            w_hat = center_by_axis(fit_bt(g.X[c] - g.X[r], reg=1.0))
            a_, b_ = w_hat[mask], w_true[mask]
            if a_.std() < 1e-12 or b_.std() < 1e-12:     # ŵ가 통째로 0인 경우 (증거 없음)
                rs.append(0.0)
                continue
            cc = np.corrcoef(a_, b_)[0, 1]
            rs.append(float(cc ** 2) if np.isfinite(cc) else 0.0)
        rel[n_train] = float(np.mean(rs))

    obj = lambda k: sum((rel[n] - n / (n + k)) ** 2 for n in N_LIST)
    k_hat = float(minimize_scalar(obj, bounds=(0.5, 500), method="bounded").x)

    print(f"{'학습 쌍':<10}{'측정 r²':>10}{'n/(n+k̂)':>11}{'n/(n+10)':>11}")
    for n_train in N_LIST:
        print(f"{n_train:<10}{rel[n_train]:>10.3f}{n_train / (n_train + k_hat):>11.3f}{n_train / (n_train + 10):>11.3f}")
    print(f"\n  적합된 k̂ = {k_hat:.0f}      01의 가정 k = 10")
    print("  → **한 자릿수가 아니라 세 자릿수다.** k=10은 '쌍 10개면 데이터와 사전이 반반'이라는")
    print("     뜻인데, 실제로는 쌍 10개가 참값 분산의 10%밖에 설명하지 못한다.")

    print(f"\n  λ 곡선 비교 (λ_min={LAMBDA_MIN}, λ_max={LAMBDA_MAX})")
    print(f"{'증거':<8}{'01 곡선 (k=10)':>16}{'k̂ 곡선':>10}{'01 step3 [B]의 최적 λ':>24}")
    best = {4: "0.2", 12: "0.4  (태그 오차 시 0.2)", 30: "0.6", 100: "0.6", 300: "0.8"}
    for n_ev in (4, 12, 30, 100, 300):
        lam_new = LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * n_ev / (n_ev + k_hat)
        print(f"{n_ev:<8}{lambda_of(n_ev):>16.2f}{lam_new:>10.2f}{best[n_ev]:>24}")
    print("  → k̂ 곡선이 초반 구간에서 실측 최적값에 훨씬 가깝다. 12쌍에서 0.55 → 0.26.")
    print("     01 step3 [B]가 '곡선이 매번 최적보다 위에 있다'고 관찰한 것의 **원인이 k다.**")
    print("     후반(100~300쌍)에서는 k̂ 곡선이 오히려 보수적이다 — 신뢰도 r²는 '방향이")
    print("     얼마나 닮았나'인데 순위를 매기는 데는 방향의 일부만 맞아도 되기 때문이다.")
    print(f"  → 결론: **k는 손으로 정할 값이 아니라 측정할 값이다.** 다만 여기서 나온 {k_hat:.0f}은")
    print("     합성 데이터의 값이다. 실제 값은 W1–2 실험(사람 10명 × 120쌍)에서 같은 방법으로")
    print("     재야 한다 — 그 실험이 k 추정용 데이터를 함께 만들어 준다.")


if __name__ == "__main__":
    main()
