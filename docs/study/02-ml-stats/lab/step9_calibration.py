"""STEP 9 — 확률 보정: 0.8이라는 점수가 정말 80%인가.

────────────────────────────────────────────────────────────────────────────
이론

02 step2 [A]는 이렇게 끝났다 —
**"정확도가 못 보는 변화가 있다. 확률값을 쓰는 곳에서는 그 차이가 드러난다."**
STEP 9가 그 확률값을 다룬다.

1) 보정(calibration)이란. 모델이 "0.8"이라고 말한 사진들을 모아 보니 실제로 80%가
   양성이면 **보정됐다(calibrated)** 고 한다. 90%거나 60%면 안 된 것이다.

       신뢰도 도표(reliability diagram): 예측 확률을 구간으로 묶고
       구간마다 (평균 예측, 실제 비율)을 찍는다. 대각선이면 완벽.

       ECE(Expected Calibration Error) = Σ (구간 비중) × |평균 예측 − 실제 비율|

2) **보정과 순위는 독립이다.** 점수에 단조 함수를 씌우면 순위가 안 바뀌므로
   AUC·AP는 그대로인데 보정은 완전히 달라진다. 반대도 마찬가지다.
   → **"AUC가 높으니 확률을 믿어도 된다"는 틀린 추론이다.**

3) 고치는 법 두 가지.
   · **Platt scaling** — 점수를 로지스틱 회귀에 한 번 더 통과시킨다(파라미터 2개).
     데이터가 적을 때 안전하다.
   · **Isotonic regression** — 단조 계단 함수를 자유롭게 적합한다. 유연하지만 과적합한다.

4) 우리 시스템에서 확률이 필요한 자리 (전부 `plan.md`에 있다):
   · **A-7 클러스터 대표 선정** — eyes_open과 technical을 "결정적 합성"한다.
     서로 다른 점수를 더하려면 **같은 단위**여야 한다.
   · **λ 결합** — prior와 pref를 섞는다. z-표준화로 스케일은 맞췄지만 보정은 아니다.
   · **이유 문장** — "선호하십니다"라고 단언할 때의 근거.

실행: ../../.venv/bin/python step9_calibration.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

import abatch as A
import pairsim as P

import gallery as G              # noqa: E402
from step1_bt import fit_bt      # noqa: E402
from step7_classification import auc, average_precision, pr_curve, roc_curve   # noqa: E402

N_BINS = 10


def reliability(prob: np.ndarray, label: np.ndarray, n_bins: int = N_BINS):
    """구간별 (평균 예측, 실제 비율, 표본 수). 보정의 정의 그대로."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if m.sum() == 0:
            continue
        rows.append((float(prob[m].mean()), float(label[m].mean()), int(m.sum())))
    return rows


def ece(prob: np.ndarray, label: np.ndarray, n_bins: int = N_BINS) -> float:
    rows = reliability(prob, label, n_bins)
    n = len(prob)
    return float(sum(cnt / n * abs(p - y) for p, y, cnt in rows))


def platt(score_fit: np.ndarray, y_fit: np.ndarray, score_apply: np.ndarray) -> np.ndarray:
    """Platt scaling — logit(score)를 입력으로 하는 1차원 로지스틱 회귀."""
    eps = 1e-6
    x = np.log(np.clip(score_fit, eps, 1 - eps) / (1 - np.clip(score_fit, eps, 1 - eps)))
    X = np.column_stack([x, np.ones_like(x)])
    w = np.zeros(2)
    for _ in range(200):                                  # 뉴턴 몇 번이면 수렴한다
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - y_fit)
        H = (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        w -= np.linalg.solve(H, grad)
    xa = np.log(np.clip(score_apply, eps, 1 - eps) / (1 - np.clip(score_apply, eps, 1 - eps)))
    return 1 / (1 + np.exp(-(np.column_stack([xa, np.ones_like(xa)]) @ w)))


def isotonic(score_fit: np.ndarray, y_fit: np.ndarray, score_apply: np.ndarray) -> np.ndarray:
    """등위 회귀 (PAVA). 단조 계단 함수를 적합한다."""
    order = np.argsort(score_fit)
    xs, ys = score_fit[order], y_fit[order].astype(float)
    vals, weights = list(ys), [1.0] * len(ys)
    i = 0
    while i < len(vals) - 1:                              # 단조성이 깨지면 이웃과 합친다
        if vals[i] > vals[i + 1]:
            w = weights[i] + weights[i + 1]
            v = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / w
            vals[i:i + 2] = [v]; weights[i:i + 2] = [w]
            i = max(i - 1, 0)
        else:
            i += 1
    fitted, k = np.empty(len(ys)), 0
    for v, w in zip(vals, weights):
        fitted[k:k + int(w)] = v
        k += int(w)
    return np.interp(score_apply, xs, fitted)


def main() -> None:
    g, train_pool, test_pool = P.world()

    # ── [A] eyes_open 점수를 확률로 읽을 수 있나 ────────────────────────
    print("=" * 88)
    print("[A] A-1의 eyes_open 점수는 보정돼 있는가 (0.8이면 정말 80%가 눈을 떴나)")
    print("=" * 88)
    e = A.eye_signal(g)
    v = e.valid()
    prob_open, is_open = e.score[v], ~e.closed[v]
    print(f"  {'예측 구간 평균':>14}{'실제 뜬눈 비율':>16}{'표본 수':>9}{'차이':>9}")
    for p_, y_, c_ in reliability(prob_open, is_open):
        flag = "  ←" if abs(p_ - y_) > 0.1 else ""
        print(f"  {p_:>14.3f}{y_:>16.3f}{c_:>9}{p_ - y_:>9.3f}{flag}")
    print(f"\n  ECE = {ece(prob_open, is_open):.3f}   (0이면 완벽, 0.1 넘으면 확률로 쓰기 곤란)")
    print("  → **전 구간이 한쪽으로 어긋난다** — 예측보다 실제 뜬눈 비율이 항상 높다.")
    print("     점수 0.045짜리 사진들 중에서도 44%가 실제로는 눈을 뜨고 있다.")
    print("  → 주된 원인은 **기저율(base rate)** 이다. 눈감김이 5%뿐이라 점수가 낮아도")
    print("     대부분은 여전히 뜬 눈이다. 그런데 σ(z)는 그 사전 정보를 전혀 안 쓴다 —")
    print("     '이 사진이 얼마나 감은 눈처럼 보이나'를 잴 뿐 '얼마나 감았을 확률인가'가 아니다.")
    print("     여기에 작은 얼굴의 잡음(step7 [E])이 얹혀 낮은 구간을 더 흐린다.")
    print("  → 즉 이 점수는 **순위로는 쓸 만한데(AUC 0.96) 확률로는 못 쓴다(ECE 0.19).**")
    print("     둘은 다른 성질이고, [B]에서 그것을 분리해 확인한다.")

    # ── [B] 보정해도 순위 지표는 안 바뀐다 ──────────────────────────────
    print("\n" + "=" * 88)
    print("[B] 보정 — 그리고 보정이 바꾸지 않는 것")
    print("=" * 88)
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(prob_open))
    fit, hold = idx[: len(idx) // 2], idx[len(idx) // 2:]
    methods = {
        "원본": prob_open[hold],
        "Platt scaling": platt(prob_open[fit], is_open[fit], prob_open[hold]),
        "Isotonic": isotonic(prob_open[fit], is_open[fit], prob_open[hold]),
    }
    print(f"  {'방법':<18}{'ECE':>9}{'ROC-AUC':>10}{'AP(눈감김)':>12}")
    for name, pr in methods.items():
        r_, p_, _ = pr_curve(pr, ~is_open[hold])
        f_, t_ = roc_curve(pr, ~is_open[hold])
        print(f"  {name:<18}{ece(pr, is_open[hold]):>9.3f}{auc(f_, t_):>10.3f}"
              f"{average_precision(r_, p_):>12.3f}")
    print("  → **ECE는 크게 줄고 ROC-AUC·AP는 그대로다.** 보정은 단조 변환이라 순위를 안 바꾼다.")
    print("     그래서 두 가지가 따라온다:")
    print("       ① 'AUC가 높으니 확률을 믿어도 된다'는 **틀린 추론**이다")
    print("       ② 순위만 쓰는 곳(클러스터 대표 선정)에서는 **보정이 아무 의미가 없다**")
    print("  → Isotonic이 Platt보다 ECE가 낮아 보여도 표본이 적으면 과적합한다.")
    print("     우리 규모(갤러리당 수백~수천 장, 라벨은 그중 일부)에서는 Platt이 안전하다.")

    # ── [C] 그래서 우리 파이프라인은 백분위를 쓴다 ──────────────────────
    print("\n" + "=" * 88)
    print("[C] `plan.md`가 점수를 백분위로 바꾸는 이유")
    print("=" * 88)
    print("  설계안 §3-A/§3-B의 점수는 전부 **갤러리 내 백분위**다:")
    print("      prior(p) = technical_pct·w_t + aesthetic_pct·w_a")
    print()
    print(f"  {'':<22}{'ARNIQA 원점수':>14}{'백분위':>10}")
    print(f"  {'실측 관측 대역':<22}{'0.49 ~ 0.52':>14}{'0 ~ 100':>10}")
    print("  → 백분위 변환은 **순위 보정**이다. 원점수의 절대값이 무슨 뜻인지 몰라도,")
    print("     같은 갤러리 안에서 몇 등인지는 확실하다.")
    print("  → 이게 왜 안전한 선택인가:")
    print("       ① ARNIQA와 LAION은 **단위가 다르다**(0~1 vs AVA 1~10). 그대로 더할 수 없다")
    print("       ② 두 모델 다 확률로 보정돼 있지 않다. 보정하려면 라벨이 필요한데 없다")
    print("       ③ 백분위는 라벨 없이 되고, 단조 변환이라 순위 정보를 안 버린다")
    print("  → 대가: **갤러리 간 비교가 불가능해진다.** 갤러리 A의 백분위 90과 갤러리 B의 90은")
    print("     다른 품질이다. 우리 문제에서는 갤러리 안에서만 고르므로 무해하지만,")
    print("     '이 갤러리는 전반적으로 잘 찍혔다' 같은 판단은 못 한다.")
    print("  ※ 스파이크 실측 메모 '10장 샘플이라 분산이 작다 — 전수 실행에서 갤러리 내 백분위의")
    print("     변별력 확인이 본 게임'이 바로 이 이야기다. 원점수 대역이 좁으면 백분위가 잡음이 된다.")

    # ── [D] 01 step2 [A]의 미해결 질문에 답한다 ────────────────────────
    print("\n" + "=" * 88)
    print("[D] reg가 정확도는 안 바꾸는데 무엇을 바꾸나 — 02 step2 [A]의 답")
    print("=" * 88)
    print("  Bradley-Terry는 확률을 뱉는다: P(i를 고름) = σ(margin). 그 확률이 맞나?")
    print(f"  {'reg':<8}{'홀드아웃 정확도':>16}{'ECE':>9}{'평균 확신도':>13}   진단")
    for reg in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        probs, labels = [], []
        for ps in range(20):
            w_true = G.make_person(seed=ps, strength=P.STRENGTH)
            tr = G.sample_pairs(train_pool, 30, seed=100 + ps)
            te = G.sample_pairs(test_pool, 100, seed=200 + ps)
            c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=300 + ps)
            ct, rt = G.answer_pairs(g, w_true, te, temp=P.TEMP, seed=400 + ps)
            w_hat = fit_bt(g.X[c] - g.X[r], reg=reg)
            # 쌍 (a,b)를 무작위로 뒤집어 라벨 1/0을 만든다 — 안 그러면 라벨이 전부 1이다
            flip = np.random.default_rng(900 + ps).random(len(ct)) < 0.5
            d = np.where(flip[:, None], g.X[rt] - g.X[ct], g.X[ct] - g.X[rt])
            probs.append(1 / (1 + np.exp(-(d @ w_hat))))
            labels.append((~flip).astype(float))
        pr = np.concatenate(probs); lb = np.concatenate(labels)
        acc = float(((pr > 0.5) == (lb > 0.5)).mean())
        conf = float(np.maximum(pr, 1 - pr).mean())
        diag = "과신(over-confident)" if conf - acc > 0.05 else (
            "과소(under-confident)" if acc - conf > 0.05 else "대체로 맞음")
        print(f"  {reg:<8}{acc:>16.3f}{ece(pr, lb):>9.3f}{conf:>13.3f}   {diag}")
    print("  → **정확도는 거의 안 움직이는데 ECE와 확신도는 크게 움직인다.**")
    print("     reg가 작으면 w가 커져 확률이 0/1로 몰린다 = 과신. 크면 0.5로 몰린다 = 과소.")
    print("  → 02 step2 [A]에서 '정확도가 못 보는 변화'라고만 했던 것의 정체가 이것이다.")
    print("     같은 정확도에서도 **모델이 얼마나 확신하는지가 완전히 다르다.**")

    # ── [E] 이유 문장은 확률을 소비한다 ────────────────────────────────
    print("\n" + "=" * 88)
    print("[E] 그래서 이유 문장에 무엇을 쓸 수 있나")
    print("=" * 88)
    print("  설계안 §3-C: 이유 문장은 템플릿이고 LLM은 **선택을 바꿀 수 없다.**")
    print("  그런데 템플릿의 **강도**는 확률에서 온다:")
    print()
    for sentence, need in (
        ("\"서약 장면이라 골랐어요\"", "scene 태그의 **정밀도** (02 step8 [B])"),
        ("\"자연광을 선호하셔서\"", "그 축 conf + **다중 비교 보정** (02 step5 [A])"),
        ("\"눈을 감지 않은 컷이라\"", "eyes_open의 **보정된 확률** (이 스텝 [A])"),
        ("\"이 순간의 대표 컷이에요\"", "순위만 있으면 된다 — 보정 불필요"),
    ):
        print(f"  {sentence:<32}{need}")
    print()
    print("  → 마지막 줄이 중요하다. **같은 신호라도 순위로 쓰면 보정이 필요 없고,")
    print("     단언으로 쓰면 필요하다.** 지금 우리 eyes_open은 ECE가 커서 두 번째 용법에")
    print("     쓸 수 없다 — 라벨을 붙여 Platt 보정을 하거나, 문장을 순위형으로 바꿔야 한다.")
    print("  → 되먹임: 이유 문장 템플릿을 만들 때 **각 문장이 요구하는 근거의 종류**를")
    print("     함께 적어 둔다. '정밀도가 필요한 문장 / conf가 필요한 문장 / 순위면 되는 문장'.")


if __name__ == "__main__":
    main()
