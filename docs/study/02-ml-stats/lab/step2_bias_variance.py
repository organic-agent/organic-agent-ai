"""STEP 2 — 편향과 분산: λ와 reg가 실제로 무엇을 사고파는가.

────────────────────────────────────────────────────────────────────────────
이론

1) 분해. 예측 f̂이 데이터에 따라 흔들릴 때, 제곱오차의 기댓값은 셋으로 갈린다.

       E[(f̂ − f)²]  =  (E[f̂] − f)²  +  E[(f̂ − E[f̂])²]  +  잡음
                        └── 편향² ──┘   └──── 분산 ────┘

   편향 = 평균적으로 얼마나 빗나가는가 (모델이 단순해서 생긴다)
   분산 = 데이터를 다시 뽑으면 얼마나 달라지는가 (모델이 복잡해서 생긴다)

2) **여기서 '데이터를 다시 뽑는다'는 것이 우리 서비스에서는 매우 구체적이다** —
   같은 고객이 온보딩을 한 번 더 하면 다른 12쌍을 보고 다르게 답한다.
   분산이 크다는 것은 **같은 고객에게 어제와 오늘 다른 30장을 준다**는 뜻이다.
   고객은 그것을 "이 AI 못 믿겠다"로 읽는다. 분산은 통계 개념이자 UX 문제다.

3) 정규화(reg)와 λ는 둘 다 같은 축 위의 손잡이다.
       reg ↑  → 계수를 0으로 끌어당김 → 편향 ↑ 분산 ↓
       λ   ↓  → 사진학 prior 쪽으로 → 편향 ↑ 분산 ↓   (prior는 개인화가 없으니 편향, 대신 안 흔들림)
   01 step3의 λ = f(evidence)는 **증거가 쌓일수록 편향을 팔고 분산을 사는 스케줄**이다.

4) 진단. 정확도가 낮을 때 무엇을 해야 하는지는 **학습 곡선의 모양**이 알려 준다.
       학습 정확도도 낮다            → 고편향 → 피처·모델을 바꿔라 (쌍을 더 받아도 안 오른다)
       학습은 높은데 홀드아웃이 낮다   → 고분산 → 데이터를 더 받아라 / 정규화를 세게 하라
   "쌍을 더 받을까, 피처를 바꿀까"는 감이 아니라 이 표로 정한다.

실행: ../../.venv/bin/python step2_bias_variance.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

import pairsim as P              # ★ 먼저 import해야 01-recsys/lab이 경로에 올라간다

import gallery as G             # noqa: E402
from step1_bt import fit_bt     # noqa: E402
from step3_lambda import z      # noqa: E402

N_PEOPLE = 12
N_REPEAT = 20          # 같은 사람에게 온보딩을 몇 번 다시 시킬 것인가 (분산의 정의)
TOP_K = 30


def repeated_fits(g, train_pool, person_seed: int, n_train: int, reg: float,
                  n_repeat: int = N_REPEAT) -> list[np.ndarray]:
    """같은 사람에게 **다른 온보딩 쌍**을 주고 여러 번 학습시킨다.

    이것이 '데이터를 다시 뽑는다'의 실체다. 사람도 취향도 그대로인데
    물어본 쌍이 다르고 그날의 대답이 다르다.
    """
    w_true = G.make_person(seed=person_seed, strength=P.STRENGTH)
    out = []
    for rep in range(n_repeat):
        tr = G.sample_pairs(train_pool, n_train, seed=9000 + 31 * rep + person_seed)
        c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=4000 + 17 * rep + person_seed)
        out.append(fit_bt(g.X[c] - g.X[r], reg=reg))
    return out


def main() -> None:
    g, train_pool, test_pool = P.world()
    eval_pairs = G.sample_pairs(test_pool, 300, seed=555)
    D = np.array([g.X[i] - g.X[j] for i, j in eval_pairs])      # (300, 33) 평가용 차이 벡터

    # ── [A] reg 격자에서 편향²과 분산을 직접 센다 ────────────────────────
    print("=" * 88)
    print("[A] 정규화 reg를 키우면 — 편향²은 오르고 분산은 내린다 (학습 30쌍)")
    print("=" * 88)
    # 비교 대상은 **로그 오즈**다. 사람의 선택 확률이 σ((u_i−u_j)/temp)이므로
    # 정답 함수는 f(d) = d·w_true / temp 이고, fit_bt는 바로 그 함수를 추정한다.
    # (여기서 정규화해서 '방향'만 비교하면 reg의 축소 효과가 지워져 분해가 무의미해진다)
    print(f"{'reg':<8}{'편향²':>10}{'분산':>10}{'합':>10}{'홀드아웃 정확도':>16}")
    for reg in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0):
        bias2 = var = acc = 0.0
        for ps in range(N_PEOPLE):
            w_true = G.make_person(seed=ps, strength=P.STRENGTH)
            target = D @ w_true / P.TEMP
            preds = np.array([D @ w for w in repeated_fits(g, train_pool, ps, 30, reg)])
            bias2 += float(((preds.mean(axis=0) - target) ** 2).mean())
            var += float(preds.var(axis=0).mean())
            acc += float(((preds * target) > 0).mean())          # 부호가 맞은 비율
        n = N_PEOPLE
        print(f"{reg:<8}{bias2 / n:>10.3f}{var / n:>10.3f}{(bias2 + var) / n:>10.3f}{acc / n:>16.3f}")
    print("  → reg가 커질수록 **분산은 내려가고 편향²은 올라간다.** 교과서 그림 그대로다.")
    print("     둘의 합(=제곱오차)이 최소인 지점이 있고, 그게 reg를 고르는 기준이다.")
    print("  ※ 그런데 **홀드아웃 정확도는 거의 안 움직인다.** 우리 지표는 마진의 크기가 아니라")
    print("     **부호만** 보기 때문이다. 01 step2 [B]의 '12쌍에서는 어떤 reg도 소용없다'가")
    print("     이것이다 — 정확도가 둔감한 것이지 모델이 안 변한 것이 아니다.")
    print("  → 교훈: **지표가 못 보는 변화가 있다.** 확률값을 쓰는 곳(이유 문장의 '선호하십니다'")
    print("     같은 표현, 점수 간 격차)에서는 이 차이가 드러난다.")

    # ── [B] 분산을 UX로 번역 — 같은 고객, 두 번의 온보딩 ─────────────────
    print("\n" + "=" * 88)
    print(f"[B] 재현성 — 같은 고객이 온보딩을 두 번 하면 상위 {TOP_K}장이 얼마나 겹치나")
    print("=" * 88)
    prior = z((g.tech + g.aes) / 2)
    print(f"{'λ':<8}{'겹침(Jaccard)':>15}{'같은 사진 수':>14}   해석")
    for lam in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        jac, same = [], []
        for ps in range(N_PEOPLE):
            tops = []
            for w_hat in repeated_fits(g, train_pool, ps, 12, reg=1.0, n_repeat=6):
                score = (1 - lam) * prior + lam * z(g.X @ w_hat)
                tops.append(set(np.argsort(-score)[:TOP_K].tolist()))
            for i in range(len(tops)):
                for j in range(i + 1, len(tops)):
                    inter = len(tops[i] & tops[j])
                    jac.append(inter / len(tops[i] | tops[j]))
                    same.append(inter)
        note = "개인화 없음(항상 같은 30장)" if lam == 0 else ""
        print(f"{lam:<8}{np.mean(jac):>15.3f}{np.mean(same):>14.1f}   {note}")
    print("  → λ=0이면 완벽히 재현된다 — 개인화를 안 하니까. λ가 오를수록 겹침이 무너진다.")
    print("     λ=1이면 온보딩을 다시 했을 뿐인데 30장 중 **2장도 안 겹친다**(Jaccard 0.034).")
    print("  → **이것이 분산의 UX 번역이다.** '어제 본 사진이 왜 없어졌지?'")
    print("     λ 상한(0.85)과 λ_min(0.2)은 정확도만의 문제가 아니라 **재현성의 문제**이기도 하다.")
    print("  ※ 실서비스는 온보딩을 두 번 시키지 않는다. 그래도 이 수치는 의미가 있다 —")
    print("     '이 추천이 그 고객의 취향인가, 그날 뽑힌 12쌍인가'의 답이기 때문이다.")

    # ── [C] 학습 곡선 — 무엇을 늘려야 하는가 ─────────────────────────────
    print("\n" + "=" * 88)
    print("[C] 학습 곡선 — '쌍을 더 받을까, 피처를 바꿀까'의 진단표")
    print("=" * 88)
    print(f"{'학습쌍':<8}{'학습 정확도':>12}{'홀드아웃':>10}{'격차':>8}   진단")
    for n_train in (9, 12, 30, 60, 120, 240):
        tr_acc, te_acc = [], []
        for ps in range(N_PEOPLE):
            w_true = G.make_person(seed=ps, strength=P.STRENGTH)
            tr = G.sample_pairs(train_pool, n_train, seed=100 + ps)
            c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=300 + ps)
            w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
            tr_acc.append(float((((g.X[c] - g.X[r]) @ w_hat) > 0).mean()))
            t = P.run_person(ps, n_train, n_hold=200)
            te_acc.append(t.acc)
        tr_m, te_m = np.mean(tr_acc), np.mean(te_acc)
        gap = tr_m - te_m
        diag = "고분산 — 쌍을 더" if gap > 0.15 else ("고편향 — 피처를 바꿔라" if te_m < 0.58 else "균형")
        print(f"{n_train:<8}{tr_m:>12.3f}{te_m:>10.3f}{gap:>8.3f}   {diag}")
    print("  → 9~12쌍 구간은 격차가 크다 = **고분산**. 여기서 할 일은 피처 교체가 아니라 쌍 확보다.")
    print("     쌍이 쌓이면 격차가 줄고 홀드아웃이 오른다. 그러다 어느 지점부터 둘 다 정체한다 —")
    print("     거기가 **이 피처로 갈 수 있는 한계**이고, 그때야 피처를 바꿀 차례다.")
    print("  → 01 step5의 '학습 60쌍' 권고가 이 곡선의 어디인지 확인할 것.")

    # ── [D] 같은 진단을 표현별로 — 태그 33 vs 임베딩 768 ─────────────────
    print("\n" + "=" * 88)
    print("[D] 표현이 바뀌면 진단도 바뀐다 — 01 step2 [A]를 편향-분산으로 다시 읽기")
    print("=" * 88)
    emb = g.emb / g.emb.std()
    print(f"{'표현':<16}{'학습쌍':>8}{'학습':>8}{'홀드아웃':>10}{'격차':>8}")
    for name, X in (("태그 원핫(33)", g.X), ("임베딩(768)", emb)):
        for n_train in (12, 60):
            tr_acc, te_acc = [], []
            for ps in range(N_PEOPLE):
                w_true = G.make_person(seed=ps, strength=P.STRENGTH)
                tr = G.sample_pairs(train_pool, n_train, seed=100 + ps)
                te = G.sample_pairs(test_pool, 200, seed=200 + ps)
                c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=300 + ps)
                ct, rt = G.answer_pairs(g, w_true, te, temp=P.TEMP, seed=400 + ps)
                w_hat = fit_bt(X[c] - X[r], reg=1.0)
                tr_acc.append(float((((X[c] - X[r]) @ w_hat) > 0).mean()))
                te_acc.append(float((((X[ct] - X[rt]) @ w_hat) > 0).mean()))
            tr_m, te_m = np.mean(tr_acc), np.mean(te_acc)
            print(f"{name:<16}{n_train:>8}{tr_m:>8.3f}{te_m:>10.3f}{tr_m - te_m:>8.3f}")
    print("  → 임베딩은 학습 정확도가 1.0에 붙는다. **12쌍을 완벽히 외운다**(p ≫ n).")
    print("     그런데 홀드아웃은 안 오른다 — 전형적인 고분산이고, 여기서는 데이터를")
    print("     더 받아도(60쌍) 격차가 거의 안 줄어든다. 768차원에 60쌍은 여전히 p ≫ n이다.")
    print("  → **그래서 표현을 바꿨다**(태그 우선). 진단이 처방을 정한 사례다.")


if __name__ == "__main__":
    main()
