"""STEP 5 — 실험 설계와 지표: 62%가 우연과 구분되는가, 몇 명 × 몇 쌍이 필요한가.

────────────────────────────────────────────────────────────────────────────
이론

1) 주지표는 홀드아웃 쌍 예측 정확도이고 기준선은 정확히 0.5다. 그래서 판단은
   "정확도가 몇이냐"가 아니라 **"0.5와 구분되느냐"** 다. 도구는 두 개:
     · 신뢰구간 (Wilson) — 이 표본에서 진짜 값이 있을 만한 범위
     · 이항검정 p값 — "진짜가 0.5인데 운으로 이만큼 나올 확률"

2) 검정력(power)과 표본 수. 진짜 정확도가 p₁일 때 유의하다고 결론 낼 확률이 검정력이다.
   단측 검정 정규근사로 필요한 표본 수는
       n ≈ [ (z_α·√(p₀(1−p₀)) + z_β·√(p₁(1−p₁))) / (p₁ − p₀) ]²
   차이(p₁−p₀)가 작을수록 **제곱으로** 커진다. 0.55를 잡으려면 수백 쌍이 필요하다.

3) 군집 데이터(clustered data). 같은 사람이 답한 쌍들은 서로 독립이 아니다.
   어떤 사람은 취향이 뚜렷해 잘 맞고 어떤 사람은 아니다. 이 사람 간 분산을 무시하고
   전체 쌍을 하나의 이항으로 세면 신뢰구간이 **실제보다 좁게** 나온다(자신 과잉).
   대응: 사람 단위 부트스트랩 — 사람을 복원추출해서 구간을 만든다.

4) 12쌍밖에 없을 때: leave-one-out. 12쌍 중 1쌍을 빼고 학습 → 그 1쌍 예측을 12번.
   홀드아웃 쌍을 낭비하지 않는 대신, 학습·평가가 겹쳐 낙관적으로 나오는 편향이 있다.

5) 증거의 **질**도 표본 수를 정한다. 필요 표본 수는 (p₁ − 0.5)의 제곱에 반비례하므로,
   태그 오차로 p₁이 조금만 내려가도 필요한 쌍이 폭발한다(실험 G). step3 [E]에서 본
   2026-08-25 VLM 실측 오차가 여기서 **실험 인원·쌍 수**라는 실무 결정으로 번역된다.

6) 랭킹 지표는 실사용 단계의 것이다. nDCG는 "상위에 좋은 것이 있는가"를 로그 할인으로
   재고, 수락률·후회율은 실제 행동 로그로 잰다. 쌍 정확도가 오프라인 대리 지표라면
   이쪽이 진짜 지표다 — 대리 지표만 올리는 튜닝을 경계할 것.

실행: ../../.venv/bin/python step5_eval.py   (실측 약 6초)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

import gallery as G
from metrics import (
    accept_rate,
    binom_p_value,
    ndcg,
    pairwise_accuracy,
    regret_rate,
    required_pairs,
    wilson_interval,
)
from step1_bt import fit_bt

STRENGTH, TEMP = 1.5, 0.7


def main() -> None:
    # ── [A] 눈앞의 숫자를 판정하기 ───────────────────────────────────────
    print("=" * 78)
    print("[A] '홀드아웃 정확도 62%' — 표본 수에 따라 의미가 완전히 달라진다")
    print("=" * 78)
    print(f"{'표본(쌍)':<10}{'맞힌 수':>8}{'정확도':>9}{'95% CI':>20}{'p값':>9}  판정")
    for n in (24, 50, 100, 300, 1000):
        k = int(round(0.62 * n))
        lo, hi = wilson_interval(k, n)
        p = binom_p_value(k, n)
        verdict = "우연과 구분 안 됨" if p >= 0.05 else "유의 (0.5 초과)"
        print(f"{n:<10}{k:>8}{k / n:>9.3f}{f'[{lo:.3f}, {hi:.3f}]':>20}{p:>9.4f}  {verdict}")
    print("  → 같은 62%인데 24쌍이면 아무 말도 못 하고 100쌍이면 말할 수 있다.")
    print("     리포트에 정확도만 쓰고 표본 수를 안 쓰면 그 숫자는 읽을 수 없는 숫자다.")

    print("\n" + "=" * 78)
    print("[B] 필요한 홀드아웃 쌍 수 (단측 α=0.05)")
    print("=" * 78)
    print(f"{'진짜 정확도':<12}{'검정력 0.8':>12}{'검정력 0.9':>12}")
    for p1 in (0.55, 0.60, 0.65, 0.70, 0.80):
        print(f"{p1:<12}{required_pairs(p1, power=0.8):>12}{required_pairs(p1, power=0.9):>12}")
    print("  → 설계안 §6의 '12쌍 중 9쌍 학습 → 3쌍 검증, 5~10명'은 홀드아웃이 15~30쌍이다.")
    print("     위 표와 대면 0.65~0.70이 나와도 유의성을 못 만든다. 실험 설계를 바꿔야 한다.")

    # ── [C] 실제 파이프라인으로 검정력 시뮬레이션 ────────────────────────
    print("\n" + "=" * 78)
    print("[C] 그럼 몇 명 × 몇 쌍인가 — 실제 학습·평가를 돌려 본 검정력")
    print("     (사람마다 학습 쌍으로 취향을 배우고 홀드아웃 쌍을 맞힌다. 300회 반복)")
    print("=" * 78)
    g = G.make_gallery(1000, seed=0)
    train_pool, test_pool = G.split_pool(G.design_pair_pool(g), 0.5, seed=7)
    print(f"{'사람 수':<8}{'학습쌍/인':>10}{'홀드아웃/인':>12}{'총 홀드아웃':>12}{'평균 정확도':>12}{'검정력':>9}")
    for n_people, n_train, n_hold in [
        (8, 9, 3),        # 설계안 §6이 적어 둔 그대로
        (8, 30, 20),
        (8, 30, 50),
        (10, 60, 60),
        (20, 30, 50),
    ]:
        hits, totals, significant = [], [], 0
        reps = 300
        for rep in range(reps):
            k_all = n_all = 0
            for pi in range(n_people):
                w = G.make_person(seed=10_000 * rep + pi, strength=STRENGTH)
                tr = G.sample_pairs(train_pool, n_train, seed=7 * rep + pi)
                te = G.sample_pairs(test_pool, n_hold, seed=13 * rep + pi)
                c, r = G.answer_pairs(g, w, tr, temp=TEMP, seed=3 * rep + pi)
                ct, rt = G.answer_pairs(g, w, te, temp=TEMP, seed=5 * rep + pi)
                w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
                acc = pairwise_accuracy(w_hat, g.X, ct, rt)
                k_all += int(round(acc * len(ct)))
                n_all += len(ct)
            hits.append(k_all); totals.append(n_all)
            if binom_p_value(k_all, n_all) < 0.05:
                significant += 1
        acc_mean = float(np.sum(hits) / np.sum(totals))
        power = significant / reps
        print(f"{n_people:<8}{n_train:>10}{n_hold:>12}{n_people * n_hold:>12}{acc_mean:>12.3f}{power:>9.2f}")
    print("  → 사람 수보다 **사람당 쌍 수**가 검정력을 만든다. 학습 쌍이 적으면 정확도 자체가")
    print("     안 오르고, 홀드아웃이 적으면 오른 정확도를 증명하지 못한다. 둘 다 늘려야 한다.")
    print("  → 실무 번역: '팀원 8명에게 각각 80쌍(약 10~15분) 답하게 한다'가 현실적인 설계다.")

    # ── [D] 사람 간 분산 — 순진한 이항 구간은 좁다 ───────────────────────
    print("\n" + "=" * 78)
    print("[D] 같은 사람의 쌍은 독립이 아니다 — 이항 CI vs 사람 단위 부트스트랩 CI")
    print("=" * 78)
    n_people, n_train, n_hold = 8, 30, 50
    per_person = []
    for pi in range(n_people):
        w = G.make_person(seed=500 + pi, strength=STRENGTH)
        tr = G.sample_pairs(train_pool, n_train, seed=100 + pi)
        te = G.sample_pairs(test_pool, n_hold, seed=200 + pi)
        c, r = G.answer_pairs(g, w, tr, temp=TEMP, seed=300 + pi)
        ct, rt = G.answer_pairs(g, w, te, temp=TEMP, seed=400 + pi)
        w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
        per_person.append(pairwise_accuracy(w_hat, g.X, ct, rt))
    per_person = np.array(per_person)
    k = int(round(per_person.mean() * n_people * n_hold))
    n = n_people * n_hold
    lo, hi = wilson_interval(k, n)
    rng = np.random.default_rng(0)
    boot = [per_person[rng.integers(0, n_people, n_people)].mean() for _ in range(5000)]
    blo, bhi = np.percentile(boot, [2.5, 97.5])
    print(f"  사람별 정확도: {np.round(per_person, 3)}")
    print(f"  전체 평균 {per_person.mean():.3f}  (총 {n}쌍)")
    print(f"  순진한 이항 95% CI      [{lo:.3f}, {hi:.3f}]   폭 {hi - lo:.3f}")
    print(f"  사람 단위 부트스트랩 CI  [{blo:.3f}, {bhi:.3f}]   폭 {bhi - blo:.3f}")
    print("  → 사람마다 잘 맞는 정도가 다르다. 그 분산을 무시하면 구간이 좁게 나와")
    print("     '유의하다'고 잘못 선언하기 쉽다. 사람이 표본 단위이므로 사람 단위로 재표집한다.")

    # ── [E] 12쌍뿐일 때: leave-one-out ──────────────────────────────────
    print("\n" + "=" * 78)
    print("[E] 온보딩 12쌍만 있을 때 — leave-one-out으로 짜내기")
    print("=" * 78)
    accs_loo, accs_split = [], []
    for pi in range(30):
        w = G.make_person(seed=900 + pi, strength=STRENGTH)
        pr = G.sample_pairs(train_pool, 12, seed=900 + pi)
        c, r = G.answer_pairs(g, w, pr, temp=TEMP, seed=901 + pi)
        hit = 0
        for i in range(12):
            m = np.ones(12, dtype=bool); m[i] = False
            w_hat = fit_bt(g.X[c[m]] - g.X[r[m]], reg=1.0)
            hit += int((g.X[c[i]] - g.X[r[i]]) @ w_hat > 0)
        accs_loo.append(hit / 12)
        w_hat = fit_bt(g.X[c[:9]] - g.X[r[:9]], reg=1.0)
        accs_split.append(pairwise_accuracy(w_hat, g.X, c[9:], r[9:]))
    print(f"  9/3 분할 평균 정확도 : {np.mean(accs_split):.3f}  (사람당 홀드아웃 3쌍)")
    print(f"  LOO 평균 정확도      : {np.mean(accs_loo):.3f}  (사람당 12번 평가)")
    print("  → LOO는 평가 횟수를 4배로 늘려 준다. 다만 매번 11쌍으로 학습하므로 실제 12쌍")
    print("     모델보다 조금 비관적이고, 같은 12쌍을 재사용하므로 사람 간 독립성은 그대로다.")
    print("     '표본이 늘었다'가 아니라 '측정 잡음이 줄었다'로 읽어야 한다.")

    # ── [F] 실사용 지표 ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[F] 실사용 지표 — nDCG · 수락률 · 후회율 (숫자로 감 잡기)")
    print("=" * 78)
    gains_good = np.array([1, 1, 1, 0, 1, 0, 0, 1, 0, 0])   # 상위에 수락이 몰린 경우
    gains_bad = gains_good[::-1].copy()                      # 같은 수락 수, 순서만 반대
    print(f"  수락 패턴 A(상위 집중) {gains_good.tolist()} → nDCG@10 {ndcg(gains_good):.3f}")
    print(f"  수락 패턴 B(하위 집중) {gains_bad.tolist()} → nDCG@10 {ndcg(gains_bad):.3f}")
    print(f"  두 경우의 수락률은 {accept_rate(gains_good):.2f}로 같다 — 수락률은 순서를 못 본다.")
    print("  → 그래서 두 지표를 함께 본다. '설득력은 순서에 있다'가 nDCG를 쓰는 이유.")
    accepted = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 1], dtype=bool)
    unselected = np.array([0, 1, 0, 0, 1, 0, 0, 0, 0, 0], dtype=bool)
    print(f"  수락률 {accept_rate(accepted):.2f} · 후회율 {regret_rate(accepted, unselected):.2f}")
    print("  → 수락률만 보면 0.70으로 좋아 보이지만 담은 것의 29%를 다시 뺐다.")
    print("     '전체 수락' 버튼 하나로 수락률은 얼마든지 부풀릴 수 있다 — 후회율이 그 방패다.")

    experiment_g()


def holdout_accuracy(g: G.Gallery, n_train: int, n_hold: int = 50, n_people: int = 40) -> float:
    """사람 n_people명의 평균 홀드아웃 쌍 정확도. 쌍 규칙 풀은 학습/평가로 분리한다."""
    train_pool, test_pool = G.split_pool(G.design_pair_pool(g), 0.5, seed=7)
    accs = []
    for pi in range(n_people):
        w = G.make_person(seed=500 + pi, strength=STRENGTH)
        tr = G.sample_pairs(train_pool, n_train, seed=100 + pi)
        te = G.sample_pairs(test_pool, n_hold, seed=200 + pi)
        c, r = G.answer_pairs(g, w, tr, temp=TEMP, seed=300 + pi)
        ct, rt = G.answer_pairs(g, w, te, temp=TEMP, seed=400 + pi)
        w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
        accs.append(pairwise_accuracy(w_hat, g.X, ct, rt))
    return float(np.mean(accs))


def experiment_g() -> None:
    """[G] 태그 오차 → 정확도 하락 → 필요 표본 수 폭발.

    [B]는 '진짜 정확도가 p₁이면 몇 쌍 필요한가'를 표로만 줬다. 그럼 **우리의 p₁은
    얼마인가**? 그건 학습 쌍 수와 **태그 정확도**가 정한다. step3 [E]의 실측 오차를
    그대로 넣어 두 세계를 나란히 재 본다.
    """
    print("\n" + "=" * 78)
    print("[G] 태그 오차가 실험 규모를 정한다 — step3 [E]의 실측 오차를 넣으면")
    print("    scene 74% · lighting 84% · expression 90% · framing 92% · subjects 98%")
    print("=" * 78)
    g_ok = G.make_gallery(1000, seed=0)
    g_err = G.make_gallery(1000, seed=0, tag_error=1.0)
    print(f"{'학습쌍/인':<12}{'태그 완벽':>10}{'필요 홀드아웃':>14}{'실측 오차':>11}{'필요 홀드아웃':>14}")
    for n_train in (12, 30, 60):
        a_ok = holdout_accuracy(g_ok, n_train)
        a_er = holdout_accuracy(g_err, n_train)
        n_ok = required_pairs(round(a_ok, 2))
        n_er = required_pairs(round(a_er, 2))
        f_ok = f"{n_ok}" if n_ok > 0 else "불가"
        f_er = f"{n_er}" if n_er > 0 else "불가(≤0.5)"
        print(f"{n_train:<12}{a_ok:>10.3f}{f_ok:>14}{a_er:>11.3f}{f_er:>14}")
    print("  → 학습 12쌍 + 태그 오차면 정확도가 **0.5 아래**다. 표본을 아무리 늘려도")
    print("     증명할 것이 없다. 온보딩 12쌍만으로 실험을 설계하면 안 되는 이유다.")
    print("  → 학습 30쌍에서 필요 홀드아웃이 153 → 428쌍으로 뛴다(약 2.8배).")
    print("     필요 표본 수는 (p₁−0.5)의 **제곱**에 반비례하므로, 정확도가 조금 내려가면")
    print("     실험 규모가 크게 는다. 태그 품질은 학습 문제이기 전에 **실험 예산 문제**다.")
    print("  ※ 이 '필요 홀드아웃'은 사람 간 분산을 무시한 **하한**이다. [D]에서 부트스트랩")
    print("     구간이 이항 구간의 약 1.5배였으므로, 실제로는 2배 이상 잡아야 안전하다.")


if __name__ == "__main__":
    main()
