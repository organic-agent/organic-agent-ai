"""STEP 1 — 구간이 진짜로 덮는가: 커버리지로 추정량을 고르는 법.

────────────────────────────────────────────────────────────────────────────
이론

01 step5에서 우리는 Wilson 신뢰구간을 썼고, 이유를 이렇게 적었다 —
"정규근사는 n이 작을 때 구간이 좁거나 [0,1]을 벗어난다". **그건 주장이었지 측정이 아니었다.**
02는 그 주장을 잰다.

1) 신뢰구간의 정의는 '확률'이 아니라 **커버리지(coverage)** 다.
   "95% 신뢰구간"은 "이 구간 안에 진짜 값이 있을 확률이 95%"가 아니다.
   **같은 실험을 무한히 반복하면 구간의 95%가 진짜 값을 포함한다**는 뜻이다.
   그래서 좋은 구간인지 아닌지는 **시뮬레이션으로 셀 수 있다** — 진짜 값을 아는 세계에서
   구간을 1만 개 만들어 몇 개가 덮는지 세면 된다. 이 실습이 그것이다.

2) 정규근사(Wald) 구간:  p̂ ± z·√(p̂(1−p̂)/n)
   문제 두 가지. ① p̂이 0이나 1이면 폭이 0이 된다(!). ② 표본 비율의 분포가
   이산이라 근사가 안 맞고, 실제 커버리지가 명목 95%보다 **낮다**.

3) Wilson 구간: p̂ 대신 **참값 p를 미지수로 두고** |p̂ − p| ≤ z·√(p(1−p)/n)를 p에 대해 푼다.
   분산에 p̂이 아니라 p가 들어가는 것이 차이의 전부다. 결과적으로 구간이 0.5 쪽으로
   살짝 밀려(shrink) 소표본에서도 커버리지가 유지된다.

4) Clopper-Pearson: 이항분포를 그대로 뒤집어 만든 '정확(exact)' 구간.
   커버리지가 절대 95% 아래로 안 내려가지만, 그 대가로 **필요 이상 넓다**(보수적).

5) 톱니(oscillation). 이항은 이산분포라 커버리지가 n이 커져도 매끄럽게 수렴하지 않고
   **진동한다.** "n이 30이 넘으면 정규근사 OK" 같은 경험칙이 위험한 이유다.

우리 결정과의 연결: 스파이크 리포트와 W1–2 실험 결과는 전부 **비율**이다
(쌍 정확도·수락률·후회율·태그 축별 정확도). 그 비율에 어떤 구간을 붙일지가
여기서 정해진다. 그리고 08-25 VLM 채점(50장, 축별 정확도)에도 그대로 적용된다.

실행: ../../.venv/bin/python step1_interval.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from scipy import stats

import pairsim as P
from metrics import wilson_interval          # 01 lab에서 그대로 빌려 쓴다

Z = stats.norm.ppf(0.975)


# ── 구간 세 종류 ────────────────────────────────────────────────────────────
def wald_interval(k: int, n: int) -> tuple[float, float]:
    """정규근사(Wald). 교과서 첫 페이지에 나오고, 소표본에서 틀린다."""
    p = k / n
    half = Z * np.sqrt(p * (1 - p) / n)
    return (p - half, p + half)


def clopper_pearson(k: int, n: int, conf: float = 0.95) -> tuple[float, float]:
    """이항분포를 직접 뒤집은 '정확' 구간. 베타분포의 분위수로 계산된다."""
    a = 1 - conf
    lo = stats.beta.ppf(a / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - a / 2, k + 1, n - k) if k < n else 1.0
    return (float(lo), float(hi))


METHODS = {
    "Wald(정규근사)": wald_interval,
    "Wilson": wilson_interval,
    "Clopper-Pearson": clopper_pearson,
}


def coverage(method, n: int, p_true: float, reps: int = 20000, seed: int = 0) -> tuple[float, float]:
    """진짜 비율이 p_true일 때, 그 방법의 구간이 p_true를 덮는 비율과 평균 폭.

    이항 표본을 reps번 뽑아 매번 구간을 만들고 '덮었나'를 센다. 이게 커버리지의 정의다.
    """
    rng = np.random.default_rng(seed)
    ks = rng.binomial(n, p_true, size=reps)
    hit = 0
    width = 0.0
    for k in ks:
        lo, hi = method(int(k), n)
        hit += lo <= p_true <= hi
        width += hi - lo
    return hit / reps, width / reps


def main() -> None:
    # ── [A] 명목 95%가 실제로 몇 %인가 ──────────────────────────────────
    print("=" * 84)
    print("[A] '95% 신뢰구간'의 실제 커버리지 — 진짜 값을 아는 세계에서 2만 번 세어 본다")
    print("=" * 84)
    print(f"{'n':<6}{'진짜 p':<9}" + "".join(f"{m:>18}" for m in METHODS))
    for n in (24, 50, 100, 300):
        for p_true in (0.55, 0.62):
            row = f"{n:<6}{p_true:<9}"
            for m in METHODS.values():
                cov, _ = coverage(m, n, p_true)
                mark = " " if cov >= 0.93 else "!"
                row += f"{cov:>17.3f}{mark}"
            print(row)
    print("  → Wald는 명목 95%인데 실제로는 0.90 언저리다. **10번 중 1번은 진짜 값을 놓친다.**")
    print("     '유의하다'고 선언할 때 그 판단이 우리가 생각한 것보다 자주 틀린다는 뜻이다.")
    print("     Wilson은 소표본에서도 0.94~0.96을 지킨다. Clopper-Pearson은 0.97+ — 과보호다.")

    # ── [B] 폭 — 안전을 얼마에 사는가 ───────────────────────────────────
    print("\n" + "=" * 84)
    print("[B] 평균 구간 폭 — 커버리지를 사는 값")
    print("=" * 84)
    print(f"{'n':<6}" + "".join(f"{m:>18}" for m in METHODS))
    for n in (24, 50, 100, 300):
        row = f"{n:<6}"
        for m in METHODS.values():
            _, w = coverage(m, n, 0.62)
            row += f"{w:>18.3f}"
        print(row)
    print("  → Clopper-Pearson은 Wilson보다 항상 넓다. 커버리지를 초과 달성한 만큼")
    print("     '결론을 못 내는' 쪽으로 손해를 본다. 우리처럼 표본이 귀한 실험에서는")
    print("     **Wilson이 커버리지와 폭의 균형점**이다 — 01 step5가 Wilson을 쓴 이유가 이것이다.")

    # ── [C] 톱니 — "n>30이면 정규근사 OK"가 위험한 이유 ──────────────────
    print("\n" + "=" * 84)
    print("[C] 커버리지는 n이 커져도 매끄럽게 수렴하지 않는다 (이산성의 톱니)")
    print("=" * 84)
    print(f"  진짜 p=0.62 · Wald 구간의 실제 커버리지")
    line = ""
    for n in range(20, 61, 4):
        cov, _ = coverage(wald_interval, n, 0.62, reps=20000)
        line += f"  n={n}:{cov:.3f}"
        if n % 20 == 0:
            print(line); line = ""
    if line:
        print(line)
    print("  → 위아래로 튄다. n을 4씩 늘렸을 뿐인데 커버리지가 오르내린다.")
    print("     '표본이 30 넘으면 정규근사 써도 된다'는 경험칙이 왜 위험한지 보여 준다.")

    # ── [D] 우리 실험에 적용 — 쌍을 세면 안 되고 사람을 세야 한다 ─────────
    print("\n" + "=" * 84)
    print("[D] 사람이 표본 단위다 — 01 step5 [D]의 주장을 커버리지로 검증")
    print("=" * 84)
    n_people, n_hold, n_train, reps = 8, 60, 30, 300

    # 모수 μ를 정확히 정의하는 것이 이 실험의 절반이다.
    # 우리가 재려는 것은 '정답 취향 벡터의 정확도'가 아니라
    # **"학습 30쌍으로 만든 모델이 홀드아웃 쌍을 맞히는 비율"의 사람 모집단 평균**이다.
    # 추정 대상을 잘못 잡으면 커버리지 표는 아무 의미가 없다.
    big = P.run_group(400, n_train=n_train, n_hold=n_hold, seed_base=100_000)
    k_mu, n_mu = P.pooled(big)
    mu = k_mu / n_mu
    print(f"  모수 μ = {mu:.3f}  = 사람 400명 × {n_hold}쌍으로 근사한 '학습 {n_train}쌍 모델'의 평균 정확도")
    print(f"           (정답 취향 벡터였다면 {np.mean([P.true_accuracy(s) for s in range(50)]):.3f} — 그건 우리가 재는 값이 아니다)")
    print(f"  실험 = 사람 {n_people}명 × 홀드아웃 {n_hold}쌍, {reps}회 반복\n")

    hit_binom = hit_boot = 0
    w_binom = w_boot = 0.0
    rng = np.random.default_rng(0)
    for rep in range(reps):
        trials = P.run_group(n_people, n_train=n_train, n_hold=n_hold, seed_base=rep * n_people)
        accs = np.array([t.acc for t in trials])

        k, n = P.pooled(trials)                       # ① 쌍을 세는 방식
        lo, hi = wilson_interval(k, n)
        hit_binom += lo <= mu <= hi
        w_binom += hi - lo

        boot = accs[rng.integers(0, n_people, (2000, n_people))].mean(axis=1)   # ② 사람을 세는 방식
        blo, bhi = np.percentile(boot, [2.5, 97.5])
        hit_boot += blo <= mu <= bhi
        w_boot += bhi - blo

    print(f"{'방법':<28}{'커버리지':>10}{'평균 폭':>10}")
    print(f"{'쌍 단위 Wilson (풀링)':<28}{hit_binom / reps:>10.3f}{w_binom / reps:>10.3f}")
    print(f"{'사람 단위 부트스트랩':<28}{hit_boot / reps:>10.3f}{w_boot / reps:>10.3f}")
    print("  → 쌍을 세면 구간이 좁고 **커버리지가 명목 95%에 한참 못 미친다.**")
    print("     사람마다 잘 맞는 정도가 다른데(사람 간 분산) 그것을 0으로 친 결과다.")
    print("     01 step5 [D]는 '구간이 1.5배 좁다'고만 했다. 여기서 그게 **틀린 구간**임이 확인된다.")
    print("  → plan.md §6의 '구간은 사람 단위 부트스트랩으로 잡는다'가 이 표를 근거로 한다.")
    print("  ※ 부트스트랩도 8명에서는 완벽하지 않다. 사람이 적으면 재표집할 원본이 부족하다 —")
    print("     구간 방법을 바꿔서 살 수 있는 것에는 한계가 있고, 그 다음은 사람을 늘리는 일이다.")


if __name__ == "__main__":
    main()
