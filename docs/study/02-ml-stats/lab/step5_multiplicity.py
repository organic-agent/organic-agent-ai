"""STEP 5 — 여러 번 보면 뭔가는 나온다: 다중 비교와 효과 크기.

────────────────────────────────────────────────────────────────────────────
이론

01 step5는 검정 **하나**를 다뤘다. 실제 리포트는 한 번도 그런 적이 없다 —
08-25 VLM 채점은 축 6개의 정확도를 한 표에 적었고, 실사용 지표는 수락률·nDCG·후회율·
라운드 수를 함께 본다. **볼 때마다 우연히 유의할 기회가 생긴다.**

1) 가족 단위 오류율(FWER). 진짜로 아무 효과가 없어도, 독립인 검정 m개를 하면
   적어도 하나가 α 수준에서 유의할 확률은

       1 − (1 − α)^m       α=0.05, m=5 → 0.23      m=15 → 0.54

   **15번 비교하면 절반 이상의 확률로 "발견"이 나온다.** 아무것도 없는데도.

2) 보정 두 가지.
   · **Bonferroni**: α를 m으로 나눈다. FWER을 확실히 α 아래로 누르지만 매우 보수적이다.
   · **Benjamini-Hochberg(BH)**: '유의하다고 말한 것 중 틀린 비율'(FDR)을 통제한다.
     탐색적 분석에 맞다 — 후보를 추리는 단계라면 이쪽.
   무엇을 통제할지가 먼저다. **"하나라도 틀리면 안 된다"면 FWER, "대체로 맞으면 된다"면 FDR.**

3) 유의성 ≠ 크기. p값은 "효과가 있나"이지 "얼마나 크냐"가 아니다. n이 크면
   아무리 작은 차이도 유의해진다. 그래서 **효과 크기**와 **실용적 기준**을 함께 적는다.

       "유의하다(p=0.03). 다만 차이는 1.2%p로, 우리가 의미 있다고 정한 5%p에 못 미친다."

4) 사전 등록(pre-registration). 무엇을 주지표로 삼을지, 어떤 차이를 의미 있다고 볼지를
   **데이터를 보기 전에** 적어 두면 다중 비교의 절반이 사라진다. 데이터를 본 뒤
   지표를 고르는 것이 가장 흔한 자기기만이다.

실행: ../../.venv/bin/python step5_multiplicity.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy import stats

import pairsim as P              # ★ 먼저 import

import gallery as G              # noqa: E402
from metrics import wilson_interval   # noqa: E402
from step1_bt import fit_bt      # noqa: E402

# 2026-08-25 VLM 고정 축 태그 채점 (spike-report.md). 50장 × 6항목.
VLM_GRADE = {
    "subjects": 49, "framing": 46, "expression": 45,
    "caption": 43, "lighting": 42, "scene": 37,
}
VLM_N = 50


def bonferroni(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    return pvals <= alpha / len(pvals)


def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """FDR 통제. 정렬한 p값을 i·α/m과 비교해 마지막으로 통과한 지점까지 기각."""
    m = len(pvals)
    order = np.argsort(pvals)
    thresh = (np.arange(1, m + 1) / m) * alpha
    passed = pvals[order] <= thresh
    out = np.zeros(m, dtype=bool)
    if passed.any():
        last = np.max(np.where(passed)[0])
        out[order[: last + 1]] = True
    return out


def cohens_h(p1: float, p2: float) -> float:
    """비율 두 개의 효과 크기. 0.2 작음 / 0.5 중간 / 0.8 큼 (관례)."""
    return float(2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2)))


def main() -> None:
    # ── [A] 아무 효과가 없어도 뭔가는 유의하다 ───────────────────────────
    print("=" * 86)
    print("[A] 취향이 전혀 없는 고객 — 축 5개를 검정하면 몇 번이나 '발견'되나")
    print("=" * 86)
    g, train_pool, test_pool = P.world()
    reps = 300
    any_sig = per_axis = 0
    for rep in range(reps):
        # strength=0 → 진짜 취향이 없다. 모든 축의 참 정확도가 정확히 0.5다.
        w_true = np.zeros(G.D_TAG)
        tr = G.sample_pairs(train_pool, 30, seed=1000 + rep)
        te = G.sample_pairs(test_pool, 300, seed=2000 + rep)
        c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=3000 + rep)
        ct, rt = G.answer_pairs(g, w_true, te, temp=P.TEMP, seed=4000 + rep)
        w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
        sig = 0
        for ax in G.AXES:
            hit = tot = 0
            for a, b in zip(ct, rt):
                if g.tags[a][ax] == g.tags[b][ax]:
                    continue
                tot += 1
                hit += int((g.X[a] - g.X[b]) @ w_hat > 0)
            if tot and stats.binomtest(hit, tot, 0.5, alternative="greater").pvalue < 0.05:
                sig += 1
        per_axis += sig
        any_sig += sig > 0
    print(f"  고객의 진짜 취향 = 없음(모든 축의 참 정확도 0.5) · {reps}회 반복")
    print(f"  축 하나가 유의할 비율        : {per_axis / (reps * 5):.3f}   (기대값 0.05)")
    print(f"  **5개 중 최소 하나**가 유의할 비율: {any_sig / reps:.3f}   (독립이라면 이론값 0.23)")
    print("  → 취향이 하나도 없는데도 '이 고객은 ○○축을 선호한다'가 심심찮게 나온다.")
    print("     이유 문장이 그 축을 근거로 쓰면 고객은 '우리가 언제?'라고 느낀다.")
    print("     conf(axis)가 있어도 이건 못 막는다 — conf는 세기를 줄일 뿐 검정이 아니다.")
    print("  ※ 실측이 이론값 0.23보다 낮다. 이유가 둘이고, 둘 다 알아 둘 값어치가 있다.")
    print("     ① 이항검정은 이산이라 실제 1종 오류율이 α보다 **낮다**(보수적) — 02 step1 [C]의")
    print("        톱니와 같은 원인이다. 그래서 축 하나의 유의 비율도 0.05에 못 미친다.")
    print("     ② 축 검정들이 **독립이 아니다.** 같은 ŵ로 모든 축을 재니 한 축이 운 좋게")
    print("        맞으면 다른 축도 함께 맞는 경향이 있다. 1−(1−α)^m은 독립 가정의 공식이다.")
    print("  → 교훈: 다중 비교의 실제 위험도 **이론식이 아니라 시뮬레이션으로** 재는 것이 정확하다.")

    # ── [B] 실제 리포트에 적용 — 08-25 VLM 채점 ─────────────────────────
    print("\n" + "=" * 86)
    print("[B] 2026-08-25 VLM 채점(50장 × 6항목)에 다중 비교를 적용하면")
    print("=" * 86)
    print("  질문 1: '이 축은 90% 기준을 넘는가' — 6번 검정한다")
    names = list(VLM_GRADE)
    ks = np.array([VLM_GRADE[a] for a in names])
    pvals = np.array([stats.binomtest(int(k), VLM_N, 0.90, alternative="less").pvalue for k in ks])
    bon = bonferroni(pvals)
    bh = benjamini_hochberg(pvals)
    print(f"{'축':<12}{'정확도':>9}{'95% CI':>18}{'p(<90%)':>10}{'단순':>7}{'Bonf.':>8}{'BH':>5}")
    for i, a in enumerate(names):
        lo, hi = wilson_interval(int(ks[i]), VLM_N)
        mark = lambda b: "  ✓" if b else "  ·"
        print(f"{a:<12}{ks[i] / VLM_N:>9.2f}{f'[{lo:.2f}, {hi:.2f}]':>18}{pvals[i]:>10.4f}"
              f"{mark(pvals[i] < 0.05):>7}{mark(bon[i]):>8}{mark(bh[i]):>5}")
    print("  → ✓ = '90%에 못 미친다'고 말할 수 있는 축. 보정하면 목록이 줄어든다.")
    print("     **`scene`만 어떤 기준으로도 살아남는다** — 그래서 프롬프트 개선의 1순위가 scene이다.")
    print("     `lighting` 84%는 단순 p값으로는 유의하지만 Bonferroni는 못 넘는다.")
    print("  ※ 50장으로는 구간이 ±0.1 수준이다. '92% vs 90%'를 구분할 표본이 아니다.")

    print("\n  질문 2: '축 A가 축 B보다 정확한가' — 6개 축의 쌍 비교는 15번이다")
    pairs = list(combinations(range(len(names)), 2))
    pp = []
    for i, j in pairs:
        tbl = [[ks[i], VLM_N - ks[i]], [ks[j], VLM_N - ks[j]]]
        pp.append(stats.fisher_exact(tbl)[1])
    pp = np.array(pp)
    n_simple, n_bon, n_bh = (pp < 0.05).sum(), bonferroni(pp).sum(), benjamini_hochberg(pp).sum()
    print(f"    비교 수 15 · 단순 p<0.05: {n_simple}쌍 · Bonferroni: {n_bon}쌍 · BH: {n_bh}쌍")
    print(f"    (아무 차이가 없어도 15번 검정하면 최소 하나가 유의할 확률 = {1 - 0.95 ** 15:.2f})")
    sig_pairs = [f"{names[i]}>{names[j]}" for (i, j), s in zip(pairs, benjamini_hochberg(pp)) if s]
    print(f"    BH 통과: {sig_pairs if sig_pairs else '없음'}")
    print("  → 축 간 우열을 말하려면 이 15번을 보정해야 한다. 보정하면 대부분 사라진다.")
    print("     리포트에 '`subjects`가 `framing`보다 정확하다'를 쓰려면 근거가 부족하다.")

    # ── [C] 유의성과 효과 크기는 다른 질문이다 ──────────────────────────
    print("\n" + "=" * 86)
    print("[C] 유의하다 ≠ 크게 낫다 — 표본이 크면 아무 차이나 유의해진다")
    print("=" * 86)
    print(f"{'비교':<26}{'차이':>8}{'Cohen h':>10}{'n=50':>12}{'n=500':>12}{'n=5000':>12}")
    for p1, p2, label in ((0.62, 0.60, "0.62 vs 0.60"),
                          (0.65, 0.60, "0.65 vs 0.60"),
                          (0.75, 0.60, "0.75 vs 0.60")):
        row = f"{label:<26}{p1 - p2:>8.2f}{cohens_h(p1, p2):>10.2f}"
        for n in (50, 500, 5000):
            tbl = [[round(p1 * n), round((1 - p1) * n)], [round(p2 * n), round((1 - p2) * n)]]
            pv = stats.fisher_exact(tbl)[1]
            row += f"{('유의' if pv < 0.05 else '—') + f' {pv:.3f}':>12}"
        print(row)
    print("  → 2%p 차이도 n=5000이면 유의해진다. **유의성은 표본 크기의 함수이기도 하다.**")
    print("     그래서 보고에는 항상 세 가지를 함께 적는다 — 차이의 크기 · 구간 · p값.")
    print()
    print("  우리 맥락에서 '의미 있는 차이'를 미리 정해 두면:")
    print("    · 쌍 정확도  : 0.5 대비 +0.10 이상 (그 아래는 λ가 어차피 사진학을 택한다)")
    print("    · 태그 정확도 : 축별 90% — 이유 문장에 쓸 수 있는 최소선")
    print("    · 수락률     : 라운드 간 +0.10 이상 (그 아래는 풀 고갈로 설명된다, 01 step6)")
    print("  → 이 값을 **실험 전에** 적어 두는 것이 사전 등록이다. 데이터를 본 뒤에 정하면")
    print("     '유의한 쪽'을 기준으로 고르게 된다 — 자기도 모르게.")

    # ── [D] 리포트 규칙으로 정리 ────────────────────────────────────────
    print("\n" + "=" * 86)
    print("[D] 그래서 리포트는 이렇게 쓴다")
    print("=" * 86)
    print("  ① 주지표를 **하나** 정한다 (홀드아웃 쌍 정확도). 나머지는 전부 보조지표라고 적는다.")
    print("  ② 주지표는 보정 없이, 보조지표는 **BH 보정 후** 보고한다.")
    print("  ③ 모든 비율에 n과 95% CI를 붙인다 (02 step1: Wilson, 사람 단위면 부트스트랩).")
    print("  ④ 차이를 말할 때는 **효과 크기와 실용 기준**을 함께 적는다.")
    print("  ⑤ 하이퍼파라미터를 고른 데이터와 보고하는 데이터를 나눈다 (02 step4).")
    print()
    print("  나쁜 문장: \"scene 74%, framing 92% — framing이 훨씬 정확하다\"")
    print("  좋은 문장: \"scene 74% (n=50, 95% CI [0.60, 0.84]). 6개 축을 함께 검정했고")
    print("             BH 보정 후에도 90% 기준 미달이 유지되는 유일한 축이다.")
    print("             축 간 우열(15개 쌍 비교)은 이 표본에서 결론 내지 않는다.\"")


if __name__ == "__main__":
    main()
