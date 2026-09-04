"""STEP 7 — 분류 지표와 임계값: 눈감김 검출(A-1)을 어디서 자를 것인가.

────────────────────────────────────────────────────────────────────────────
이론

step1~6은 소재가 전부 쌍 비교였다. 여기서부터는 **전수 분석(A)** 이고, 문제의 종류가
추천이 아니라 **분류**다. 지표도 다르다.

1) 정확도는 희귀 사건에서 쓸모없다. 눈감김이 5%면 **"전부 정상"이라고 답해도 95%**다.
   그런데 그 검출기는 아무 일도 안 한다. 정확도는 다수 클래스의 비율을 되풀이할 뿐이다.

2) 혼동행렬이 기본 단위다. 네 칸에서 모든 분류 지표가 나온다.

                       예측: 눈감김    예측: 정상
       정답: 눈감김        TP            FN      ← 놓친 것
       정답: 정상          FP            TN
                       ↑ 잘못 잡은 것

       정밀도 = TP/(TP+FP)   "눈감김이라고 한 것 중 진짜 비율"  — 잘못 잡는 비용
       재현율 = TP/(TP+FN)   "진짜 눈감김 중 잡아낸 비율"      — 놓치는 비용
       F1     = 둘의 조화평균

3) 임계값을 바꾸면 정밀도와 재현율이 교환된다. 그 곡선이 **PR 곡선**이다.
   **ROC 곡선**(TPR vs FPR)은 다른 것을 그린다 — 그리고 희귀 사건에서
   ROC는 **낙관적으로 보인다.** FPR의 분모가 거대한 다수 클래스라서
   오탐이 아무리 늘어도 FPR이 잘 안 오르기 때문이다.

4) 임계값은 통계가 아니라 **비용**이 정한다. 눈감김 한 장을 놓치는 손해와
   멀쩡한 사진 한 장을 눈감김으로 모는 손해가 같지 않다. 어느 쪽이 비싼지는
   **제품 결정**이지 데이터가 답해 주지 않는다.

5) 그리고 우리 설계는 애초에 **자르지 않는다** (CLAUDE.md: 품질 점수로 하드 필터링 금지).
   그럼 임계값은 어디에 쓰이나 — [D]에서 답한다.

실행: ../../.venv/bin/python step7_classification.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

import abatch as A
import pairsim as P

TAU = 0.5          # 흔히 쓰는 기본 임계값. 이 값이 왜 임의인지가 이 스텝의 주제다


def confusion(score: np.ndarray, closed: np.ndarray, tau: float) -> tuple[int, int, int, int]:
    """점수가 tau 미만이면 '눈감김'으로 판정. (TP, FP, FN, TN)"""
    pred_closed = score < tau
    tp = int((pred_closed & closed).sum())
    fp = int((pred_closed & ~closed).sum())
    fn = int((~pred_closed & closed).sum())
    tn = int((~pred_closed & ~closed).sum())
    return tp, fp, fn, tn


def pr_curve(score: np.ndarray, closed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """모든 임계값에서의 (재현율, 정밀도, 임계값). 점수가 낮을수록 '눈감김'이다."""
    order = np.argsort(score)                  # 낮은 점수부터 = 확신 있는 눈감김부터
    y = closed[order]
    tp = np.cumsum(y)
    fp = np.cumsum(~y)
    recall = tp / max(int(closed.sum()), 1)
    precision = tp / np.maximum(tp + fp, 1)
    return recall, precision, score[order]


def roc_curve(score: np.ndarray, closed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(score)
    y = closed[order]
    tpr = np.cumsum(y) / max(int(closed.sum()), 1)
    fpr = np.cumsum(~y) / max(int((~closed).sum()), 1)
    return fpr, tpr


def auc(x: np.ndarray, y: np.ndarray) -> float:
    return float(np.trapezoid(y, x)) if len(x) > 1 else float("nan")


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    """PR 곡선 아래 면적(AP). 재현율 증가분 × 그 지점의 정밀도."""
    dr = np.diff(np.concatenate([[0.0], recall]))
    return float((dr * precision).sum())


def main() -> None:
    g, _, _ = P.world()
    e = A.eye_signal(g)
    v = e.valid()                               # 얼굴이 있는 사진만
    score, closed, ratio = e.score[v], e.closed[v], e.face_ratio[v]

    # ── [A] 정확도의 함정 ───────────────────────────────────────────────
    print("=" * 86)
    print("[A] 눈감김은 희귀 사건이다 — 정확도가 쓸모없어지는 자리")
    print("=" * 86)
    prev = closed.mean()
    print(f"  얼굴이 있는 사진 {v.sum()}장 중 진짜 눈감김 {closed.sum()}장 ({prev:.1%})")
    print()
    print(f"{'검출기':<28}{'정확도':>9}{'정밀도':>9}{'재현율':>9}{'F1':>8}")
    tp, fp, fn, tn = confusion(score, closed, TAU)
    for name, (a, p, r, f) in {
        "① 전부 '정상'이라고 답": (1 - prev, float("nan"), 0.0, float("nan")),
        "② 전부 '눈감김'이라고 답": (prev, prev, 1.0, 2 * prev / (1 + prev)),
        f"③ eyes_open < {TAU}": (
            (tp + tn) / len(score), tp / max(tp + fp, 1), tp / max(tp + fn, 1),
            2 * tp / max(2 * tp + fp + fn, 1)),
    }.items():
        print(f"{name:<28}{a:>9.3f}{p:>9.3f}{r:>9.3f}{f:>8.3f}")
    print("  → ①은 **아무 일도 안 하는데 정확도 0.95**다. 정확도만 보면 쓸 만해 보인다.")
    print("     재현율이 0인 것을 보고서야 이 검출기가 무용지물임을 알 수 있다.")
    print("  → 희귀 사건에서는 **정확도를 아예 보고하지 않는 편이 낫다.** 오해만 만든다.")

    # ── [B] 혼동행렬 ───────────────────────────────────────────────────
    print("\n" + "=" * 86)
    print(f"[B] 혼동행렬 (임계값 {TAU}) — 네 칸이 모든 지표의 출처다")
    print("=" * 86)
    print(f"                    예측: 눈감김   예측: 정상")
    print(f"    정답: 눈감김        {tp:>6}       {fn:>6}   ← 놓친 것(FN)")
    print(f"    정답: 정상          {fp:>6}       {tn:>6}")
    print(f"                     ↑ 잘못 잡은 것(FP)")
    print()
    print(f"    정밀도 {tp}/{tp + fp} = {tp / max(tp + fp, 1):.3f}   "
          f"재현율 {tp}/{tp + fn} = {tp / max(tp + fn, 1):.3f}")
    print(f"  → 임계값 {TAU}는 **아무 근거 없는 값**이다. 0.5인 이유는 σ의 중앙값이라서일 뿐,")
    print("     우리 비용 구조와는 아무 관계가 없다. [D]에서 근거 있는 값을 고른다.")

    # ── [C] PR과 ROC — 희귀 사건에서 갈린다 ─────────────────────────────
    print("\n" + "=" * 86)
    print("[C] 임계값을 움직이면 — PR 곡선과 ROC 곡선이 다른 말을 한다")
    print("=" * 86)
    rec, prec, taus = pr_curve(score, closed)
    fpr, tpr = roc_curve(score, closed)
    print(f"  ROC-AUC {auc(fpr, tpr):.3f}      Average Precision(PR-AUC) {average_precision(rec, prec):.3f}")
    print(f"  (무작위 검출기의 기준선:  ROC-AUC 0.500 ·  AP = 유병률 {prev:.3f})")
    print()
    print(f"{'임계값':<10}{'예측 눈감김':>12}{'정밀도':>9}{'재현율':>9}{'FPR':>9}")
    for t in (0.10, 0.20, 0.30, 0.50, 0.70):
        a, b, c, d = confusion(score, closed, t)
        print(f"{t:<10}{a + b:>12}{a / max(a + b, 1):>9.3f}{a / max(a + c, 1):>9.3f}"
              f"{b / max(b + d, 1):>9.3f}")
    print("  → **ROC-AUC는 높고 AP는 낮다.** 같은 검출기인데 두 지표가 다른 인상을 준다.")
    print("     이유: FPR의 분모가 정상 사진 전체(거대한 다수 클래스)라 오탐이 늘어도")
    print("     FPR이 잘 안 오른다. 정밀도의 분모는 '눈감김이라고 한 것'이라 바로 무너진다.")
    print("  → **희귀 사건에서는 PR을 본다.** ROC는 낙관적으로 보이게 만든다.")
    print("     `plan.md`가 08-22에 골든셋 AUC를 폐기한 것과는 다른 이야기지만(그건 정답")
    print("     라벨이 없어서였다), '어떤 곡선을 볼 것인가'는 지금도 유효한 질문이다.")

    # ── [D] 비용으로 임계값을 정한다 — 그리고 우리는 자르지 않는다 ────────
    print("\n" + "=" * 86)
    print("[D] 임계값은 통계가 아니라 비용이 정한다")
    print("=" * 86)
    print("  가정: 눈감김을 놓쳐 초안에 넣으면 손해 C_FN, 멀쩡한 사진을 눈감김으로 몰면 C_FP")
    print(f"{'C_FN : C_FP':<14}{'최적 임계값':>12}{'정밀도':>9}{'재현율':>9}{'총비용':>10}")
    for cfn, cfp in ((1, 1), (3, 1), (10, 1), (1, 3)):
        best_t, best_cost = None, np.inf
        for t in np.linspace(0.02, 0.95, 94):
            a, b, c, _ = confusion(score, closed, t)
            cost = cfn * c + cfp * b
            if cost < best_cost:
                best_t, best_cost = t, cost
        a, b, c, _ = confusion(score, closed, best_t)
        print(f"{f'{cfn} : {cfp}':<14}{best_t:>12.2f}{a / max(a + b, 1):>9.3f}"
              f"{a / max(a + c, 1):>9.3f}{best_cost:>10.0f}")
    print("  → 비용비를 바꾸면 최적 임계값이 움직인다. **데이터가 정해 주지 않는다.**")
    print("     '눈감김을 놓치는 것이 3배 비싸다'는 제품 결정이고, 그것부터 해야 한다.")
    print()
    print("  ★ 그런데 우리 설계는 애초에 **자르지 않는다** (품질 점수로 하드 필터링 금지).")
    print("     그럼 eyes_open을 어디에 쓰나 — `plan.md` §3-A가 이미 답을 갖고 있다:")
    print("       ① 사진학 prior의 재료 (순위를 흔들 뿐 후보를 없애지 않는다)")
    print("       ② **클러스터 대표 선정** (A-7) — 연사 5장 중 눈 뜬 컷을 고른다")
    print("     ②가 이 스텝의 진짜 쓰임새다. 그리고 ②는 **절대 판정이 아니라 상대 비교**다 —")
    print("     같은 연사 안에서 누가 더 눈을 떴나. 임계값이 아예 필요 없다.")
    print("  → 교훈: **분류 문제로 보이는 것이 사실은 순위 문제일 때가 있다.**")
    print("     그러면 임계값·정밀도·재현율이 통째로 필요 없어진다. 문제를 어떻게 세울지가 먼저다.")

    # ── [E] 스파이크 리포트의 미해결 관찰을 재현한다 ─────────────────────
    print("\n" + "=" * 86)
    print("[E] 작은 얼굴에서 정말 신호가 못 미더운가 (2026-08-25 미해결 관찰)")
    print("=" * 86)
    print(f"{'얼굴 bbox 비율':<18}{'장수':>6}{'뜬눈 점수 중앙값':>16}{'뜬눈 최소':>10}{'ROC-AUC':>10}{'AP':>8}")
    bands = ((0.000, 0.05, "~0.05 (단체·풍경)"), (0.05, 0.12, "0.05~0.12 (전신)"),
             (0.12, 0.25, "0.12~0.25 (상반신)"), (0.25, 1.0, "0.25~ (클로즈업)"))
    for lo, hi, label in bands:
        m = (ratio >= lo) & (ratio < hi)
        if m.sum() < 20 or closed[m].sum() < 3:
            continue
        s, c = score[m], closed[m]
        r_, p_, _ = pr_curve(s, c)
        f_, t_ = roc_curve(s, c)
        open_scores = s[~c]
        print(f"{label:<18}{m.sum():>6}{np.median(open_scores):>16.2f}{open_scores.min():>10.2f}"
              f"{auc(f_, t_):>10.3f}{average_precision(r_, p_):>8.3f}")
    print("  → **뜬눈 점수의 중앙값은 얼굴 크기와 거의 무관하다.** 작은 얼굴이라고 점수가")
    print("     낮아지는 게 아니다. 달라지는 것은 **최소값과 분리도(AUC·AP)** 다.")
    print("  → 즉 스파이크 리포트의 '2인 컷에서 eyes_open 최소값이 0.2대'는")
    print("     **점수가 낮은 것이 아니라 분산이 큰 것**이다. 둘은 다른 진단이고 처방도 다르다:")
    print("       · 점수가 낮다   → 임계값을 얼굴 크기별로 다르게 (싸다)")
    print("       · 분산이 크다   → 얼굴 크롭 후 재검출 (원가 상승, 리포트가 우려한 그것)")
    print("  → **최소값 하나로는 둘을 못 가른다. 라벨을 붙여 AUC/AP를 재야 갈린다.**")
    print("     스파이크 리포트의 '확인 필요' 항목에 필요한 것이 바로 그 작업이다 —")
    print("     작은 얼굴 사진 50장에 눈감김 라벨을 달고 이 표를 실제로 뽑는 것.")


if __name__ == "__main__":
    main()
