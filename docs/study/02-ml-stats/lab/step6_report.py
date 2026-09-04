"""STEP 6 — 리포트 한 장: 배운 규칙을 전부 적용해 실험 결과를 보고한다.

────────────────────────────────────────────────────────────────────────────
01은 시스템의 세로 슬라이스였고(온보딩 → 초안 → 피드백), 02의 세로 슬라이스는
**실험의 세로 슬라이스**다 — 사전 등록 → 수집 → 분석 → 보고.

W1–2 쌍 비교 실험(사람 10명 × 120쌍, `plan.md` §6-A)을 합성 데이터로 미리 한 번
돌려 본다. 목적은 숫자를 얻는 것이 아니라 **리포트의 형식을 확정하는 것**이다.
실제 사람 데이터가 들어오면 이 스크립트의 분석부에 그대로 넣으면 된다.

적용하는 규칙 (전부 앞 스텝에서 나온 것)

    step1  비율에는 Wilson, 사람 단위는 부트스트랩
    step2  학습 곡선으로 '쌍을 더 vs 피처를 바꿔'를 진단
    step3  λ의 k를 데이터에서 추정
    step4  하이퍼파라미터를 고른 데이터와 보고할 데이터를 나눈다
    step5  보조지표는 BH 보정, 효과 크기와 실용 기준을 함께 적는다
    01 s5  표본 단위는 사람. 정확도만 쓰지 말고 n·CI·p값을 함께

실행: ../../.venv/bin/python step6_report.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

import pairsim as P              # ★ 먼저 import

import gallery as G              # noqa: E402
from metrics import wilson_interval   # noqa: E402
from step1_bt import fit_bt      # noqa: E402
from step3_lambda import LAMBDA_MAX, LAMBDA_MIN   # noqa: E402
from step3_shrinkage import center_by_axis, learnable_mask   # noqa: E402
from step5_multiplicity import benjamini_hochberg            # noqa: E402

# ── 사전 등록서 (데이터를 보기 전에 확정한다) ────────────────────────────────
PREREG = {
    "주지표": "홀드아웃 쌍 예측 정확도 (사람 단위 평균)",
    "귀무가설": "정확도 = 0.50 (우연)",
    "의미 있는 차이(MDE)": "+0.10 이상",
    "표본": "사람 10명 × 120쌍 (학습 60 / 홀드아웃 60)",
    "분할": "검증 6명(하이퍼파라미터 선택) / 시험 4명(주지표 보고)",
    "구간": "사람 단위 부트스트랩 95%",
    "보조지표": "축별 정확도 5개 — BH 보정 후 보고, 단독 결론 금지",
    "λ 곡선": "실험 전에 k를 별도 추정해 확정 (튜닝 대상 아님)",
}

N_PEOPLE, N_TRAIN, N_HOLD = 10, 60, 60
N_VAL, N_TEST = 6, 4


def bootstrap_ci(values: np.ndarray, rng, reps: int = 5000) -> tuple[float, float]:
    idx = rng.integers(0, len(values), (reps, len(values)))
    boot = values[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def estimate_k(g, train_pool, n_people: int = 30) -> float:
    """step3 [D]의 절차 — 신뢰도 곡선에 n/(n+k)를 맞춘다."""
    mask = learnable_mask()
    ns = (6, 12, 24, 48, 96)
    rel = {}
    for n_train in ns:
        rs = []
        for ps in range(n_people):
            w_true = center_by_axis(G.make_person(seed=5000 + ps, strength=P.STRENGTH))
            tr = G.sample_pairs(train_pool, n_train, seed=6000 + ps)
            c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=7000 + ps)
            w_hat = center_by_axis(fit_bt(g.X[c] - g.X[r], reg=1.0))
            a_, b_ = w_hat[mask], w_true[mask]
            if a_.std() < 1e-12 or b_.std() < 1e-12:
                rs.append(0.0); continue
            cc = np.corrcoef(a_, b_)[0, 1]
            rs.append(float(cc ** 2) if np.isfinite(cc) else 0.0)
        rel[n_train] = float(np.mean(rs))
    obj = lambda k: sum((rel[n] - n / (n + k)) ** 2 for n in ns)
    return float(minimize_scalar(obj, bounds=(0.5, 500), method="bounded").x)


def axis_accuracy(g, w_hat, ct, rt) -> dict[str, tuple[int, int]]:
    """축별로 (맞힌 수, 그 축이 갈린 홀드아웃 쌍 수)."""
    out = {}
    for ax in G.AXES:
        hit = tot = 0
        for a, b in zip(ct, rt):
            if g.tags[a][ax] == g.tags[b][ax]:
                continue
            tot += 1
            hit += int((g.X[a] - g.X[b]) @ w_hat > 0)
        out[ax] = (hit, tot)
    return out


def main() -> None:
    rng = np.random.default_rng(0)
    g, train_pool, test_pool = P.world()

    # ── [A] 사전 등록 ───────────────────────────────────────────────────
    print("=" * 88)
    print("[A] 사전 등록서 — **데이터를 보기 전에** 확정한다")
    print("=" * 88)
    for k, v in PREREG.items():
        print(f"  {k:<20} {v}")
    print("  → 이걸 먼저 적어 두면 step4(승자의 저주)와 step5(다중 비교)의 절반이 사라진다.")
    print("     데이터를 본 뒤에 주지표를 고르는 것이 가장 흔한 자기기만이다.")

    # ── [B] λ 곡선을 실험 전에 확정 ────────────────────────────────────
    print("\n" + "=" * 88)
    print("[B] 실험 전 준비 — λ 곡선의 k를 따로 추정해 못 박는다 (step3 [D])")
    print("=" * 88)
    k_hat = estimate_k(g, train_pool)
    lam = lambda n: LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * n / (n + k_hat)
    print(f"  추정된 k = {k_hat:.0f}  →  λ(12)={lam(12):.2f}  λ(60)={lam(60):.2f}  λ(120)={lam(120):.2f}")
    print("  → **이 실험의 참가자와 겹치지 않는 데이터로** 추정했다(다른 시드 대역).")
    print("     실험 데이터로 k를 정하고 같은 데이터로 성능을 보고하면 step4의 그 편향이 생긴다.")

    # ── [C] 데이터 수집 + 3분할 ────────────────────────────────────────
    print("\n" + "=" * 88)
    print("[C] 수집 — 사람 10명, 사람 단위로 검증/시험 분할")
    print("=" * 88)
    people = list(range(N_PEOPLE))
    perm = rng.permutation(people)
    val_ids = sorted(int(x) for x in perm[:N_VAL])
    test_ids = sorted(int(x) for x in perm[N_VAL:])
    trials = {ps: P.run_person(ps, N_TRAIN, n_hold=N_HOLD) for ps in people}
    print(f"  검증 {N_VAL}명 {val_ids}   시험 {N_TEST}명 {test_ids}")
    print(f"  {'사람':<6}{'맞힘/홀드아웃':>14}{'정확도':>9}   분할")
    for ps in people:
        t = trials[ps]
        print(f"  {ps:<6}{f'{t.k}/{t.n}':>14}{t.acc:>9.3f}   {'검증' if ps in val_ids else '시험'}")

    # ── [D] 규칙을 안 지킨 리포트 ──────────────────────────────────────
    print("\n" + "=" * 88)
    print("[D] 리포트 두 장 — 같은 데이터, 다른 규칙")
    print("=" * 88)
    all_k = sum(t.k for t in trials.values())
    all_n = sum(t.n for t in trials.values())
    lo_w, hi_w = wilson_interval(all_k, all_n)
    p_all = stats.binomtest(all_k, all_n, 0.5, alternative="greater").pvalue
    # 축별 정확도를 전원에서 모아 가장 좋은 축을 고른다 (= 흔히 하는 일)
    agg: dict[str, list[int]] = {ax: [0, 0] for ax in G.AXES}
    for ps in people:
        w_true = G.make_person(seed=ps, strength=P.STRENGTH)
        te = G.sample_pairs(test_pool, N_HOLD, seed=200 + ps)
        ct, rt = G.answer_pairs(g, w_true, te, temp=P.TEMP, seed=400 + ps)
        for ax, (h, t) in axis_accuracy(g, trials[ps].w_hat, ct, rt).items():
            agg[ax][0] += h; agg[ax][1] += t
    best_ax = max(agg, key=lambda a: (agg[a][0] / agg[a][1]) if agg[a][1] else 0)

    print("  ┌─ 나쁜 리포트 ────────────────────────────────────────────────────┐")
    print(f"  │ 홀드아웃 쌍 정확도 {all_k / all_n:.0%} — 우연(50%)을 크게 웃돈다.")
    print(f"  │ 축별로 보면 `{best_ax}`가 {agg[best_ax][0] / agg[best_ax][1]:.0%}로 가장 잘 학습된다.")
    print(f"  │ λ는 만족도가 최대인 0.4가 최적이었다.")
    print("  └──────────────────────────────────────────────────────────────────┘")
    print("  틀린 곳: ① 사람을 쌍으로 풀링(구간이 좁아진다, step1 [D])")
    print("           ② 축 5개를 보고 최고를 골라 보고(다중 비교 + 승자의 저주, step4·5)")
    print("           ③ λ를 같은 데이터에서 고르고 같은 데이터로 자랑(step4)")
    print("           ④ n·구간·효과 크기가 없다(01 step5 [A])")

    # ── [E] 규칙을 지킨 리포트 ─────────────────────────────────────────
    test_acc = np.array([trials[ps].acc for ps in test_ids])
    val_acc = np.array([trials[ps].acc for ps in val_ids])
    blo, bhi = bootstrap_ci(test_acc, rng)
    k_t = sum(trials[ps].k for ps in test_ids)
    n_t = sum(trials[ps].n for ps in test_ids)
    p_t = stats.binomtest(k_t, n_t, 0.5, alternative="greater").pvalue

    ax_names = list(G.AXES)
    ax_p = np.array([
        stats.binomtest(agg[a][0], agg[a][1], 0.5, alternative="greater").pvalue if agg[a][1] else 1.0
        for a in ax_names
    ])
    ax_sig = benjamini_hochberg(ax_p)

    print()
    print("  ┌─ 정직한 리포트 ──────────────────────────────────────────────────┐")
    print(f"  │ [주지표] 홀드아웃 쌍 예측 정확도 (시험 {N_TEST}명, {n_t}쌍)")
    print(f"  │   {test_acc.mean():.3f}   95% CI(사람 단위 부트스트랩) [{blo:.3f}, {bhi:.3f}]   p={p_t:.4f}")
    print(f"  │   사람별: {np.round(test_acc, 3).tolist()}")
    mde_ok = "충족" if test_acc.mean() - 0.5 >= 0.10 else "**미달**"
    print(f"  │   사전 등록한 MDE(+0.10) 대비: {test_acc.mean() - 0.5:+.3f} → {mde_ok}")
    print(f"  │ [참고] 검증 {N_VAL}명 평균 {val_acc.mean():.3f} — 하이퍼파라미터 선택에만 사용")
    print(f"  │ [보조] 축별 정확도 (BH 보정, 단독 결론 금지)")
    for a, pv, s in zip(ax_names, ax_p, ax_sig):
        h, t = agg[a]
        if t == 0:
            print(f"  │   {a:<11} 갈린 쌍 0개 — 판정 불가 (쌍 규칙의 구조적 결과)")
            continue
        lo, hi = wilson_interval(h, t)
        print(f"  │   {a:<11} {h}/{t} = {h / t:.3f}  [{lo:.2f}, {hi:.2f}]  p={pv:.3f}  {'유의' if s else '—'}")
    print(f"  │ [사전 확정] λ 곡선 k={k_hat:.0f} — 이 실험 데이터로 고르지 않았다")
    print("  │ [한계] 축별 지표는 쌍 단위로 풀링했다 — step1 [D]가 보여 준 대로 구간이")
    print("  │        실제보다 좁다. 사람 수가 적어 축별 부트스트랩이 불가능하기 때문이고,")
    print("  │        그래서 '단독 결론 금지'라고 못 박아 둔 것이다.")
    print("  └──────────────────────────────────────────────────────────────────┘")

    print(f"\n  두 리포트의 같은 숫자: 전체 풀링 정확도 {all_k / all_n:.3f} "
          f"[{lo_w:.3f}, {hi_w:.3f}] p={p_all:.5f}")
    print(f"  두 리포트의 다른 결론: 시험셋 {test_acc.mean():.3f} [{blo:.3f}, {bhi:.3f}]")
    print("  → 점추정은 비슷해도 **구간이 훨씬 넓고, 말할 수 있는 것이 줄어든다.**")
    print("     그게 정직해진 대가이자, 멘토 앞에서 방어할 수 있게 되는 값이다.")

    # ── [F] 진단 — 다음에 무엇을 할 것인가 ─────────────────────────────
    print("\n" + "=" * 88)
    print("[F] 리포트의 마지막 줄은 항상 '그래서 다음에 무엇을 하는가'다 (step2 [C]의 진단표)")
    print("=" * 88)
    tr_acc = []
    for ps in people:
        w_true = G.make_person(seed=ps, strength=P.STRENGTH)
        tr = G.sample_pairs(train_pool, N_TRAIN, seed=100 + ps)
        c, r = G.answer_pairs(g, w_true, tr, temp=P.TEMP, seed=300 + ps)
        d = g.X[c] - g.X[r]
        tr_acc.append(float(((d @ trials[ps].w_hat) > 0).mean()))
    tr_m = float(np.mean(tr_acc))
    te_m = float(np.mean([t.acc for t in trials.values()]))
    gap = tr_m - te_m
    print(f"  학습 정확도 {tr_m:.3f} · 홀드아웃 {te_m:.3f} · 격차 {gap:.3f}")
    if gap > 0.15:
        print("  → **고분산.** 처방은 쌍을 더 받는 것 — 사람을 늘리기 전에 사람당 쌍을 늘린다.")
    elif te_m < 0.58:
        print("  → **고편향.** 처방은 표현을 바꾸는 것 — 태그 어휘·축 구성을 재검토한다.")
    else:
        print("  → **균형.** 이 표현으로 갈 만큼 갔다. 다음 이득은 태그 품질(VLM 프롬프트)에서 온다.")
    print("  ※ 이 진단 한 줄이 로드맵의 다음 이슈를 정한다. 정확도 숫자 자체보다 중요하다.")


if __name__ == "__main__":
    main()
