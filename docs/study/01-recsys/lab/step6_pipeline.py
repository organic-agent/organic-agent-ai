"""STEP 6 — 세로 슬라이스: 온보딩 → 초안 → 피드백 → 재추천 한 바퀴.

────────────────────────────────────────────────────────────────────────────
step1~5에서 따로 만든 조각을 설계안 §3-B 순서대로 이어 붙인다. 여기서 확인할 것은
"정확도가 몇이냐"가 아니라 **루프가 실제로 돌아가는가** — 라운드가 갈수록 수락률이
오르는가, 오른다면 무엇 때문에 오르는가다.

파이프라인
    ① 온보딩 12쌍          → BT로 취향 벡터 추정 (step1·2)
    ② λ(evidence)·conf(axis) → prior와 취향을 섞어 점수 (step3)
    ③ 커버리지 + 클러스터 + MMR → 제시할 30장 (step4)
    ④ 고객 반응(수락/거부) → 같은 라운드 안에서 쌍으로 바꿔 evidence에 추가
    ⑤ 자연어 피드백 → {axis, tag, delta}로 번역해 선호에 반영 (LLM 자리)
    ⑥ 다시 ②로. 이미 담은 사진은 제외.

시뮬레이션이므로 '고객'도 코드다. 고객은 만족도가 높을수록 담을 확률이 높고,
같은 축이 과하게 나오면 말로 불평한다. 진짜 사람이 아니라는 것을 잊지 말 것 —
이 스크립트가 증명하는 것은 **배관이 이어졌다**는 사실이지 서비스가 좋다는 게 아니다.

실행: ../../.venv/bin/python step6_pipeline.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

import gallery as G
from metrics import binom_p_value, ndcg, pairwise_accuracy
from step1_bt import fit_bt
from step3_lambda import axis_confidence, lambda_of, pref_scores, satisfaction, z
from step4_mmr import ils, select_with_coverage

STRENGTH, TEMP = 1.5, 0.7
ROUND_K = 30          # 라운드마다 제시할 장수
ACCEPT_TEMP = 0.6     # 고객이 담을지 말지의 흔들림
N_ROUNDS = 4


def customer_accepts(sat: np.ndarray, idx: list[int], threshold: float, rng) -> np.ndarray:
    """고객 반응 시뮬레이션. 만족도가 문턱을 넘을수록 담을 확률이 높다."""
    p = 1.0 / (1.0 + np.exp(-(sat[idx] - threshold) / ACCEPT_TEMP))
    return rng.random(len(idx)) < p


def feedback_to_delta(g: G.Gallery, shown: list[int], accepted: np.ndarray) -> tuple[str, str, float]:
    """자연어 피드백 → {axis, tag, delta} 번역을 **규칙으로** 흉내 낸 것.

    실제로는 고객이 "클로즈업이 너무 많아요"라고 쓰고 Bedrock Haiku가 이 구조체를
    돌려준다(설계안 §3-C). 여기서는 '많이 보여줬는데 안 담긴 태그'를 찾아 음수 delta를 준다.
    """
    worst, worst_gap, worst_axis = None, 0.0, None
    for ax in G.AXES:
        for tag in G.AXES[ax]:
            shown_n = sum(1 for i in shown if g.tags[i][ax] == tag)
            if shown_n < 4:                      # 몇 장 안 보여준 태그로는 불평하지 않는다
                continue
            acc_n = sum(1 for i, a in zip(shown, accepted) if a and g.tags[i][ax] == tag)
            gap = shown_n / len(shown) - acc_n / max(int(accepted.sum()), 1)
            if gap > worst_gap:
                worst, worst_gap, worst_axis = tag, gap, ax
    if worst is None:
        return ("", "", 0.0)
    return (worst_axis, worst, -0.3 * min(worst_gap * 3, 1.0))


def run_loop(
    g: G.Gallery,
    prior: np.ndarray,
    train_pool: list[tuple[int, int]],
    person_seed: int,
    rng_seed: int,
    n_rounds: int = N_ROUNDS,
    verbose: bool = False,
    holdout: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[dict]:
    """온보딩 → 초안 → 반응 → 재추천을 n_rounds번 돈다. 라운드별 관측치를 돌려준다.

    holdout을 주면 라운드마다 **학습에 안 쓴 쌍**으로 취향 추정을 재 본다(step5의 주지표).
    수락률과 달리 이 값은 사진 풀이 줄어드는 것과 무관하다 — 실험 [B]의 핵심.
    """
    rng = np.random.default_rng(rng_seed)
    w_true = G.make_person(seed=person_seed, strength=STRENGTH)
    sat = satisfaction(g, w_true)
    threshold = float(np.quantile(sat, 0.75))     # 상위 25%쯤이면 담고 싶어지는 고객

    # ① 온보딩 12쌍
    onboarding = G.sample_pairs(train_pool, 12, seed=1)
    chosen, rejected = G.answer_pairs(g, w_true, onboarding, temp=TEMP, seed=2)
    chosen, rejected = list(chosen), list(rejected)
    nl_delta = np.zeros(G.D_TAG)                  # 자연어 피드백 누적분

    rows: list[dict] = []
    taken: set[int] = set()
    for rnd in range(1, n_rounds + 1):
        # ② 점수
        d = np.array([g.X[c] - g.X[r] for c, r in zip(chosen, rejected)])
        w_hat = fit_bt(d, reg=1.0)
        conf = axis_confidence(g, w_hat, np.array(chosen), np.array(rejected))
        lam = lambda_of(len(chosen))
        pref = z(pref_scores(g, w_hat, conf) + g.X @ nl_delta)
        score = (1 - lam) * prior + lam * pref
        score[list(taken)] = -np.inf              # 이미 담은 것은 다시 제시하지 않는다

        # ③ 커버리지 + 클러스터 + MMR
        shown = select_with_coverage(g, score, ROUND_K, lam_mmr=0.7)

        # ④ 고객 반응
        accepted = customer_accepts(sat, shown, threshold, rng)
        order = sorted(range(len(shown)), key=lambda i: -score[shown[i]])
        gains = np.array([1.0 if accepted[i] else 0.0 for i in order])
        n_scene = len({g.scene(i) for i in shown})

        # ⑤ 자연어 피드백 (한 라운드에 한 번, 있을 때만)
        ax, tag, delta = feedback_to_delta(g, shown, accepted)
        fb = f'"{tag} 그만" → {{{ax}, {tag}, {delta:+.2f}}}' if ax else "없음"

        if verbose:
            print(f"{rnd:<7}{len(chosen):>9}{lam:>7.2f}{len(shown):>6}{int(accepted.sum()):>6}"
                  f"{accepted.mean():>8.2f}{ndcg(gains):>8.3f}{ils(g.emb, shown):>7.3f}{n_scene:>6}  {fb}")

        rest = np.setdiff1d(np.arange(len(g)), np.fromiter(taken, dtype=int, count=len(taken)))
        rows.append({
            "evidence": len(chosen), "lam": lam, "rate": float(accepted.mean()),
            "ndcg": ndcg(gains), "lift": float(sat[shown].mean() - sat[rest].mean()),
            "pool": float(sat[rest].mean()),
            "holdout": pairwise_accuracy(w_hat, g.X, *holdout) if holdout else float("nan"),
        })

        # ⑥ 반응을 evidence로: 같은 라운드 안에서 수락 × 거부 쌍을 만든다
        acc_idx = [s for s, a in zip(shown, accepted) if a]
        rej_idx = [s for s, a in zip(shown, accepted) if not a]
        for a_i, r_i in zip(acc_idx, rng.permutation(rej_idx)[: len(acc_idx)] if rej_idx else []):
            chosen.append(int(a_i)); rejected.append(int(r_i))
        taken.update(acc_idx)
        if ax:
            nl_delta[G.FEATURE_INDEX[f"{ax}={tag}"]] += delta
    return rows


def main() -> None:
    g = G.make_gallery(1000, seed=0)
    prior = z((g.tech + g.aes) / 2)
    train_pool, test_pool = G.split_pool(G.design_pair_pool(g), 0.5, seed=7)

    print(f"갤러리 {len(g)}장 · 온보딩 쌍 12개로 시작\n")
    print(f"{'라운드':<7}{'evidence':>9}{'λ':>7}{'제시':>6}{'수락':>6}{'수락률':>8}"
          f"{'nDCG':>8}{'ILS':>7}{'장면':>6}  자연어 피드백")
    print("-" * 96)
    run_loop(g, prior, train_pool, person_seed=3, rng_seed=0, verbose=True)

    print("-" * 96)
    print("\n읽는 법")
    print(" · evidence가 늘면서 λ가 오른다 = 라운드가 갈수록 취향에 더 걸고 사진학에 덜 건다.")
    print(" · 수락률이 오르면 피드백 루프가 작동하는 것이다. 단 라운드가 갈수록 '남은 사진'의")
    print("   질이 떨어지므로(좋은 것부터 담아 갔으니) 수락률 상승은 그 하락과 싸운 결과다.")
    print(" · nDCG는 '담은 것이 위쪽에 있었나'. 수락률이 같아도 nDCG가 낮으면 순서가 틀린 것.")
    print(" · 자연어 피드백은 선택을 바꾸지 않는다 — 태그 가중치만 움직인다(설계안 §3-C 가드레일).")
    print("\n주의 — 이 숫자로 주장할 수 있는 것과 없는 것")
    print(" · 할 수 있다: 배관이 이어졌다. 라운드마다 evidence·λ·점수·재랭킹이 실제로 갱신된다.")
    print(" · 할 수 없다: 서비스가 설득력 있다. 고객이 코드이므로 수락률은 내가 만든 가정의")
    print("   되풀이일 뿐이다. 진짜 수치는 사람에게 받은 쌍(step5의 설계)에서만 나온다.")

    experiment_b(g, prior, train_pool, test_pool)


def experiment_b(g, prior, train_pool, test_pool, n_people: int = 20) -> None:
    """[B] 위 표를 한 사람으로 읽으면 안 되는 이유 — step5의 도구를 여기 적용한다.

    위 [A]는 고객 한 명·시드 하나다. step5에서 배운 대로 **재현되는지** 먼저 본다.
    그리고 수락률이 라운드 품질의 좋은 지표인지도 함께 검증한다.
    """
    print("\n" + "=" * 96)
    print(f"[B] 고객 {n_people}명으로 다시 — 라운드가 갈수록 정말 좋아지는가")
    print("=" * 96)

    te = G.sample_pairs(test_pool, 200, seed=999)
    rows_by_round: list[list[dict]] = [[] for _ in range(N_ROUNDS)]
    for ps in range(n_people):
        w_true = G.make_person(seed=ps, strength=STRENGTH)
        ct, rt = G.answer_pairs(g, w_true, te, temp=TEMP, seed=998)
        rows = run_loop(g, prior, train_pool, person_seed=ps, rng_seed=ps * 7 + 1,
                        holdout=(ct, rt))
        for i, row in enumerate(rows):
            rows_by_round[i].append(row)

    def col(i: int, key: str) -> float:
        return float(np.mean([r[key] for r in rows_by_round[i]]))

    print(f"{'라운드':<7}{'evidence':>9}{'λ':>7}{'수락률':>9}{'홀드아웃 정확도':>16}"
          f"{'남은 풀 평균':>13}{'lift':>8}")
    for i in range(N_ROUNDS):
        print(f"{i + 1:<7}{col(i, 'evidence'):>9.0f}{col(i, 'lam'):>7.2f}{col(i, 'rate'):>9.3f}"
              f"{col(i, 'holdout'):>16.3f}{col(i, 'pool'):>13.3f}{col(i, 'lift'):>8.3f}")

    print(f"\n  R1 → R4에서 개선된 고객 수 (n={n_people}, 이항검정 p값)")
    for key, label in (("rate", "수락률"), ("holdout", "홀드아웃 정확도"), ("lift", "lift")):
        up = sum(1 for j in range(n_people)
                 if rows_by_round[-1][j][key] > rows_by_round[0][j][key])
        print(f"    {label:<16} {up:>2}/{n_people}   p={binom_p_value(up, n_people):.4f}")

    print("\n  → **[A]의 '수락률 0.47 → 0.83'은 시드 하나의 우연이었다.** 20명 평균으로는")
    print("     거의 평평하고 이항검정도 통과하지 못한다. step5를 배우지 않았다면 이 표를")
    print("     '피드백 루프가 작동한다'는 근거로 리포트에 썼을 것이다.")
    print("  → 그런데 루프는 실제로 작동하고 있다 — **홀드아웃 쌍 정확도**가 그것을 보여 준다.")
    print("     수락률이 평평한 이유는 두 힘이 상쇄되기 때문이다:")
    print("       (+) evidence가 쌓여 취향 추정이 좋아진다        → 홀드아웃 정확도 상승")
    print("       (−) 좋은 사진부터 담아 가서 남은 풀이 나빠진다   → '남은 풀 평균'이 계속 하락")
    print("     수락률은 이 둘의 합이라 **무엇이 일어났는지 말해 주지 못한다.**")
    print("  → 교훈: 라운드 품질은 **풀 고갈에 영향받지 않는 지표**로 재야 한다.")
    print("     실서비스에서는 lift(제시한 것 vs 남은 것의 만족도 차)나 라운드별 홀드아웃이")
    print("     그 역할이다. 수락률만 KPI로 걸면 좋아져도 나빠져도 이유를 모른다.")


if __name__ == "__main__":
    main()
