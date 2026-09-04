"""STEP 3 — 신뢰도 λ와 conf(axis): "증거가 적으면 취향을 덜 믿는다"를 수식으로.

────────────────────────────────────────────────────────────────────────────
이론

설계안 §3-B의 점수식:
    score(p) = (1−λ)·prior(p) + λ·pref(p) + coverage − mmr
    λ = f(evidence 수)      # 0개 → ~0.2, 12쌍+별점 20개 → ~0.7, 상한 있음
    pref(p) = Σ_axis conf(axis)·P(tag_axis(p) | 선호분포) + β·max_sim(p, 선호벡터)

1) 왜 섞는가. 취향 추정 pref는 증거가 적으면 거의 잡음이다(step2에서 확인). 잡음에
   전부를 걸면 "12쌍 보고 클로즈업만 100장" 같은 사고가 난다. 사진학 prior는 개인화는
   안 되지만 절대 크게 틀리지 않는다. λ는 **둘 사이의 베팅 비율**이다.

2) λ = f(n)의 형태. 베이지안에서 사후평균은
       posterior_mean = (n/(n+k))·데이터평균 + (k/(n+k))·사전평균
   으로, 데이터가 늘수록 데이터 쪽 가중이 커진다. k는 "사전분포가 데이터 몇 개만큼의
   힘을 갖는가"(가상 표본 크기). λ도 같은 꼴로 쓴다:
       λ(n) = λ_min + (λ_max − λ_min)·n/(n+k)
   설계안의 "0개→0.2, 12쌍+별점20개(=32)→0.7"에 맞추면 λ_min=0.2, λ_max=0.85, k≈10.

3) conf(axis) — 축별 신뢰도. 전체 증거량이 같아도 축마다 사정이 다르다. framing은
   12쌍 중 6쌍이 걸려 일관되게 갈렸는데 lighting은 1쌍뿐일 수 있다. 축별로
   베타-이항 사후평균으로 눌러 준다:
       p̂_axis = (승리 수 + a) / (해당 축 쌍 수 + 2a),   conf = max(0, 2·p̂ − 1)
   쌍이 0개면 conf=0 → 그 축은 점수에 기여하지 않는다(= 사진학이 대신 결정).
   a는 사전 강도. 이것이 설계안의 "선호가 50:50에 가까우면 conf=0".

4) 스케일을 맞추지 않으면 λ는 의미가 없다. prior와 pref의 단위가 다르면
   λ=0.5가 반반이 아니다. 갤러리 안에서 각각 z-표준화한 뒤 섞는다.

5) 증거는 '많고 적음'만 있는 게 아니라 **정확하고 부정확함**도 있다. λ = f(n)은 쌍의
   **개수**만 본다. 그런데 우리가 배우는 피처는 VLM이 적어 준 태그이고, 2026-08-25
   실측(`spike-report.md`)은 그 태그가 축마다 74~98%라고 말한다. 개수가 같아도 태그가
   틀리면 같은 λ를 쓸 수 없다 — 실험 [E]에서 확인한다.

실행: ../../.venv/bin/python step3_lambda.py   (실측 약 1초)
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

import gallery as G
from step1_bt import fit_bt

STRENGTH, TEMP, N_PERSONS = 1.5, 0.7, 12
TOP_K = 30                  # 초안으로 제시할 장수
QUALITY_WEIGHT = 0.5        # '고객 만족'에서 사진학 품질이 차지하는 몫 (평가용 정답 정의)

LAMBDA_MIN, LAMBDA_MAX, LAMBDA_K = 0.2, 0.85, 10.0


def lambda_of(n_evidence: float) -> float:
    """설계안 λ 곡선. n은 증거 단위 수(쌍 1개 = 1, 별점 1개 = 1)."""
    return LAMBDA_MIN + (LAMBDA_MAX - LAMBDA_MIN) * n_evidence / (n_evidence + LAMBDA_K)


def z(v: np.ndarray) -> np.ndarray:
    s = v.std()
    return (v - v.mean()) / (s if s > 1e-12 else 1.0)


def axis_confidence(
    g: G.Gallery,
    w_hat: np.ndarray,
    chosen: np.ndarray,
    rejected: np.ndarray,
    a: float = 1.0,
    reliability: dict[str, float] | None = None,
) -> dict[str, float]:
    """축별 conf. 그 축이 갈린 쌍에서 추정 선호 방향이 실제 선택과 얼마나 맞았나.

    a는 베타 사전 강도. 쌍이 적을수록 conf가 0.5(=무의견) 쪽으로 눌린다.

    reliability는 축별 태그 신뢰도(예: `G.AXIS_RELIABILITY_2026_08_25`)를 **곱해** 주는
    선택지다. "이 사람이 이 축에 의견이 있는가"(데이터)와 "이 축의 태그를 믿을 수
    있는가"(측정)는 다른 질문이니 곱하는 게 자연스러워 보인다. 실험 [E]에서
    **실제로는 거의 이득이 없다**는 것을 본다 — 이유도 거기 있다.
    """
    conf: dict[str, float] = {}
    for ax, vals in G.AXES.items():
        cols = [G.FEATURE_INDEX[f"{ax}={v}"] for v in vals]
        w_ax = w_hat[cols] - w_hat[cols].mean()
        n_ax = wins = 0
        for c, r in zip(chosen, rejected):
            tc, tr_ = g.tags[c][ax], g.tags[r][ax]
            if tc == tr_:
                continue
            n_ax += 1
            if w_ax[vals.index(tc)] > w_ax[vals.index(tr_)]:
                wins += 1
        p_hat = (wins + a) / (n_ax + 2 * a) if n_ax else 0.5
        conf[ax] = max(0.0, 2 * p_hat - 1)
        if reliability is not None:
            conf[ax] *= reliability.get(ax, 1.0)
    return conf


def pref_scores(g: G.Gallery, w_hat: np.ndarray, conf: dict[str, float] | None) -> np.ndarray:
    """취향 점수. conf가 주어지면 축별로 가중한다 (설계안의 Σ_axis conf(axis)·…)."""
    if conf is None:
        return g.X @ w_hat
    w = w_hat.copy()
    for ax, vals in G.AXES.items():
        cols = [G.FEATURE_INDEX[f"{ax}={v}"] for v in vals]
        w[cols] = (w[cols] - w[cols].mean()) * conf[ax]
    return g.X @ w


def satisfaction(g: G.Gallery, w_true: np.ndarray) -> np.ndarray:
    """평가용 '정답 만족도'. 취향 + 사진학 품질. 둘 다 실제로 고객 만족에 기여한다."""
    return z(G.utility(g, w_true)) + QUALITY_WEIGHT * z((g.tech + g.aes) / 2)


def main() -> None:
    g = G.make_gallery(1000, seed=0)
    persons = [G.make_person(seed=s, strength=STRENGTH) for s in range(N_PERSONS)]
    pool = G.design_pair_pool(g)
    train_pool, _ = G.split_pool(pool, test_frac=0.5, seed=7)
    prior = z((g.tech + g.aes) / 2)

    print("=" * 78)
    print("[A] λ 곡선 — λ(n) = 0.2 + 0.65·n/(n+10)")
    print("=" * 78)
    for n in (0, 4, 8, 12, 20, 32, 60, 200):
        lam = lambda_of(n)
        print(f"  증거 {n:>3}개 → λ={lam:.2f}  " + "█" * int(round(lam * 40)))
    print("  → 처음에는 사진학이 80%를 쥔다. 증거가 쌓여도 λ는 0.85에서 멈춘다(상한).")
    print("     상한이 없으면 사진학이 0이 되어 '취향에 맞지만 흔들린 사진'을 추천하게 된다.")

    # ── [B] λ 선택이 실제 추천 품질에 미치는 영향 ────────────────────────
    print("\n" + "=" * 78)
    print(f"[B] 증거량 × λ — 상위 {TOP_K}장의 평균 만족도 (고객 {N_PERSONS}명 평균)")
    print("     만족도 = z(취향효용) + 0.5·z(사진학). 클수록 좋다.")
    print("=" * 78)
    lam_grid = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    header = f"{'증거(쌍)':<10}" + "".join(f"{f'λ={l:g}':>9}" for l in lam_grid)
    print(header + f"{'λ=f(n)':>11}{'  ← 설계 곡선':<0}")
    for n_pairs in (0, 4, 12, 30, 100, 300):
        rows = {l: [] for l in lam_grid}
        rows["curve"] = []
        for pi, w_true in enumerate(persons):
            sat = satisfaction(g, w_true)
            if n_pairs == 0:
                pref = np.zeros(len(g))
                conf = None
            else:
                tr = G.sample_pairs(train_pool, n_pairs, seed=1000 + pi)
                c, r = G.answer_pairs(g, w_true, tr, temp=TEMP, seed=2000 + pi)
                w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
                conf = axis_confidence(g, w_hat, c, r)
                pref = z(pref_scores(g, w_hat, conf))
            for l in lam_grid:
                score = (1 - l) * prior + l * pref
                rows[l].append(sat[np.argsort(-score)[:TOP_K]].mean())
            l_c = lambda_of(n_pairs)
            score = (1 - l_c) * prior + l_c * pref
            rows["curve"].append(sat[np.argsort(-score)[:TOP_K]].mean())
        line = f"{n_pairs:<10}" + "".join(f"{np.mean(rows[l]):>9.3f}" for l in lam_grid)
        print(line + f"{np.mean(rows['curve']):>11.3f}")
    print("  → 증거 0~4개 구간에서 λ=1(취향만)은 λ=0(사진학만)보다 나쁘다. 잡음에 전부를 건 결과.")
    print("     증거가 쌓이면 역전된다. 고정 λ 하나로는 두 구간을 다 만족시킬 수 없다 —")
    print("     그래서 λ를 증거량의 함수로 둔다. 설계 곡선이 각 행의 최댓값 근처면 곡선이 맞는 것.")

    # ── [C] conf(axis)의 효과 ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[C] conf(axis) 가중을 껐을 때 vs 켰을 때 (λ는 설계 곡선)")
    print("=" * 78)
    print(f"{'증거(쌍)':<10}{'conf 없음':>12}{'conf 적용':>12}   |  축별 conf 평균")
    for n_pairs in (4, 12, 30, 100):
        off, on = [], []
        conf_by_axis: dict[str, list[float]] = {ax: [] for ax in G.AXES}
        for pi, w_true in enumerate(persons):
            sat = satisfaction(g, w_true)
            tr = G.sample_pairs(train_pool, n_pairs, seed=1000 + pi)
            c, r = G.answer_pairs(g, w_true, tr, temp=TEMP, seed=2000 + pi)
            w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
            conf = axis_confidence(g, w_hat, c, r)
            for ax in G.AXES:
                conf_by_axis[ax].append(conf[ax])
            l = lambda_of(n_pairs)
            for bucket, cf in ((off, None), (on, conf)):
                score = (1 - l) * prior + l * z(pref_scores(g, w_hat, cf))
                bucket.append(sat[np.argsort(-score)[:TOP_K]].mean())
        axes_str = " ".join(f"{ax[:4]} {np.mean(v):.2f}" for ax, v in conf_by_axis.items())
        print(f"{n_pairs:<10}{np.mean(off):>12.3f}{np.mean(on):>12.3f}   |  {axes_str}")
    print("  → 축별 conf를 보라. **scene은 언제나 0이다** — 쌍 규칙상 갈리는 쌍이 0개다(step2 [D]).")
    print("     설계안이 말한 '증거 없는 축은 점수에 기여하지 않는다'가 여기서 문자 그대로 일어난다.")
    print("  → 이 실행에서는 conf가 전 구간에서 조금씩 이득이고, 증거가 쌓일수록 폭이 커진다.")
    print("     이득 폭이 작은 데는 이유가 있다: conf를 곱한 뒤 pref를 z-표준화하므로")
    print("     '전체를 줄이는' 효과는 상쇄되고 **축 간 상대 가중**만 남는다.")
    print("  → 역할 분담이 이렇게 갈린다 — 전역 축소는 λ, 축 사이의 차등은 conf.")
    print("     둘을 겹쳐 두면 이중으로 눌려 취향이 사라진다. 설계안이 λ와 conf를 분리한 이유다.")
    print("  ※ 이 표의 차이는 작다. 고객 수(N_PERSONS)와 seed를 바꿔 재실행해서 방향이 유지되는지")
    print("     확인할 것. 한 번 돌린 숫자로 설계를 확정하지 않는 습관이 step5의 내용이다.")

    # ── [D] 실패 모드: λ를 1로 두면 무엇이 무너지는가 ────────────────────
    print("\n" + "=" * 78)
    print(f"[D] 실패 모드 — 12쌍만 받고 λ=1로 상위 {TOP_K}장을 뽑으면 (고객 #0)")
    print("=" * 78)
    w_true = persons[0]
    tr = G.sample_pairs(train_pool, 12, seed=1000)
    c, r = G.answer_pairs(g, w_true, tr, temp=TEMP, seed=2000)
    w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
    pref = z(pref_scores(g, w_hat, axis_confidence(g, w_hat, c, r)))
    for label, l in (("λ=1 (취향만)", 1.0), (f"λ={lambda_of(12):.2f} (설계 곡선)", lambda_of(12)), ("λ=0 (사진학만)", 0.0)):
        top = np.argsort(-((1 - l) * prior + l * pref))[:TOP_K]
        framing = {}
        for i in top:
            framing[g.tags[i]["framing"]] = framing.get(g.tags[i]["framing"], 0) + 1
        n_scene = len({g.scene(i) for i in top})
        worst_q = (g.tech[top] + g.aes[top]).min() / 2
        dist = " ".join(f"{k}:{v}" for k, v in sorted(framing.items(), key=lambda kv: -kv[1]))
        print(f"  {label:<22} framing 분포 [{dist}]")
        print(f"  {'':<22} 장면 종류 {n_scene}개 · 최저 품질 백분위 {worst_q:.0f}")
    print("  → λ=1은 한 축으로 쏠리고 저품질 컷이 섞인다. 12쌍짜리 추정을 100% 믿은 대가.")
    print("     장면 종류 수는 λ로 해결되지 않는다 — 커버리지와 MMR의 일이다(step4).")

    experiment_e(g, persons, prior)


def best_lambda_row(
    g: G.Gallery, persons: list[np.ndarray], prior: np.ndarray,
    train_pool: list[tuple[int, int]], n_pairs: int,
    lam_grid: tuple[float, ...], reliability: dict[str, float] | None = None,
) -> tuple[list[float], float]:
    """λ 격자별 평균 만족도와, 설계 곡선 λ(n)에서의 만족도."""
    rows = {l: [] for l in lam_grid}
    curve: list[float] = []
    for pi, w_true in enumerate(persons):
        sat = satisfaction(g, w_true)
        tr = G.sample_pairs(train_pool, n_pairs, seed=1000 + pi)
        c, r = G.answer_pairs(g, w_true, tr, temp=TEMP, seed=2000 + pi)
        w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
        conf = axis_confidence(g, w_hat, c, r, reliability=reliability)
        pref = z(pref_scores(g, w_hat, conf))
        for l in lam_grid:
            rows[l].append(sat[np.argsort(-((1 - l) * prior + l * pref))[:TOP_K]].mean())
        l_c = lambda_of(n_pairs)
        curve.append(sat[np.argsort(-((1 - l_c) * prior + l_c * pref))[:TOP_K]].mean())
    return [float(np.mean(rows[l])) for l in lam_grid], float(np.mean(curve))


def experiment_e(g_perfect: G.Gallery, persons: list[np.ndarray], prior: np.ndarray) -> None:
    """[E] 태그가 틀릴 때 — 2026-08-25 VLM 실측 오차를 주입한다.

    [A]~[D]는 전부 **"VLM 태그가 100% 맞다"** 는 가정 위에 있었다. 실측은 아니라고 한다.
    같은 갤러리·같은 사람·같은 시드에서 태그만 흐려 보면 λ가 어디로 움직여야 하는지 보인다.
    """
    print("\n" + "=" * 78)
    print("[E] 태그가 틀릴 때 — 2026-08-25 VLM 실측 오차 주입")
    print("    실측 축별 정확도: scene 74% · lighting 84% · expression 90% · framing 92% · subjects 98%")
    print("=" * 78)

    g = G.make_gallery(1000, seed=0, tag_error=1.0)
    err = G.tag_error_rate(g)
    print("  주입된 오답률: " + "  ".join(f"{ax} {err[ax]*100:.0f}%" for ax in G.AXES))
    print("  ※ scene 오답률이 실측 26%보다 낮은 것은 정상이다. 오차 모델은 **조건부** 확률")
    print("     P(오답|정답=snap)=0.38을 쓰는데, 이 갤러리는 snap이 27%뿐이기 때문이다.")
    print("     실측 표본은 snap이 68%였다 — `scene_profile=\"measured\"`로 돌리면 24%가 나온다.")

    # ① 쌍 규칙이 새는 것 — 설계 규칙은 관측 태그로 판정할 수밖에 없다
    pool_p = G.design_pair_pool(g_perfect)
    pool_e = G.design_pair_pool(g)
    ok = 0
    for i, j in pool_e:
        a, b = g.tags_true[i], g.tags_true[j]
        if a["scene"] == b["scene"] and sum(a[ax] != b[ax] for ax in G.AXES) == 1:
            ok += 1
    print(f"\n  ① 쌍 규칙 오염 — 설계 규칙(같은 scene·한 축만 다름)은 **관측 태그로만** 판정된다")
    print(f"     후보 풀 {len(pool_p)}쌍(정답 태그) → {len(pool_e)}쌍(관측 태그)")
    print(f"     그 {len(pool_e)}쌍 중 정답 태그로도 규칙을 만족하는 것: {ok}쌍 ({ok/len(pool_e)*100:.0f}%)")
    print(f"     나머지 {100-ok/len(pool_e)*100:.0f}%는 실제로는 장면이 다르거나 두 축 이상 다른 쌍이다.")
    print("     step2 §4가 '교란변수를 제거한다'고 한 그 장치가 태그 오차만큼 샌다.")

    # ② λ 최적점이 어디로 움직이는가
    train_p, _ = G.split_pool(pool_p, test_frac=0.5, seed=7)
    train_e, _ = G.split_pool(pool_e, test_frac=0.5, seed=7)
    prior_e = z((g.tech + g.aes) / 2)
    lam_grid = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

    print(f"\n  ② 상위 {TOP_K}장 평균 만족도와 **최적 λ** (고객 {N_PERSONS}명 평균)")
    print(f"     {'':<18}" + "".join(f"{f'λ={l:g}':>8}" for l in lam_grid) + f"{'최적λ':>7}{'설계곡선':>9}")
    for n_pairs in (12, 100):
        for label, gg, pp, tt in (
            ("태그 완벽", g_perfect, prior, train_p),
            ("태그 실측오차", g, prior_e, train_e),
        ):
            vals, curve = best_lambda_row(gg, persons, pp, tt, n_pairs, lam_grid)
            star = lam_grid[int(np.argmax(vals))]
            name = f"{n_pairs}쌍 · {label}"
            print(f"     {name:<18}" + "".join(f"{v:>8.3f}" for v in vals)
                  + f"{star:>7.1f}{curve:>9.3f}")
    print(f"  → 12쌍에서 최적 λ가 0.4 → 0.2로 내려간다. 설계 곡선은 λ(12)={lambda_of(12):.2f}다.")
    print("     **태그가 흐리면 같은 쌍 수라도 취향을 덜 믿어야 한다.** λ = f(n)은 쌍의 개수만")
    print("     보는데, 증거의 질은 태그 정확도에도 달려 있다. 100쌍 구간에서는 최적점이")
    print("     그대로다 — 양이 쌓이면 오차를 견딘다. 흔들리는 건 **초반 구간**뿐이다.")
    print("  ※ 되먹임: λ 곡선의 k(=10)는 태그 정확도가 확정되기 전에는 확정할 수 없다.")

    # ③ conf에 축 신뢰도를 곱하면 이득이 있는가 — 있을 것 같지만 없다
    print("\n  ③ conf(axis)에 축별 태그 신뢰도를 곱해 보면 (태그 실측오차 갤러리)")
    print(f"     {'증거(쌍)':<10}{'conf만':>10}{'conf×신뢰도':>13}")
    for n_pairs in (12, 100):
        plain, rel = [], []
        for pi, w_true in enumerate(persons):
            sat = satisfaction(g, w_true)
            tr = G.sample_pairs(train_e, n_pairs, seed=1000 + pi)
            c, r = G.answer_pairs(g, w_true, tr, temp=TEMP, seed=2000 + pi)
            w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
            l = lambda_of(n_pairs)
            for bucket, relmap in ((plain, None), (rel, G.AXIS_RELIABILITY_2026_08_25)):
                cf = axis_confidence(g, w_hat, c, r, reliability=relmap)
                sc = (1 - l) * prior_e + l * z(pref_scores(g, w_hat, cf))
                bucket.append(sat[np.argsort(-sc)[:TOP_K]].mean())
        print(f"     {n_pairs:<10}{np.mean(plain):>10.3f}{np.mean(rel):>13.3f}")
    print("  → 거의 차이가 없다. 그럴듯한 아이디어인데 왜 안 먹히나:")
    print("     (a) conf를 곱한 뒤 pref를 z-표준화하므로 **모든 축을 같은 비율로 줄이는 부분은")
    print("         상쇄된다**. [C]에서 본 것과 같은 이유다. 남는 건 축 사이의 상대 변화뿐이고,")
    print("         신뢰도 0.74~0.98은 그 상대 변화가 작다.")
    print("     (b) 태그가 부정확한 축은 선택과 어긋나는 쌍이 늘어 p̂이 0.5로 내려간다 —")
    print("         **conf가 이미 부분적으로 흡수하고 있다.** 곱하면 이중으로 누르는 셈이다.")
    print("  → 교훈은 [C]와 같다: 같은 정보를 두 군데서 빼지 않는다. 태그 품질은 conf가 아니라")
    print("     **λ 쪽(=증거의 질)** 이나 태그 파이프라인 자체에서 다뤄야 한다.")


if __name__ == "__main__":
    main()
