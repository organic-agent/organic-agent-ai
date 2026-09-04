"""STEP 2 — 피처 설계: 33차원 태그 vs 768차원 임베딩, 정규화, 쌍 생성 규칙.

────────────────────────────────────────────────────────────────────────────
이론

1) p ≫ n 문제. 학습할 파라미터 수 p가 데이터 수 n보다 크면 해가 무수히 많다.
   온보딩 쌍은 8~12개인데 DINOv2 임베딩은 768차원 — 12개 방정식으로 768개 미지수를
   정하는 셈이다. 정규화 없이 풀면 학습 쌍은 100% 맞히고 홀드아웃은 우연 수준이 된다(과적합).
   대응은 셋 중 하나 이상: ① 차원을 줄인다(PCA·태그) ② 강하게 정규화한다(L2)
   ③ 데이터를 늘린다(쌍을 더 받는다).

2) L2 정규화 = 사전분포. ½·reg·‖w‖² 를 더하는 것은 "w는 평균 0 근처일 것"이라는
   가우시안 사전분포를 넣는 것과 같다(MAP). 증거가 적을수록 사전분포로 끌어당긴다 —
   설계안의 λ = f(증거량)과 같은 사상이다(step3).

3) 태그 원핫의 식별 문제. 사진 한 장은 축마다 태그를 정확히 하나 갖는다. 그래서
   한 축 안의 원핫 값들을 전부 +c 해도 차이 벡터 d의 그 축 성분 합은 0 →
   예측이 안 바뀐다. 즉 **축 내부의 상대값만 의미가 있다.** 계수를 해석할 때는
   축별로 평균을 빼고 봐야 한다. L2는 이 자유도를 "가장 작은 노름"으로 자동 고정해 준다.

4) 쌍을 아무렇게나 만들면 안 되는 이유. 설계안의 규칙(같은 scene · 한 축만 다름 ·
   품질 근접)은 답이 그 한 축의 취향만 반영하도록 **교란변수를 제거**하는 장치다.
   랜덤 쌍은 "화질이 좋아서" 고른 건지 "클로즈업이라서" 고른 건지 섞인다.
   반대급부: ① 축 간 중요도를 못 배우고 ② scene은 항상 같으므로 **scene 취향은
   원리적으로 배울 수 없다**(실험 D에서 확인). 그래서 장면 배분은 취향이 아니라
   커버리지라는 별도 장치가 맡는다(step4).

5) 천장(ceiling). 사람이 확률적으로 고르므로 정답 w_true를 알아도 100%는 못 맞힌다.
   게다가 설계 규칙 쌍은 일부러 '둘 다 괜찮은' 어려운 쌍이라 천장이 더 낮다.
   모든 숫자는 0.5(우연)와 천장 사이 어디인지로 읽어야 한다.

상세 해설(서비스 예시로 푼 것): `../step2-explained.md`
선행: `step1_bt.py` / `../step1-explained.md`
────────────────────────────────────────────────────────────────────────────

■ 실행 방법

    # 준비가 안 됐다면 step1_bt.py의 실행 방법을 먼저 볼 것 (venv + requirements)
    cd study/01-recsys/lab
    ../../.venv/bin/python step2_features.py

**실측 약 13초** (step1은 0.5초). 차이는 실험 규모다 —
표현 4종 × 쌍 수 5종 × 사람 12명 × reg 후보 6개(교차검증) = fit_bt 수천 번.
CPU만 쓰고 GPU·사진·DB는 필요 없다. 모든 시드가 고정돼 있어 **재실행하면 숫자가 같다.**

느린 기계에서 답답하면 `N_PERSONS`를 4로 줄인다. 단, 표준편차가 커져서 실험 A의 결론이 흐려진다.

■ 이 스크립트가 증명하려는 것 (출력 블록과 1:1 대응)

    ※ 천장   설계 규칙 쌍은 '둘 다 괜찮은' 어려운 쌍이라 천장이 0.662로 낮다.
             (step1은 0.744) 모든 숫자를 0.500~0.662 사이 어디인지로 읽게 만드는 기준선.
    [A]      표현 4종 × 쌍 수 5종. **12쌍으로는 33차원도 못 배운다**(0.484±0.09)는 것과,
             쌍이 적을 때 사진학 2열이 이긴다는 것 → λ 구조의 실험적 근거.
    [B]      정규화 세기의 효과. 차원 수에 따라 방향이 반대이고,
             **12쌍에서는 어떤 reg도 소용없다** → "정규화는 없는 정보를 만들지 않는다".
    [C]      쌍 생성 규칙의 가치. 평가 쌍 종류만 바꿔도 숫자가 크게 흔들린다
             → 정확도를 보고할 때 쌍 규칙을 함께 적어야 한다는 근거.
    [D]      배운 w를 축별로 해석. scene은 구조적으로 학습 불가,
             의견 없는 축은 계수가 노이즈 → conf(axis)가 필요한 이유.

■ 출력을 읽는 법 (자주 하는 오독)

    · [A]의 0.484를 "우연보다 나쁘다"로 읽지 말 것. ±0.09가 붙어 있다 =
      12명이 0.39~0.58로 흩어진다는 뜻. **추정이 잡음**이라는 뜻이지 방향이 반대라는 게 아니다.
    · [C]의 0.754를 "랜덤 쌍이 더 좋다"로 읽지 말 것. 평가 쌍이 쉬웠을 뿐이다
      (랜덤 평가 쌍의 천장 0.790, 설계 평가 쌍의 천장 0.603). 세로로 같은 평가 쌍끼리 비교할 것.
    · [D]의 framing 상관 −0.68을 실패로 읽지 말 것. 사람 #0은 framing에 의견이 없다
      (진짜 취향의 중심화 표준편차 0.077). 노이즈끼리의 상관이라 부호에 의미가 없다.

■ 변형 과제

    1. PCA 차원을 8/16/64로 바꿔 본다 → 어느 지점에서 태그를 따라잡는가? (차원 축소의 한계)
    2. `pct_gap`(품질 근접 기준, gallery.design_pair_pool)을 30으로 풀어 본다
       → 쌍이 쉬워지면서 천장과 정확도가 함께 올라간다. "정확도가 올랐다"고 말할 수 있나?
    3. STRENGTH를 0.5로 낮춰 본다 (취향이 약한 고객) → 몇 쌍부터 우연과 구분되나?
    4. TEMP를 1.5로 올려 본다 (변덕이 심한 고객) → 천장이 얼마나 내려가나?
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA

import gallery as G
from metrics import pairwise_accuracy
from step1_bt import fit_bt      # ★ step1에서 만든 학습기를 그대로 재사용한다

# 실험 설정 — 사람의 취향 세기와 변덕(temp)은 이 값으로 고정한다
#   STRENGTH: w_true의 크기. 크면 취향이 뚜렷한 고객 = 쌍이 쉬워진다
#   TEMP    : 선택의 무작위성. P = σ((u_i − u_j)/temp). 크면 변덕스럽다 = 라벨 노이즈 증가
#   두 값이 천장을 직접 결정하므로, 바꾸면 모든 표의 기준선이 함께 움직인다.
STRENGTH, TEMP = 1.5, 0.7

# 사람 12명. step1은 1명이었다. 여기서 여러 명을 쓰는 이유는 **표준편차를 보기 위해서**다.
# 평균만 보면 "12쌍에서 0.484"가 되지만, ±0.09가 붙어야 "잡음"이라는 진짜 결론이 나온다.
N_PERSONS = 12

# reg 후보. 교차검증으로 이 중 하나를 고른다(pick_reg_by_cv).
REG_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def build_features(g: G.Gallery) -> dict[str, np.ndarray]:
    """같은 갤러리를 네 가지 방식으로 표현한다. 열 스케일을 맞춰 reg 비교를 공정하게.

    역할: 실험 A의 '가로줄 4개'를 만든다. 갤러리·사람·쌍은 완전히 동일하고
          **표현 방식만** 다르게 해서, 차이가 오직 피처에서 왔음을 보장한다.

    왜 스케일을 맞추는가 (중요)
      L2는 ‖w‖²를 벌하므로, 열의 크기가 크면 그 열의 w가 작아도 되어 **덜 벌 받는다**.
      스케일이 제각각이면 "reg=1.0"이 표현마다 다른 세기를 뜻하게 되어 실험 B가 무의미해진다.
      그래서 PCA는 열별 std로, 임베딩은 전체 평균 std로 나눠 대역을 맞춘다.

    네 표현이 각각 대표하는 것
      태그 원핫(33)   : 실제 파이프라인의 VLM 고정 축 태그. 저차원·해석 가능
      태그+사진학(35) : 위 + technical_pct/aesthetic_pct. 사진학 prior가 피처로 들어간 형태
      임베딩 raw(768) : DINOv2 자리. 고차원·해석 불가 → p≫n의 주인공
      임베딩 PCA(32)  : 차원 축소가 실제로 듣는지 보는 대조군
    """
    pca32 = PCA(n_components=32, random_state=0).fit_transform(g.emb)
    pca32 /= pca32.std(axis=0, keepdims=True)
    emb = g.emb / g.emb.std(axis=0, keepdims=True).mean()
    prior = np.column_stack([g.tech / 100.0, g.aes / 100.0])   # 백분위 0~100 → 0~1
    return {
        "태그 원핫 (33)": g.X,
        "태그+사진학 (35)": np.hstack([g.X, prior]),
        "임베딩 raw (768)": emb,
        "임베딩 PCA (32)": pca32,
    }


def pick_reg_by_cv(F: np.ndarray, c: np.ndarray, r: np.ndarray, folds: int = 4) -> float:
    """정규화 강도를 **학습 쌍 안에서** 교차검증으로 고른다.

    홀드아웃 정확도를 보고 reg를 고르면 그 홀드아웃 수치는 더 이상 정직하지 않다.
    쌍이 12개뿐이면 fold 하나가 3쌍이라 이 선택 자체가 불안정하다 — 그것도 결과다.

    역할: 실험 A를 공정하게 만드는 장치. 표현마다 최적 reg가 다른데 하나로 고정하면
          "reg가 잘 맞은 표현"이 이겨 버린다. 그래서 표현마다 알아서 고르게 한다.

    무엇을 증명하는가 (실험 방법론 그 자체)
      **평가에 쓸 데이터로 하이퍼파라미터를 고르면 안 된다.** 홀드아웃을 한 번이라도
      들여다보고 고른 선택은 그 홀드아웃 수치를 낙관적으로 부풀린다(선택 편향).
      그래서 학습 쌍만 4겹으로 쪼개 그 안에서 고른다.

    구현 메모
      - `idx[f::folds]` : f번째 fold(검증용). `np.setdiff1d`로 나머지가 학습용이 된다.
      - 쌍이 folds*2 미만이면 CV 자체가 불가능하므로 보수적으로 10.0(강한 정규화)을 쓴다.
        데이터가 없을 때는 "덜 믿는 쪽"이 기본값이어야 한다 — λ와 같은 사상.
    """
    n = len(c)
    if n < folds * 2:
        return 10.0
    idx = np.arange(n)
    best, best_acc = 10.0, -1.0
    for reg in REG_GRID:
        accs = [
            pairwise_accuracy(
                fit_bt(F[c[np.setdiff1d(idx, idx[f::folds])]] - F[r[np.setdiff1d(idx, idx[f::folds])]], reg=reg),
                F, c[idx[f::folds]], r[idx[f::folds]],
            )
            for f in range(folds)
        ]
        acc = float(np.mean(accs))
        if acc > best_acc:
            best, best_acc = reg, acc
    return best


def run_one(F, g, w_true, train_pool, test_pool, n_pairs, person_id, reg=None):
    """한 사람 × 한 설정: 학습 쌍 n_pairs개로 학습 → 홀드아웃 풀 전체로 평가.

    역할: 이 스크립트의 모든 표는 이 함수를 조건만 바꿔 반복 호출한 결과다.
          "한 번의 온보딩 → 한 번의 채점"이 여기 한 줄로 압축돼 있다.

    흐름 (실제 서비스와 1:1 대응)
      sample_pairs  : 후보 풀에서 n_pairs개를 뽑는다        = 온보딩에서 무엇을 물어볼지 고름
      answer_pairs  : 그 사람이 확률적으로 고른다            = 부부가 화면에서 클릭
      pick_reg_by_cv: 학습 쌍 안에서 정규화 세기 선택        = 하이퍼파라미터 튜닝
      fit_bt        : 차이 벡터로 w 추정 (step1의 그 함수)   = 취향 학습
      pairwise_acc  : 홀드아웃 풀 전체로 채점               = 숨겨둔 쌍으로 검증

    시드 규칙 (1000/2000/3000 + person_id)
      사람마다 다른 쌍·다른 답을 받되 **재실행하면 같은 결과**가 나오게 하는 장치.
      학습 답(2000+)과 평가 답(3000+)의 시드를 분리해 둔 것이 중요하다 —
      같은 시드면 같은 난수열이라 평가가 학습을 따라가 버린다.

    누수 방지
      train_pool과 test_pool은 main에서 이미 겹치지 않게 갈라져 있다(split_pool).
      같은 쌍이 양쪽에 있으면 정확도가 부풀려진다.
    """
    tr = G.sample_pairs(train_pool, n_pairs, seed=1000 + person_id)
    c_tr, r_tr = G.answer_pairs(g, w_true, tr, temp=TEMP, seed=2000 + person_id)
    c_te, r_te = G.answer_pairs(g, w_true, test_pool, temp=TEMP, seed=3000 + person_id)
    if reg is None:
        reg = pick_reg_by_cv(F, c_tr, r_tr)
    w_hat = fit_bt(F[c_tr] - F[r_tr], reg=reg)
    return pairwise_accuracy(w_hat, F, c_te, r_te)


def main() -> None:
    # ── 세계 만들기 ──────────────────────────────────────────────────────
    # 1,000장은 실제 웨딩 갤러리 규모다. step1의 200장보다 크게 잡은 이유는
    # 설계 규칙을 통과하는 쌍이 워낙 적어서 — 아래 출력에서 771개(0.154%)가 나온다.
    g = G.make_gallery(1000, seed=0)
    feats = build_features(g)

    # 사람 12명. make_person은 축 5개 중 **2개만** 뚜렷한 의견을 주고 나머지는 거의 0으로 둔다
    # (실제 사람이 모든 축에 의견이 있지는 않다). 실험 D가 이 구조를 그대로 드러낸다.
    persons = [G.make_person(seed=s, strength=STRENGTH) for s in range(N_PERSONS)]

    # 설계안 §3-B의 온보딩 쌍 규칙을 만족하는 **모든** 후보 쌍을 먼저 만들고,
    # 학습용/평가용으로 절반씩 갈라 둔다. 쌍 단위로 갈라야 누수가 없다(step1 5절과 같은 이유).
    pool = G.design_pair_pool(g)
    train_pool, test_pool = G.split_pool(pool, test_frac=0.5, seed=7)
    print(f"갤러리 {len(g)}장 → 설계 규칙 후보 쌍 {len(pool)}개")
    print(f"  학습용 풀 {len(train_pool)} / 홀드아웃 풀 {len(test_pool)} (겹치지 않게 분리)")
    print(f"  가상 고객 {N_PERSONS}명, 취향 세기 {STRENGTH}, 변덕 temp={TEMP}\n")

    # ── 천장 ────────────────────────────────────────────────────────────
    # 정답 w_true를 그대로 넣었을 때의 정확도. 이 값이 아래 모든 표의 상한이다.
    # 100%가 아닌 이유 두 가지:
    #   ① 사람이 확률적으로 고른다(temp) → 라벨 노이즈
    #   ② 설계 규칙 쌍은 효용 차이가 아주 작다(실측 중앙값 0.142 ≈ 승률 52%) → 원래 어려운 문제
    # 이 줄이 없으면 "0.64면 좋은 건가?"에 답할 수 없다.
    ceiling = float(np.mean([
        pairwise_accuracy(w, g.X, *G.answer_pairs(g, w, test_pool, temp=TEMP, seed=3000 + i))
        for i, w in enumerate(persons)
    ]))
    print(f"※ 천장(정답 w_true를 그대로 썼을 때) = {ceiling:.3f} · 우연 = 0.500\n")

    # ── 실험 A ──────────────────────────────────────────────────────────
    # 묻는 것: "어떤 피처로, 쌍이 몇 개 있어야 배워지나?"
    # 통제: 갤러리·사람·쌍 풀·평가 쌍이 전부 동일하고 **표현과 쌍 수만** 바뀐다.
    # reg는 표현마다 CV로 고르므로 "reg가 우연히 맞은 표현이 이기는" 일이 없다.
    print("=" * 80)
    print("[A] 사진 표현 × 학습 쌍 수 — 홀드아웃 쌍 정확도 (평균±표준편차, reg는 학습 쌍 CV)")
    print("=" * 80)
    n_list = (8, 12, 30, 100, 300)
    print(f"{'표현':<18}" + "".join(f"{f'{n}쌍':>13}" for n in n_list))
    for name, F in feats.items():
        row = []
        for n_pairs in n_list:
            # 12명 각각 독립 실행 → 평균과 표준편차. 표준편차가 이 실험의 핵심 산출물이다.
            a = [run_one(F, g, w, train_pool, test_pool, n_pairs, i) for i, w in enumerate(persons)]
            row.append(f"{np.mean(a):.3f}±{np.std(a):.02f}")
        print(f"{name:<18}" + "".join(f"{v:>13}" for v in row))
    print(f"\n  읽는 법 (천장 {ceiling:.2f}, 우연 0.50 사이 어디인지로 본다):")
    print("   ① 8~12쌍 구간은 어떤 표현도 0.5를 겨우 넘거나 그 아래로 떨어진다. 0.5 미만은")
    print("      '우연보다 나쁘다'가 아니라 **추정이 사실상 잡음**이라는 뜻이다(표준편차를 보라).")
    print("   ② 쌍이 늘면 태그 표현이 천장에 붙는다. 768차원은 끝까지 못 따라온다 — p≫n.")
    print("   ③ 쌍이 아주 적을 때는 '태그+사진학'이 앞선다. 사진학 2개 열이 태그 여러 개를")
    print("      요약한 저차원 대리 변수 역할을 하기 때문. 증거가 없을수록 저차원이 이긴다.")
    print("  → 결론: 온보딩 8~12쌍은 **서비스 입력량**이고, 실험용 데이터량과는 별개다.")
    print("     실험에는 사람당 수십~수백 쌍이 필요하다. step5에서 필요한 수를 계산한다.")

    # ── 실험 B ──────────────────────────────────────────────────────────
    # 묻는 것: "정규화를 세게 하면 p≫n을 이길 수 있나?"
    # 여기서는 reg를 CV로 고르지 않고 **직접 지정**한다(run_one의 reg 인자).
    # CV로 고르면 reg의 효과가 가려지기 때문 — 실험의 목적이 reg 자체이므로 통제 방식이 다르다.
    # 33차원과 768차원, 12쌍과 300쌍의 2×2를 보면 방향이 상황마다 반대인 것이 드러난다.
    print("\n" + "=" * 80)
    print("[B] 정규화 강도 reg의 효과 — 차원 수에 따라 방향이 반대다")
    print("=" * 80)
    F = feats["태그 원핫 (33)"]      # 이 F는 실험 C·D에서도 계속 쓰인다 (기본 표현)
    grid = (0.01, 0.1, 1.0, 10.0, 100.0)   # 1만 배 범위. 그런데도 12쌍에서는 거의 안 변한다
    print(f"{'표현':<16}{'학습쌍':<7}" + "".join(f"{f'reg={r:g}':>10}" for r in grid))
    for fname in ("태그 원핫 (33)", "임베딩 raw (768)"):
        Fb = feats[fname]
        for n_pairs in (12, 300):
            row = [
                f"{np.mean([run_one(Fb, g, w, train_pool, test_pool, n_pairs, i, reg=reg) for i, w in enumerate(persons)]):.3f}"
                for reg in grid
            ]
            print(f"{fname:<16}{n_pairs:<7}" + "".join(f"{v:>10}" for v in row))
    print("  읽는 법 — 세 가지가 동시에 보인다:")
    print("   ① 768차원·300쌍은 reg를 **키울수록** 좋아진다 = 약한 정규화에서 과적합하고 있다는 뜻.")
    print("   ② 33차원·300쌍은 정반대로 reg를 키울수록 손해 — 정보가 충분한데 눌렀기 때문.")
    print("   ③ 12쌍에서는 reg를 뭘 써도 거의 같다. 정규화는 **없는 정보를 만들어 주지 않는다.**")
    print("  → 이 셋이 설계안 λ 곡선의 근거다: 증거가 적으면 취향을 덜 믿고 사진학으로 돌아간다(step3).")

    # ── 실험 C ──────────────────────────────────────────────────────────
    # 묻는 것: "설계 규칙 쌍이 랜덤 쌍보다 나은가?" — 그런데 진짜 교훈은 그게 아니다.
    #
    # 2×2 설계: {설계 규칙으로 학습, 랜덤으로 학습} × {설계 쌍으로 평가, 랜덤 쌍으로 평가}
    #   가로로 읽으면 '평가 쌍의 난이도' 효과, 세로로 읽으면 '학습 쌍 규칙'의 효과가 나온다.
    #   반드시 **세로로**(같은 평가 쌍끼리) 비교해야 한다. 가로 비교는 난이도가 다른 시험지 비교다.
    #
    # 랜덤 평가 풀은 학습에 쓴 랜덤 쌍(seed=11)을 제외해서 만든다 — 누수 방지.
    print("\n" + "=" * 80)
    print("[C] 쌍 생성 규칙: 설계 규칙 쌍 vs 랜덤 쌍 (평가 쌍 종류를 바꿔 가며)")
    print("=" * 80)
    rand_pool_tr = G.make_pairs_random(g, 400, seed=11)
    rand_pool_te = [p for p in G.make_pairs_random(g, 400, seed=12) if p not in set(rand_pool_tr)]
    print(f"{'학습쌍 종류':<14}{'학습쌍':>8}{'설계 쌍으로 평가':>18}{'랜덤 쌍으로 평가':>18}")
    for kind, tr_pool in (("설계 규칙", train_pool), ("랜덤", rand_pool_tr)):
        for n_pairs in (12, 100):
            a_design = np.mean([run_one(F, g, w, tr_pool, test_pool, n_pairs, i) for i, w in enumerate(persons)])
            a_rand = np.mean([run_one(F, g, w, tr_pool, rand_pool_te, n_pairs, i) for i, w in enumerate(persons)])
            print(f"{kind:<14}{n_pairs:>8}{a_design:>18.3f}{a_rand:>18.3f}")
    print("  → 랜덤 쌍은 '쉬운 쌍'(품질 차이가 큰 쌍)이 섞여 랜덤 평가에서 점수가 높게 나온다.")
    print("     즉 정확도 숫자는 **어떤 쌍으로 쟀는지**를 밝히지 않으면 비교 불가다.")
    print("     실험 리포트에 쌍 생성 규칙을 반드시 함께 적을 것.")

    # ── 실험 D ──────────────────────────────────────────────────────────
    # 묻는 것: "배운 w를 사람이 읽을 수 있나?" — 이유 문장을 쓰려면 반드시 답해야 하는 질문.
    #
    # 300쌍(충분한 증거)으로 한 사람(#0)의 취향을 배운 뒤, 축별로 정답과 비교한다.
    # 여기서만 F가 '태그 원핫'이어야 한다 — 임베딩은 열에 이름이 없어서 해석 자체가 불가능하다.
    # (해석 가능성이 태그 표현을 쓰는 또 하나의 이유다)
    print("\n" + "=" * 80)
    print("[D] 배운 취향 읽기 — 축 내부 평균을 빼야 해석된다 (사람 #0, 학습 300쌍)")
    print("=" * 80)
    w_true = persons[0]
    tr = G.sample_pairs(train_pool, 300, seed=1000)
    c_tr, r_tr = G.answer_pairs(g, w_true, tr, temp=TEMP, seed=2000)
    w_hat = fit_bt(F[c_tr] - F[r_tr], reg=pick_reg_by_cv(F, c_tr, r_tr))
    for ax, vals in G.AXES.items():
        cols = [G.FEATURE_INDEX[f"{ax}={v}"] for v in vals]

        # ★ 축별 평균 빼기(중심화). 이론 3)의 식별 문제 때문에 필수다.
        #   한 축의 계수 전부에 +c를 더해도 예측이 안 바뀌므로, 절대값은 비교 대상이 아니다.
        #   중심화하지 않고 정답과 상관을 재면 그 자유도가 상관계수를 오염시킨다.
        est, tru = w_hat[cols] - w_hat[cols].mean(), w_true[cols] - w_true[cols].mean()

        # 표준편차가 0이면 그 축은 학습에서 신호를 한 번도 못 받은 것이다.
        # scene이 여기 걸린다 — 설계 규칙이 '같은 scene끼리만' 비교하기 때문(이론 4).
        # 버그가 아니라 규칙의 논리적 귀결. 그래서 장면 배분은 커버리지가 맡는다(step4).
        if np.std(est) < 1e-9:
            print(f"  {ax:<11} 학습 불가 — 설계 규칙상 두 사진의 {ax}가 항상 같아 신호가 0")
            continue

        # 상관계수: 추정한 축 내부 순서가 정답의 순서와 얼마나 같은가.
        # 주의 — 이 사람이 **의견이 없는 축**에서는 정답 자체가 노이즈라 부호에 의미가 없다.
        # (사람 #0의 framing은 진짜 취향의 중심화 표준편차가 0.077뿐이라 상관 −0.68이 나온다)
        # 실전에서 이걸 구분해 주는 장치가 conf(axis)다 — step3.
        corr = float(np.corrcoef(est, tru)[0, 1])
        print(f"  {ax:<11} 정답과의 상관 {corr:+.2f}   가장 선호로 추정된 값: {vals[int(np.argmax(est))]}")
    print("  → 상관이 높은 축이 그 사람이 '의견을 가진 축'. 설계안 conf(axis)가 재려는 것이 이것.")
    print("  → scene이 학습 불가인 것은 버그가 아니라 쌍 규칙의 논리적 귀결이다.")


if __name__ == "__main__":
    main()
