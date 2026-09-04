"""STEP 1 — Bradley-Terry는 '차이 벡터에 대한 로지스틱 회귀'다.

────────────────────────────────────────────────────────────────────────────
이론 (읽고 나서 실행할 것)

1) BT 원형. 아이템 i가 j를 이길 확률
       P(i ≻ j) = π_i / (π_i + π_j),   π > 0
   β = log π 로 치환하면
       P(i ≻ j) = e^{β_i} / (e^{β_i} + e^{β_j}) = σ(β_i − β_j),  σ(z)=1/(1+e^{-z})
   즉 "승률은 두 실력의 **차이**만으로 정해진다". 절대값은 의미 없다 —
   모든 β에 상수를 더해도 확률이 안 바뀐다(식별 불가). 그래서 절편을 두지 않는다.

2) contextual BT (우리가 쓸 형태). 아이템별 자유 파라미터 β_i 대신
       β_i = w · x_i    (x_i = 사진 i의 피처 벡터)
   를 넣으면
       P(i ≻ j) = σ(w · (x_i − x_j)) = σ(w · d_ij)
   → **피처 차이 벡터 d에 대한, 절편 없는 로지스틱 회귀**와 완전히 같은 모델.
   BT loss를 따로 구현할 필요가 없다는 뜻이다.

3) 왜 우리는 반드시 2)여야 하나. 1)의 β_i는 "이 사진"에만 붙는 값이라
   처음 보는 사진에 값을 줄 수 없다. 갤러리마다 사진 집합이 완전히 다른 우리 문제에서
   1)은 원리적으로 못 쓴다. 이 스크립트가 그걸 숫자로 보여준다.

4) 로그우도와 학습.
       NLL(w) = − Σ_k log σ(w · d_k)     (k = 쌍, d_k = x_chosen − x_rejected)
       ∇NLL   = − Σ_k (1 − σ(w · d_k)) · d_k
   L2를 붙이면 MAP 추정(= w에 평균 0 가우시안 사전분포). 쌍이 적을수록 필수.

5) 부호 대칭 증강. 학습 데이터를 (d, y=1)로만 만들면 y가 전부 1이라
   sklearn이 학습을 거부한다. (−d, y=0)을 함께 넣으면 같은 해를 주면서 두 클래스가 생긴다
   (log σ(w·d) = log(1 − σ(−w·d)) 이므로 같은 항이 두 번 세어질 뿐).

상세 해설(서비스 예시로 푼 것): `../step1-explained.md`
────────────────────────────────────────────────────────────────────────────

■ 실행 방법

    # 1) 준비 — study/ 루트에 venv 하나. 이미 있으면 건너뛴다.
    cd study/01-recsys
    python3.11 -m venv ../.venv
    ../.venv/bin/pip install -r lab/requirements.txt

    # 2) 실행 — lab/ 안에서 돌린다 (metrics.py를 같은 폴더에서 import 하므로)
    cd lab
    ../../.venv/bin/python step1_bt.py

CPU로 1초 안에 끝난다. GPU도, 실제 사진도, DB 연결도 필요 없다.
난수 시드가 `RNG = default_rng(0)`으로 고정돼 있어 **몇 번을 돌려도 출력이 같다.**
숫자가 달라졌다면 코드를 건드린 것이다.

■ 이 스크립트가 증명하려는 것 (출력 블록과 1:1 대응)

    [a vs b]  BT를 직접 구현한 해 == sklearn 로지스틱 회귀의 해
              → BT loss를 따로 짤 필요가 없다 (이론 2, 5의 확인)
    [홀드아웃] 처음 보는 쌍의 선택을 맞히는가 + 그 숫자를 믿어도 되는가
              → 바닥(0.5) / 천장(w_true) / 신뢰구간 세 개를 함께 읽는 법
    [c]       사진마다 파라미터를 두는 원형 BT는 왜 못 쓰는가 (이론 3)
    [d]       쌍이 몇 개 있어야 하는가 → 온보딩 쌍 개수 설계의 근거

■ 출력을 읽는 법 (자주 하는 오독)

    · "정확도 0.733" 을 절대 수치로 읽지 말 것. 바닥은 0.5, 천장은 0.744다.
      천장이 100%가 아닌 이유는 사람이 확률적으로 고르기 때문(라벨 노이즈).
    · [d] 표는 단조 증가가 아니다. 홀드아웃 90쌍의 95% CI 폭이 ±9%p라서
      0.589와 0.689의 차이는 "구분 가능한 차이"가 아니다. 노이즈다.
    · w_true는 합성 세계에만 있다. 실전에는 천장을 알려주는 줄이 없다.

■ 변형 과제 (숫자가 어떻게 움직이는지 직접 볼 것)

    1. reg = 0.0 으로 두고 N_PAIRS 를 8로 줄여 본다 → w가 폭주하는가? (선형분리 → MLE 발산)
    2. C = 1.0/(2*reg) 의 2를 지워 본다 → [a vs b]의 두 해가 얼마나 벌어지는가?
    3. fit_intercept=True 로 바꿔 본다 → clf.intercept_ 가 0 근처인가? (부호 대칭 증강의 효과)
    4. D = 33 으로 올려 본다 (태그 차원) → 8~12쌍에서 정확도가 어떻게 되는가? = step2의 질문
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from metrics import binom_p_value, pairwise_accuracy, wilson_interval

# 시드 고정 — 재현 가능해야 "코드를 바꿨더니 숫자가 변했다"를 신뢰할 수 있다.
RNG = np.random.default_rng(0)

# N_ITEMS: 갤러리 사진 수에 해당 / D: 피처 차원 / N_PAIRS: 부부가 답한 쌍 비교 수
# 여기서 D=5는 일부러 작게 잡았다. 실제 태그는 33차원인데, 그건 step2의 주제다.
N_ITEMS, D, N_PAIRS = 200, 5, 300


# ── 직접 구현: BT 음의 로그우도와 기울기 ────────────────────────────────────
def bt_nll_and_grad(w: np.ndarray, d: np.ndarray, reg: float) -> tuple[float, np.ndarray]:
    """이론 4)의 두 식을 그대로 코드로 옮긴 것. 최적화기에 넘길 '목적함수 + 기울기'.

    역할
      - nll : 이 w가 얼마나 나쁜가 (작을수록 좋음). 최적화기가 낮추려는 대상.
      - grad: 어느 방향으로 움직이면 nll이 줄어드는가.

    무엇을 증명하는가
      - 정확도(계단함수)로는 못 하는 일을 NLL은 한다. NLL은 매끄럽고 볼록(convex)해서
        어디서 출발하든 같은 전역 최적해로 간다 → `np.zeros`에서 시작해도 되는 근거.

    인자
      w   : (D,)   현재 취향 벡터 후보
      d   : (K, D) 쌍마다의 차이 벡터. **전부 chosen − rejected 방향**으로 만들어져 있어서
                   모든 항이 σ(w·d) 형태가 된다 (그래서 y 라벨이 식에 등장하지 않는다).
      reg : L2 세기. reg = 1/τ² (τ = w에 대한 가우시안 사전분포의 표준편차).
    """
    # z_k = w·d_k. 부호가 곧 예측이다: z > 0 이면 "chosen이 이긴다"고 맞게 예측한 것.
    z = d @ w

    # NLL = Σ −log σ(z) = Σ softplus(−z) = Σ log(1 + e^{−z}).
    # np.logaddexp(0, -z)를 쓰는 이유는 수치 안정성이다. z가 -800쯤 되면
    # np.exp(-z)가 inf로 터지지만 logaddexp는 안 터진다.
    # 뒤의 0.5*reg*w@w 가 L2 = MAP의 사전분포 항. 이게 없으면 쌍이 선형분리 가능할 때
    # ‖w‖를 키울수록 NLL이 계속 내려가서 최소점이 무한대에 있다(= 발산).
    nll = np.logaddexp(0.0, -z).sum() + 0.5 * reg * w @ w

    sig = 1.0 / (1.0 + np.exp(-z))  # σ(z_k) = 이 쌍의 선택을 맞힐 확률

    # ∇NLL = −Σ (1 − σ(z_k))·d_k + reg·w
    # 가중치 (1 − σ)가 곧 '이 쌍을 틀릴 확률'이다. 그래서:
    #   이미 잘 맞추는 쌍(σ≈1) → 가중치 0 → 아무것도 안 가르친다
    #   틀리는 쌍(σ≈0)        → 가중치 1 → w를 그 쌍의 d 방향으로 세게 당긴다
    # "모델은 지금 틀리는 쌍에서만 배운다" = 온보딩 쌍 선택 전략(active learning)의 근거.
    grad = -((1.0 - sig)[:, None] * d).sum(axis=0) + reg * w
    return float(nll), grad


def fit_bt(d: np.ndarray, reg: float = 1.0) -> np.ndarray:
    """쌍 차이 벡터 d(모두 chosen−rejected)로 w의 MAP 추정.

    역할: 위 목적함수를 L-BFGS-B로 최소화해 취향 벡터 하나를 돌려준다.
          이 함수의 반환값이 STEP 1의 유일한 산출물이다 (추천 목록이 아니다).

    구현 메모
      - `jac=True` : 기울기를 우리가 직접 준다는 뜻. 수치 미분보다 빠르고 정확하다.
      - 초기값 `np.zeros(D)` : w=0은 "아무 취향 없음"(모든 쌍을 50:50으로 예측).
        목적함수가 볼록이라 초기값에 결과가 의존하지 않는다 → 재실행 멱등.
      - 이 함수는 d의 열이 무엇인지 모른다. 그래서 (b)에서는 피처 차이,
        (c)에서는 원핫 차이를 그대로 넣어 **같은 코드로 두 모델을 비교**할 수 있다.
    """
    res = minimize(bt_nll_and_grad, np.zeros(d.shape[1]), args=(d, reg), jac=True, method="L-BFGS-B")
    return res.x


def main() -> None:
    # ── 가짜 세계 ────────────────────────────────────────────────────────
    # 여기서 만드는 것은 "정답을 아는 우주"다. 실전에는 w_true가 없다.
    # 굳이 만드는 이유는 단 하나 — **천장(도달 가능한 최대 정확도)을 알기 위해서**다.
    w_true = np.array([1.0, -0.5, 0.3, 0.0, 0.8])   # 이 부부의 진짜 취향 (실전에선 미지)

    # 피처는 난수다. 태그도 임베딩도 아니다. 의도적이다 —
    # STEP 1이 확인하려는 건 "사진이 무엇이든 상관없이 참인 성질"이라,
    # 여기서 진짜 태그를 쓰면 "태그가 좋아서 잘 나온 건지 수학이 성립해서인지" 구분이 안 된다.
    X = RNG.normal(size=(N_ITEMS, D))
    u = X @ w_true                                   # 진짜 효용

    # 무작위로 쌍을 만들고 자기 자신끼리 붙은 것만 제거한다.
    pairs = RNG.integers(0, N_ITEMS, size=(N_PAIRS, 2))
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]

    # ★ 핵심: 사람은 확률적으로 고른다 — BT가 가정하는 그 방식 그대로 라벨을 만든다.
    #   점수 차가 크면 거의 항상 그쪽을 고르지만(σ(3)=0.95), 차이가 작으면 자주 뒤집힌다.
    #   이 한 줄 때문에 "정답 w를 알아도 정확도 100%가 안 되는" 천장이 생긴다.
    p_first = 1.0 / (1.0 + np.exp(-(u[pairs[:, 0]] - u[pairs[:, 1]])))
    first_wins = RNG.random(len(pairs)) < p_first
    chosen = np.where(first_wins, pairs[:, 0], pairs[:, 1])
    rejected = np.where(first_wins, pairs[:, 1], pairs[:, 0])

    # ── 학습/홀드아웃 분할은 '쌍 단위'로 ────────────────────────────────
    # (증강 후에 나누면 같은 쌍의 +d와 −d가 양쪽에 갈라져 정보가 샌다)
    # 실전 대응: 부부가 답한 12쌍 중 9쌍으로 배우고 3쌍은 숨겨뒀다 채점에 쓴다.
    # 여기서 '정답'은 우리가 정한 라벨이 아니라 **그 사람이 실제로 한 선택**이다.
    idx = RNG.permutation(len(chosen))
    n_tr = int(0.7 * len(idx))
    tr, te = idx[:n_tr], idx[n_tr:]

    # 학습 입력은 차이 벡터 하나로 압축된다. 두 사진이 같은 값을 가진 축은 여기서 0이 되어
    # 사라진다 → "쌍은 두 사진이 다른 축만 가르친다" (온보딩 쌍 규칙의 뿌리).
    d_tr = X[chosen[tr]] - X[rejected[tr]]

    print("=" * 74)
    print(f"쌍 {len(chosen)}개 (학습 {len(tr)} / 홀드아웃 {len(te)}), 피처 {D}차원")
    print("=" * 74)

    # ── (a) 직접 구현한 BT MLE ──────────────────────────────────────────
    # 이론 4)를 그대로 푼 것. 비교의 기준선(reference implementation)이다.
    reg = 1.0
    w_bt = fit_bt(d_tr, reg=reg)

    # ── (b) sklearn 로지스틱 회귀 (부호 대칭 증강) ───────────────────────
    # 증명하려는 것: "BT = 절편 없는 로지스틱 회귀"라면 (a)와 (b)의 해가 같아야 한다.
    #
    # 증강이 필요한 이유: d를 chosen−rejected로만 만들면 정답이 전부 y=1이라
    #   sklearn이 "클래스가 하나뿐"이라며 학습을 거부한다.
    #   log σ(w·d) = log(1 − σ(−w·d)) 이므로 (−d, 0)은 같은 사실을 뒤집어 쓴 것뿐이다.
    #   부수 효과로 데이터가 원점 대칭이 되어, 절편을 켜도 0으로 추정된다.
    #
    # 증강하면 각 쌍의 손실이 두 번 세어진다. sklearn 목적함수는
    #   C · Σ logloss + ½‖w‖²  이므로  C = 1/(2·reg) 로 맞추면 (a)와 같은 문제가 된다.
    #   (유도: sklearn 쪽을 2C로 나누면 Σlogloss + (1/4C)‖w‖². 1/(4C) = ½·reg → C = 1/(2reg))
    #   sklearn의 C는 정규화의 '역수'라는 점에 주의.
    # tol을 1e-10까지 조인 이유: 두 해가 같음을 소수점 6자리로 보이려면 수렴을 끝까지 시켜야 한다.
    Xa = np.vstack([d_tr, -d_tr])
    ya = np.concatenate([np.ones(len(d_tr)), np.zeros(len(d_tr))])
    clf = LogisticRegression(fit_intercept=False, C=1.0 / (2 * reg), tol=1e-10, max_iter=5000)
    clf.fit(Xa, ya)
    w_lr = clf.coef_[0]

    # 기대 출력: 최대 절대 차이가 1e-5 이하 → 두 방법이 같은 모델임을 수치로 확인.
    # 실무 함의: 앞으로 BT loss를 직접 짜지 않고 sklearn을 쓰면 된다(코드·버그 절약).
    print("\n[a vs b] 직접 구현 BT MLE  vs  sklearn 로지스틱 회귀")
    print(f"  w_true    : {np.round(w_true, 3)}")
    print(f"  BT (직접) : {np.round(w_bt, 3)}")
    print(f"  LogReg    : {np.round(w_lr, 3)}")
    print(f"  두 해의 최대 절대 차이: {np.abs(w_bt - w_lr).max():.2e}  ← 0에 가까우면 같은 모델")

    # ── 홀드아웃 정확도와 그 유의성 ─────────────────────────────────────
    # pairwise_accuracy: 학습에 안 쓴 쌍에서 (X[chosen]−X[rejected])·w > 0 인 비율.
    #   설계안 §6의 주지표. 정답 라벨이 아니라 '그 사람의 선택'으로 채점한다는 점이 핵심.
    acc = pairwise_accuracy(w_bt, X, chosen[te], rejected[te])
    k, n = int(round(acc * len(te))), len(te)

    # 정확도 하나만 보면 안 된다. 표본이 작으면 우연히 나온 숫자일 수 있으므로
    #   wilson_interval : 그 비율의 95% 신뢰구간 (작은 표본에서 정규근사보다 정직)
    #   binom_p_value   : "우연 50%보다 낫다"의 단측 이항검정
    # 판정 기준: CI가 0.5를 포함하면 아직 아무 말도 할 수 없다.
    lo, hi = wilson_interval(k, n)
    print(f"\n[홀드아웃] 쌍 예측 정확도 {acc:.3f}  ({k}/{n})")
    print(f"  95% CI [{lo:.3f}, {hi:.3f}]   p(우연 50% 대비) = {binom_p_value(k, n):.4f}")

    # 천장(ceiling). 정답 w_true를 그대로 써도 100%가 안 나온다 —
    # 라벨 생성이 확률적이라 라벨 자체에 노이즈가 있기 때문이다.
    # 그래서 0.733은 "73%밖에"가 아니라 "천장의 98.5%"로 읽어야 한다.
    acc_oracle = pairwise_accuracy(w_true, X, chosen[te], rejected[te])
    print(f"  참고: 정답 w_true를 그대로 써도 {acc_oracle:.3f}  ← 도달 가능한 천장")
    print("  → 우연 기준선은 0.500이고 천장은 100%가 아니다. 사람이 확률적으로 고르기 때문에")
    print("     라벨 자체에 노이즈가 있다. 목표치는 천장 대비로 읽어야 한다.")

    # ── (c) 아이템별 BT는 왜 못 쓰는가 ──────────────────────────────────
    # 피처 대신 '사진 한 장 = 파라미터 하나'로 두면(원형 BT), 학습에서 본 적 없는
    # 사진에는 β가 없다. 우리 문제에서 갤러리가 바뀌면 전부 처음 보는 사진이다.
    #
    # 실험 설계: 같은 fit_bt에 '원핫 차이'를 넣으면 그게 곧 원형 BT다.
    #   onehot[i] − onehot[j] 는 i번째만 +1, j번째만 −1인 벡터이므로
    #   w·d = β_i − β_j 가 되어 이론 1)의 식이 그대로 재현된다.
    #   → 두 모델의 차이는 오직 '피처를 쓰느냐'뿐이라는 공정한 대조군이 된다.
    onehot = np.eye(N_ITEMS)
    d_item_tr = onehot[chosen[tr]] - onehot[rejected[tr]]
    beta = fit_bt(d_item_tr, reg=1.0)

    # 학습에서 한 번이라도 등장한 사진 집합. 여기 없는 사진은 β가 0(사전분포 그대로)이라
    # 사실상 동전 던지기다. 실전에서는 새 갤러리의 100%가 이 상태가 된다.
    seen = set(chosen[tr]) | set(rejected[tr])
    unseen_mask = np.array([c not in seen or r not in seen for c, r in zip(chosen[te], rejected[te])])
    acc_item_all = pairwise_accuracy(beta, onehot, chosen[te], rejected[te])
    print(f"\n[c] 아이템별 BT(피처 없이 사진마다 파라미터 1개)")
    print(f"  홀드아웃 정확도 {acc_item_all:.3f}  (그중 처음 보는 사진이 낀 쌍 {unseen_mask.sum()}/{n}개)")
    print("  → 파라미터 수가 사진 수만큼 늘고, 새 갤러리에는 아예 값이 없다.")
    print("     그래서 우리는 β_i = w·x_i (contextual BT)만 쓴다.")

    # ── (d) 쌍 개수 → 정확도 곡선 ───────────────────────────────────────
    # 학습 쌍 수만 바꾸고 나머지(세계·홀드아웃·정규화)는 고정해서, 온보딩을 몇 쌍
    # 받아야 하는지의 감을 잡는다. tr[:m]으로 앞에서부터 잘라 쓰므로 m이 커질수록 포함 관계다.
    #
    # 주의(오독 방지): 이 표는 단조 증가가 아니다. 홀드아웃 90쌍의 CI 폭이 ±9%p라
    # 0.589와 0.689의 차이는 통계적으로 구분 가능한 차이가 아니다. 노이즈다.
    print("\n[d] 쌍이 몇 개 있어야 하나 (같은 세계, 학습 쌍 수만 변경)")
    print("  학습쌍 | 홀드아웃 정확도")
    for m in (8, 12, 25, 50, 100, len(tr)):
        w_m = fit_bt(X[chosen[tr[:m]]] - X[rejected[tr[:m]]], reg=reg)
        print(f"  {m:>6} | {pairwise_accuracy(w_m, X, chosen[te], rejected[te]):.3f}")
    print("  → 온보딩 8~12쌍만으로 5차원도 이 정도다. 33차원 태그·768차원 임베딩은 step2에서.")


if __name__ == "__main__":
    main()
