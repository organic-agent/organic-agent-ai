"""합성 웨딩 갤러리 — 실습 전용 데이터 생성기.

실제 사진도 임베딩도 없이 "쌍 비교 → 취향 학습 → 추천"의 수학만 먼저 돌려보기 위한
가짜 데이터다. 축 어휘는 실제 설계(`photoselect/docs/feature-design.md` §고정 축 어휘)와
같은 enum을 쓴다. 나중에 진짜 VLM 태그·DINOv2 임베딩이 나오면 이 모듈만 교체하면
step1~6 코드는 그대로 돈다.

핵심 용어
  - 태그 원핫(one-hot): 사진 한 장을 축별 태그의 0/1 벡터로 표현한 것. 33차원.
  - 효용(utility) u(p): 그 사람이 사진 p를 얼마나 좋아하는가. u = w · x.
  - w: 그 사람의 취향 벡터. 태그 하나당 실수 하나. 우리가 쌍 비교로 추정하려는 대상.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ── 고정 축 어휘 (feature-design.md와 동일) ──────────────────────────────────
AXES: dict[str, list[str]] = {
    "scene": [
        "prep", "entrance", "vow", "ring", "kiss", "family",
        "group", "bouquet", "walk", "snap", "detail", "unknown",
    ],
    "framing": ["closeup", "half", "full", "wide"],
    "lighting": ["natural", "backlit", "indoor", "flash", "lowlight"],
    "expression": ["smile", "laugh", "serious", "candid", "eyes_closed", "none"],
    "subjects": ["bride", "groom", "couple", "family", "friends", "none"],
}

FEATURE_NAMES: list[str] = [f"{ax}={v}" for ax, vals in AXES.items() for v in vals]
FEATURE_INDEX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
D_TAG = len(FEATURE_NAMES)  # 33

# ── 장면 분포 프로파일 ──────────────────────────────────────────────────────
# 실제 갤러리의 장면 분포는 균등하지 않다 (스냅·디테일이 많고 반지·부케는 적다).
# 두 프로파일을 둔 이유는 2026-08-25 VLM 1차 실측(`spike-report.md` 발견 3) 때문이다.
# 채점을 반영한 실제 분포가 `snap` 68%였고 예식 5개 값(entrance/vow/ring/family/group)이
# **0장**이었다. 그 분포가 `../dataset`의 편향인지 웨딩 갤러리의 성격인지 아직 모른다
# (본식 갤러리를 확보해야 갈린다). 그래서 기본값은 바꾸지 않고 **전환 가능한 가정**으로 둔다.
#   balanced  — 설계 문서를 만들 때 가정했던 완만한 분포. step1~6 해설의 숫자는 전부 이 값.
#   measured  — 08-25 실측 50장 채점 결과(snap 34 / walk 6 / prep 4 / 기타 6 / 예식 0).
# step4에서 두 프로파일로 각각 돌려 보면 "커버리지가 배분할 것이 있는가"가 갈린다.
SCENE_PROFILES: dict[str, list[float]] = {
    #          prep entr  vow ring kiss fam grp bouq walk snap detail unknown
    "balanced": [12,   6,  10,   3,   3,  8,  6,   3,   6,  25,   16,      2],
    "measured": [ 8,   0,   0,   0,   2,  0,  0,   2,  12,  68,    4,      4],
}


def scene_weights(profile: str = "balanced") -> np.ndarray:
    w = np.array(SCENE_PROFILES[profile], dtype=float)
    return w / w.sum()


SCENE_WEIGHTS = scene_weights("balanced")   # 하위 호환 (기존 스크립트가 참조)

# ── 태그 오류 모델 (2026-08-25 VLM 실측) ────────────────────────────────────
# **lab의 원래 가정은 "태그는 정확하다"였다.** X가 학습 피처이면서 동시에 정답이었다.
# 실측은 그렇지 않다고 말한다 — 축별 정확도 scene 74% / lighting 84% / expression 90% /
# framing 92% / subjects 98% (4bit 로컬 실행의 **하한**이라 프로덕션은 이보다 낫다).
#
# 형식: 축 → (**조건부** 오답률, 방향 | None)
#   방향이 있으면 실측된 혼동 방향만 일으키고, 오답률은 **그 정답값을 가진 사진에 대한
#   조건부 확률**이다. 전체 오답률로 적으면 안 된다 — 장면 분포가 바뀌면 뜻이 달라지기
#   때문이다(같은 0.26이 snap 27% 갤러리에서는 조건부 0.96, 68% 갤러리에서는 0.38).
#   08-25 오답은 두 축 모두 **한 방향**이었다.
#     · scene   : 정답 snap 을 예식 장면으로 과잉 분류. 채점 50장 중 snap 34장에서 13건
#                 → P(오답 | snap) = 13/34 = 0.38
#     · lighting: 정답 indoor 를 natural 로 (8건 중 8건, 예외 없음). 채점 표본의 indoor
#                 장수는 리포트에 없어 갤러리 분포(약 30%)로 환산 → 8/15 ≈ 0.53 **가정**
#   None이면 방향 정보가 없다는 뜻 — 사진마다 그 확률로 같은 축의 다른 값으로 오분류한다.
TAG_ERROR_2026_08_25: dict[str, tuple[float, dict[str, dict[str, float]] | None]] = {
    "scene":      (0.38, {"snap": {"walk": 5, "prep": 3, "detail": 2, "unknown": 2, "kiss": 1}}),
    "lighting":   (0.53, {"indoor": {"natural": 1}}),
    "expression": (0.10, None),
    "framing":    (0.08, None),
    "subjects":   (0.02, None),
}

# 축별 태그 신뢰도 = 실측 정확도. step3의 conf(axis)에 곱해 쓰는 값이다.
# conf는 원래 "이 사람이 이 축에 의견이 있는가"만 쟀는데, 실측이 **두 번째 요인**을 준다:
# "이 축의 태그를 믿을 수 있는가". 두 요인은 독립이므로 곱한다.
AXIS_RELIABILITY_2026_08_25: dict[str, float] = {
    "scene": 0.74, "framing": 0.92, "lighting": 0.84,
    "expression": 0.90, "subjects": 0.98,
}


@dataclass
class Gallery:
    """한 갤러리 = 사진 N장. 열 순서는 FEATURE_NAMES와 같다.

    태그가 두 벌인 것에 주의. 태그 오류를 끄면(기본) 둘은 완전히 같은 값이다.
      tags / X            — **VLM이 뱉은 관측 태그.** 파이프라인이 실제로 보는 것.
                            쌍 규칙 판정도, 취향 학습도, 커버리지도 전부 이쪽을 쓴다.
      tags_true / X_true  — **사진의 실제 내용.** 사람의 효용·사진학 점수·임베딩은
                            이쪽에서 나온다. 관측 불가, 실습에서만 존재하는 정답.
    이 분리가 08-25 실측의 되먹임이다. 원래는 한 벌뿐이었고, 그건 "VLM이 100% 맞다"는
    가정이었다.
    """

    tags: list[dict[str, str]]      # 사진별 축→태그 (관측)
    X: np.ndarray                   # (N, 33) 태그 원핫 (관측)
    tech: np.ndarray                # (N,) technical 백분위 0~100 (경량 3종 산출물 대역)
    aes: np.ndarray                 # (N,) aesthetic 백분위 0~100
    cluster: np.ndarray             # (N,) 근접 중복 클러스터 id
    emb: np.ndarray                 # (N, 768) DINOv2 자리를 대신하는 가짜 임베딩
    tags_true: list[dict[str, str]] | None = None   # 정답 태그 (오류 off면 tags와 동일)
    X_true: np.ndarray | None = None                # 정답 원핫 (오류 off면 X와 동일)

    def __len__(self) -> int:
        return len(self.tags)

    def scene(self, i: int) -> str:
        return self.tags[i]["scene"]

    @property
    def Xt(self) -> np.ndarray:
        """정답 원핫. 효용 계산처럼 '사진의 실제 내용'이 필요한 곳에서만 쓴다."""
        return self.X if self.X_true is None else self.X_true


def _apply_tag_error(
    tags: list[dict[str, str]],
    rng: np.random.Generator,
    spec: dict[str, tuple[float, dict[str, dict[str, float]] | None]],
    scale: float = 1.0,
) -> list[dict[str, str]]:
    """정답 태그 → 관측 태그. 실측 오답 방향과 **조건부** 오답률을 그대로 흉내낸다.

    방향이 지정된 축은 **그 정답값을 가진 사진에만** 오류가 나고, 확률은 조건부다.
    그래서 전체 오답률은 갤러리의 장면 분포에 따라 달라진다 — 그게 맞는 동작이다.
    스냅이 많은 갤러리일수록 scene 태그가 전체적으로 더 많이 틀린다.
    """
    obs = [dict(t) for t in tags]
    for ax, (rate, directions) in spec.items():
        p_cond = min(1.0, rate * scale)
        if directions:
            elig = [i for i, t in enumerate(obs) if t[ax] in directions]
            if not elig:
                continue
            for i in elig:
                if rng.random() < p_cond:
                    cand = directions[obs[i][ax]]
                    vals = list(cand)
                    wts = np.array([cand[v] for v in vals], dtype=float)
                    obs[i][ax] = str(rng.choice(vals, p=wts / wts.sum()))
        else:
            for t in obs:
                if rng.random() < p_cond:
                    others = [v for v in AXES[ax] if v != t[ax]]
                    t[ax] = str(rng.choice(others))
    return obs


def make_gallery(
    n: int = 300,
    seed: int = 0,
    emb_dim: int = 768,
    scene_profile: str = "balanced",
    tag_error: float = 0.0,
) -> Gallery:
    """사진 n장짜리 합성 갤러리.

    임베딩은 완전 난수가 아니라 태그·클러스터 구조를 담아 만든다. 난수로 두면
    step2의 표현 비교도, step4의 MMR도 "아무 일이 안 일어나는" 실습이 되어 버린다.
    실제 갤러리에서 관측되는 유사도 대역(같은 클러스터 ~0.9, 같은 장면 ~0.4)에 맞췄다.

    scene_profile — "balanced"(기본, 기존 숫자 재현) 또는 "measured"(08-25 실측 분포).
    tag_error     — 0이면 태그가 완벽하다(기본, 기존 숫자 재현). 1.0이면 08-25 실측
                    오답률 그대로, 0.5면 그 절반. **기본값을 바꾸지 않는 이유**는
                    step1·step2 해설에 박힌 숫자(천장 0.662, 12쌍 0.484 …)를
                    무효로 만들지 않기 위해서다. 실측 반영은 실험에서 켠다.
    """
    rng = np.random.default_rng(seed)

    # 근접 중복 클러스터를 먼저 만든다. 연사는 같은 순간이므로 **장면이 같다** —
    # 클러스터 단위로 scene을 정하는 것이 실제 갤러리에 가깝다.
    cluster = np.zeros(n, dtype=int)
    cid, i = 0, 0
    while i < n:
        size = int(rng.integers(1, 6))
        cluster[i : i + size] = cid
        cid += 1
        i += size
    cluster_scene = rng.choice(AXES["scene"], size=cid, p=scene_weights(scene_profile))
    scenes = cluster_scene[cluster]

    tags: list[dict[str, str]] = []
    for s in scenes:
        t = {"scene": str(s)}
        t["framing"] = str(rng.choice(AXES["framing"], p=[0.3, 0.35, 0.25, 0.10]))
        t["lighting"] = str(rng.choice(AXES["lighting"], p=[0.4, 0.15, 0.3, 0.1, 0.05]))
        if s == "detail":
            t["expression"], t["subjects"] = "none", "none"
        else:
            t["expression"] = str(
                rng.choice(AXES["expression"], p=[0.35, 0.2, 0.15, 0.22, 0.05, 0.03])
            )
            t["subjects"] = str(rng.choice(AXES["subjects"], p=[0.2, 0.15, 0.4, 0.15, 0.1, 0.0]))
        tags.append(t)

    X = onehot(tags)

    # 사진학 점수. 연사 안에서는 노출·구도가 거의 같으므로 **클러스터 공통 성분이 크고**,
    # 개별 차이는 눈 감김 같은 순간의 사고에서 온다. 이 구조 때문에 점수 상위 N장에
    # 쌍둥이 컷이 몰려 들어온다 — MMR이 필요한 진짜 이유.
    cl_tech = rng.normal(size=cid)[cluster]
    cl_aes = rng.normal(size=cid)[cluster]
    raw_tech = 0.85 * cl_tech + 0.4 * rng.normal(size=n)
    raw_tech -= 1.2 * X[:, FEATURE_INDEX["expression=eyes_closed"]]
    raw_tech -= 0.8 * X[:, FEATURE_INDEX["lighting=lowlight"]]
    raw_aes = 0.85 * cl_aes + 0.4 * rng.normal(size=n) + 0.5 * X[:, FEATURE_INDEX["lighting=natural"]]
    tech = _to_pct(raw_tech)
    aes = _to_pct(raw_aes)

    # 가짜 임베딩 = 태그 신호(장면 비중이 크다) + **클러스터 공통 성분** + 노이즈.
    # 클러스터 성분이 핵심이다: 실제 갤러리는 연사라 같은 클러스터끼리 코사인이 0.9에 가깝고,
    # 같은 장면 다른 클러스터는 0.3~0.4다. 이 구조가 없으면 MMR을 실습해도 아무 일도 안 난다.
    proj = rng.normal(size=(D_TAG, emb_dim))
    axis_gain = np.ones(D_TAG)
    for v in AXES["scene"]:
        axis_gain[FEATURE_INDEX[f"scene={v}"]] = 3.74
    emb = (X * axis_gain) @ proj + 4.47 * rng.normal(size=(cid, emb_dim))[cluster]
    emb += 1.41 * rng.normal(size=(n, emb_dim))
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)  # 코사인 유사도용 정규화

    # ── 여기까지가 '사진의 실제 내용'. 아래에서 VLM이 본 것(관측)으로 갈라진다 ──
    # 효용·사진학 점수·임베딩은 실제 사진에서 나오고, 태그만 틀린다. 그래서 오류를 켜면
    # "사람은 A를 보고 골랐는데 모델은 B라고 적힌 피처로 배우는" 구조가 된다.
    if tag_error <= 0:
        return Gallery(tags=tags, X=X, tech=tech, aes=aes, cluster=cluster, emb=emb)

    obs = _apply_tag_error(tags, rng, TAG_ERROR_2026_08_25, scale=tag_error)
    return Gallery(
        tags=obs, X=onehot(obs), tech=tech, aes=aes, cluster=cluster, emb=emb,
        tags_true=tags, X_true=X,
    )


def onehot(tags: list[dict[str, str]]) -> np.ndarray:
    X = np.zeros((len(tags), D_TAG))
    for i, t in enumerate(tags):
        for ax, v in t.items():
            X[i, FEATURE_INDEX[f"{ax}={v}"]] = 1.0
    return X


def tag_error_rate(g: Gallery) -> dict[str, float]:
    """관측 태그가 정답과 다른 비율 (축별). 오류를 끈 갤러리면 전부 0."""
    if g.tags_true is None:
        return {ax: 0.0 for ax in AXES}
    n = len(g)
    return {
        ax: sum(1 for a, b in zip(g.tags, g.tags_true) if a[ax] != b[ax]) / n
        for ax in AXES
    }


def _to_pct(raw: np.ndarray) -> np.ndarray:
    """갤러리 내 백분위(0~100). 설계안의 technical_pct·aesthetic_pct와 같은 정의."""
    order = np.argsort(np.argsort(raw))
    return 100.0 * order / max(len(raw) - 1, 1)


# ── 가상의 고객 한 명 ────────────────────────────────────────────────────────
def make_person(seed: int = 42, strength: float = 1.0) -> np.ndarray:
    """취향 벡터 w_true (33차원). 우리가 맞혀야 할 정답.

    실제 사람은 모든 태그에 의견이 있지 않다. 몇 개 축만 뚜렷하고 나머지는 무관심 —
    그 구조를 반영해 일부 축만 큰 값을 준다. (설계안의 conf(axis)가 노리는 상황)
    """
    rng = np.random.default_rng(seed)
    w = np.zeros(D_TAG)
    opinionated = rng.choice(list(AXES.keys()), size=2, replace=False)
    for ax, vals in AXES.items():
        scale = 1.0 if ax in opinionated else 0.15
        for v in vals:
            w[FEATURE_INDEX[f"{ax}={v}"]] = rng.normal(scale=scale)
    return strength * w


def utility(g: Gallery, w: np.ndarray) -> np.ndarray:
    """u(p) = w · x(p). 그 사람이 각 사진에 느끼는 효용(관측 불가, 정답용).

    **정답 태그를 쓴다.** 사람은 사진을 보고 고르지, VLM이 적어 준 태그를 보고 고르지 않는다.
    태그 오류를 켜면 이 한 줄이 "우리가 배우는 피처 ≠ 사람이 반응한 대상"을 만든다.
    """
    return g.Xt @ w


# ── 쌍 생성 ─────────────────────────────────────────────────────────────────
def _axes_differing(a: dict[str, str], b: dict[str, str]) -> list[str]:
    return [ax for ax in AXES if a[ax] != b[ax]]


def design_pair_pool(g: Gallery, pct_gap: float = 15.0) -> list[tuple[int, int]]:
    """설계안 §3-B의 온보딩 쌍 규칙을 만족하는 **모든** 후보 쌍.

    같은 scene · **한 축만 다름** · technical/aesthetic 백분위 차이 ≤ pct_gap ·
    서로 다른 클러스터. "둘 다 괜찮은데 어느 스타일이 좋으세요"가 되도록 만든 규칙.

    규칙이 셀수록 후보가 급감한다 — 400장에서 100여 쌍뿐인 것도 실험 설계의 제약이다.
    """
    n = len(g)
    cand: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if g.scene(i) != g.scene(j):
                continue
            if g.cluster[i] == g.cluster[j]:
                continue
            diff = _axes_differing(g.tags[i], g.tags[j])
            if len(diff) != 1:
                continue
            if abs(g.tech[i] - g.tech[j]) > pct_gap or abs(g.aes[i] - g.aes[j]) > pct_gap:
                continue
            cand.append((i, j))
    return cand


def sample_pairs(
    pool: list[tuple[int, int]], n_pairs: int, seed: int = 0
) -> list[tuple[int, int]]:
    """후보 풀에서 중복 없이 n_pairs개 뽑는다."""
    if not pool:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pool), size=min(n_pairs, len(pool)), replace=False)
    return [pool[k] for k in idx]


def split_pool(
    pool: list[tuple[int, int]], test_frac: float = 0.5, seed: int = 0
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """학습용/홀드아웃용 쌍 풀을 **겹치지 않게** 나눈다.

    같은 쌍이 학습과 평가 양쪽에 들어가면 정확도가 부풀려진다(누수).
    실험을 돌리기 전에 반드시 여기서 갈라 놓는다.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pool))
    n_te = int(len(pool) * test_frac)
    te = [pool[k] for k in idx[:n_te]]
    tr = [pool[k] for k in idx[n_te:]]
    return tr, te


def make_pairs_design(
    g: Gallery, n_pairs: int, seed: int = 0, pct_gap: float = 15.0
) -> list[tuple[int, int]]:
    """후보 풀 생성 + 샘플링을 한 번에 (간단한 실습용 편의 함수)."""
    return sample_pairs(design_pair_pool(g, pct_gap), n_pairs, seed)


def make_pairs_random(g: Gallery, n_pairs: int, seed: int = 0) -> list[tuple[int, int]]:
    """비교군: 아무 두 장이나. 설계 규칙의 가치를 재보기 위한 대조군."""
    rng = np.random.default_rng(seed)
    n = len(g)
    out = set()
    while len(out) < n_pairs:
        i, j = rng.integers(0, n, size=2)
        if i != j:
            out.add((int(min(i, j)), int(max(i, j))))
    return sorted(out)


def answer_pairs(
    g: Gallery,
    w_true: np.ndarray,
    pairs: list[tuple[int, int]],
    temp: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """사람이 쌍을 보고 고른 결과를 시뮬레이션한다.

    결정론적으로 '효용 큰 쪽'을 고르게 하지 않는다. 실제 사람은 흔들린다 —
    Bradley-Terry가 가정하는 그 확률적 선택(P = σ((u_i − u_j)/temp))으로 뽑는다.
    temp가 클수록 사람이 변덕스럽다 = 라벨 노이즈가 크다.

    반환: (chosen, rejected) — 각각 사진 인덱스 배열
    """
    rng = np.random.default_rng(seed)
    u = utility(g, w_true)
    chosen, rejected = [], []
    for i, j in pairs:
        p_i = 1.0 / (1.0 + np.exp(-(u[i] - u[j]) / temp))
        if rng.random() < p_i:
            chosen.append(i); rejected.append(j)
        else:
            chosen.append(j); rejected.append(i)
    return np.array(chosen), np.array(rejected)


def design_matrix(
    X: np.ndarray, chosen: np.ndarray, rejected: np.ndarray, augment: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """쌍 비교 → 로지스틱 회귀 입력.

    핵심: 피처는 **두 사진 피처의 차이** d = x_chosen − x_rejected, 라벨은 1.
    이대로면 y가 전부 1이라 학습이 안 되므로 부호를 뒤집은 (−d, 0)을 함께 넣는다.
    (절편 없는 로지스틱 회귀에서는 이 증강이 수학적으로 같은 해를 주면서
     sklearn이 두 클래스를 보게 해 준다. step1에서 직접 확인한다.)
    """
    d = X[chosen] - X[rejected]
    if not augment:
        return d, np.ones(len(d))
    return np.vstack([d, -d]), np.concatenate([np.ones(len(d)), np.zeros(len(d))])
