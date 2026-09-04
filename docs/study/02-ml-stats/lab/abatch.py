"""A 배치(전수 분석) 시뮬레이터 — step7~9의 공용 기반.

────────────────────────────────────────────────────────────────────────────
step1~6은 소재가 전부 01의 쌍 비교였다. step7~9는 **다른 워크로드**를 다룬다 —
`plan.md` §3-A의 전수 분석(A), 즉 갤러리당 1회 도는 배치다.

    A-1  얼굴·눈·미소     MediaPipe          → eyes_open, smile      ← step7
    A-2  기술 품질        ARNIQA             → technical_pct         ← step9
    A-3  미학            LAION Aesthetic     → aesthetic_pct         ← step9
    A-4  고정 축 태그     오픈 VLM            → scene/framing/…       ← step8
    A-7  클러스터 대표    A-1·A-2 결정적 합성  → cluster_rank          ← step9

여기서 나오는 문제는 추천이 아니라 **분류**다. 그래서 지표도 다르다 —
혼동행렬, 정밀도/재현율, PR·ROC 곡선, 확률 보정. 01에서 한 번도 안 나온 것들이다.

이 모듈은 두 가지를 만든다.
  ① eyes_open 점수 — 진짜 라벨(눈감김)과 **얼굴 크기**에 의존하는 잡음을 함께
  ② VLM 태그 혼동 — `gallery.py`의 실측 오차 모델을 그대로 (01과 공유)

**얼굴 크기 의존성이 핵심이다.** 2026-08-25 스파이크 리포트의 미해결 관찰:

    작은 얼굴의 blink 신호 신뢰도: 2인 컷(얼굴 bbox 비율 ~0.036)에서
    eyes_open 최소값이 0.2대로 낮게 나온다 — 실제 눈 감김인지
    blendshape가 작은 얼굴에서 불안정한 건지 확인 필요.

step7 [E]가 그 관찰을 재현하고, 무엇을 재야 답이 나오는지 보여 준다.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pairsim as P              # ★ 먼저 import (01-recsys/lab을 경로에 올린다)

import gallery as G              # noqa: E402

# 구도·인물 구성 → 화면에서 얼굴이 차지하는 비율. 스파이크 리포트의 실측 대역에 맞췄다.
FACE_RATIO_BY_FRAMING = {"closeup": 0.30, "half": 0.14, "full": 0.06, "wide": 0.025}
FACE_RATIO_BY_SUBJECTS = {"bride": 1.0, "groom": 1.0, "couple": 0.75,
                          "family": 0.45, "friends": 0.35, "none": 1.0}


@dataclass
class EyeSignal:
    """A-1이 사진마다 내놓는 것."""

    score: np.ndarray        # (N,) eyes_open 점수 0~1. 클수록 '눈을 떴다'
    closed: np.ndarray       # (N,) bool. **진짜** 눈감김 여부 (관측 불가, 정답용)
    face_ratio: np.ndarray   # (N,) 얼굴 bbox 비율. 작을수록 신호가 불안정하다
    has_face: np.ndarray     # (N,) bool. detail 컷 등 얼굴이 없는 사진

    def valid(self) -> np.ndarray:
        return self.has_face


def face_ratios(g: G.Gallery, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """구도·인물 구성에서 얼굴 크기를 만든다. 정답 태그를 쓴다 — 사진의 실제 내용이므로."""
    tags = g.tags_true if g.tags_true is not None else g.tags
    ratio = np.zeros(len(g))
    has_face = np.ones(len(g), dtype=bool)
    for i, t in enumerate(tags):
        if t["subjects"] == "none":                 # detail 컷 — 얼굴이 없다
            has_face[i] = False
            continue
        base = FACE_RATIO_BY_FRAMING[t["framing"]] * FACE_RATIO_BY_SUBJECTS[t["subjects"]]
        ratio[i] = float(np.clip(base * rng.lognormal(0.0, 0.25), 0.005, 0.6))
    return ratio, has_face


def eye_signal(g: G.Gallery, seed: int = 0) -> EyeSignal:
    """A-1의 eyes_open 점수를 흉내 낸다.

    모형: 잠재 로짓 z = (뜬 눈이면 +2.2, 감은 눈이면 −2.2) + 잡음,  score = σ(z)
          **잡음의 크기가 얼굴 비율에 반비례한다** — 작은 얼굴일수록 blendshape가 흔들린다.

    이 한 줄이 스파이크 리포트의 관찰을 만든다. 그리고 step7 [E]에서 보듯,
    '작은 얼굴에서 점수가 낮다'와 '작은 얼굴에서 점수를 믿을 수 없다'는 **다른 말**이다.
    """
    rng = np.random.default_rng(seed)
    ratio, has_face = face_ratios(g, rng)
    tags = g.tags_true if g.tags_true is not None else g.tags
    closed = np.array([t["expression"] == "eyes_closed" for t in tags])

    # 얼굴이 클수록 잡음이 작다. 0.30(클로즈업)에서 σ≈0.9, 0.03(단체컷)에서 σ≈2.9
    sigma = 0.5 + 0.4 / np.sqrt(np.maximum(ratio, 0.005))
    z = np.where(closed, -2.2, 2.2) + rng.normal(0.0, sigma)
    score = 1.0 / (1.0 + np.exp(-z))
    score[~has_face] = np.nan                       # 얼굴이 없으면 점수 자체가 없다
    return EyeSignal(score=score, closed=closed, face_ratio=ratio, has_face=has_face)


def confusion_matrix(true: list[str], pred: list[str], labels: list[str]) -> np.ndarray:
    """행=정답, 열=예측. sklearn 없이 직접 짠다 — 무엇을 세는지 보이게."""
    idx = {l: i for i, l in enumerate(labels)}
    M = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true, pred):
        M[idx[t], idx[p]] += 1
    return M


def per_class(M: np.ndarray) -> dict[str, np.ndarray]:
    """혼동행렬에서 클래스별 정밀도·재현율·F1·지원 수."""
    tp = np.diag(M).astype(float)
    support = M.sum(axis=1).astype(float)          # 정답이 그 클래스인 사진 수
    predicted = M.sum(axis=0).astype(float)        # 그 클래스로 예측된 사진 수
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), np.nan)
    return {"precision": precision, "recall": recall, "f1": f1, "support": support}


def tagged_gallery(n: int = 1000, seed: int = 0, tag_error: float = 1.0,
                   scene_profile: str = "balanced") -> G.Gallery:
    """관측 태그와 정답 태그를 둘 다 가진 갤러리. step8의 혼동행렬 재료."""
    return G.make_gallery(n, seed=seed, tag_error=tag_error, scene_profile=scene_profile)
