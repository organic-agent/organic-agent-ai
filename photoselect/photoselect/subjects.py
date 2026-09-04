"""CLIP zero-shot 라벨 — 이미 계산하는 CLIP 이미지 임베딩에 텍스트 프롬프트를 대는 것. 추가 비용 ~0.

두 라벨러가 같은 구조다(프롬프트 평균 → 정규화 → 코사인 argmax). SCORE 잡이 사진마다 계산해
`photo_analysis`에 저장하고, CATEGORIZE 잡은 저장된 라벨만 읽는다 — 그래서 카테고리 쪽은 CLIP
텍스트 인코더(torch)가 필요 없다(#26).

    SubjectsTagger  피사체 유형 — 신부 단독 / 신랑 단독 / 둘 / 단체. 2026-08-29 갤러리 1 검증에서
                    확신 라벨 36/36 정답. margin(1위-2위 코사인 차) < 0.01 이면 `unknown` —
                    이 구간의 argmax 는 커플을 신부/신랑 단독으로 오인하는 경우(신랑이 등만 보이는 컷 등)가
                    84장 중 9장이었다. wes 가 세부폴더 칩(BRIDE/GROOM/COUPLE/GROUP)의 다수결에 쓴다.
    ParentTagger    부모(큰 분류) 고정 목록 — 검증 전용. 822장 실측 일치 79%라 판정에는 못 쓰고,
                    naming 이 그룹 다수결을 VLM 부모와 비교해 needs_review 를 켠다. '기타'는 후보에 없다.
"""

from __future__ import annotations

import numpy as np

from photoselect.config import PARENT_PROMPTS, PARENTS

SUBJECTS = ("bride", "groom", "couple", "group")

PROMPTS: dict[str, list[str]] = {
    "bride": ["a photo of the bride alone in a white wedding dress",
              "a portrait of a woman in a wedding dress, no one else"],
    "groom": ["a photo of the groom alone in a suit",
              "a portrait of a man in a tuxedo, no one else"],
    "couple": ["a photo of the bride and groom together",
               "a wedding couple, a woman in a wedding dress and a man in a suit"],
    "group": ["a group photo of many people at a wedding",
              "the bride and groom with family and friends"],
}

class _ZeroShot:
    """라벨별 프롬프트 평균 벡터를 만들어 두고, 이미지 벡터와의 코사인 순위를 돌려준다."""

    def __init__(self, laion_runner, prompts: dict[str, list[str]]) -> None:
        self._labels = list(prompts)
        vecs = []
        for lab in self._labels:
            t = laion_runner.embed_texts(prompts[lab]).mean(axis=0)
            vecs.append(t / np.linalg.norm(t))
        self._T = np.stack(vecs) if vecs else np.zeros((0, 768))

    def ranked(self, image_emb: np.ndarray) -> tuple[list[str], np.ndarray]:
        """(라벨, 코사인) 내림차순. image_emb 는 CLIP(정규화) 임베딩 — DINOv3 가 아니다."""
        sims = self._T @ image_emb
        order = np.argsort(-sims)
        return [self._labels[int(i)] for i in order], sims[order]


class SubjectsTagger(_ZeroShot):
    def __init__(self, laion_runner, min_margin: float = 0.01) -> None:
        super().__init__(laion_runner, PROMPTS)
        self.min_margin = min_margin

    def tag(self, image_emb: np.ndarray) -> tuple[str, float]:
        """(label, margin). margin 이 작으면 unknown — 균형 집계에서 빠진다."""
        labels, sims = self.ranked(image_emb)
        margin = float(sims[0] - sims[1])
        if margin < self.min_margin:
            return "unknown", margin
        return labels[0], margin


class ParentTagger(_ZeroShot):
    def __init__(self, laion_runner) -> None:
        super().__init__(laion_runner, {p: PARENT_PROMPTS[p] for p in PARENTS if p in PARENT_PROMPTS})

    def tag(self, image_emb: np.ndarray) -> str | None:
        """사진 한 장의 부모 argmax. 후보가 없으면 None."""
        if not len(self._T):
            return None
        labels, _ = self.ranked(image_emb)
        return labels[0]


def majority(labels: list[str | None]) -> str | None:
    """저장된 사진별 라벨의 그룹 다수결. None 은 표에서 뺀다. 표가 없으면 None."""
    votes = [lab for lab in labels if lab]
    if not votes:
        return None
    counts: dict[str, int] = {}
    for lab in votes:
        counts[lab] = counts.get(lab, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -votes.index(kv[0])))[0]
