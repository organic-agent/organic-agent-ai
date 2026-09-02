"""피사체 zero-shot 태깅 — 이미 계산하는 CLIP 임베딩에 텍스트 프롬프트를 대는 것. 추가 비용 ~0.

4단계(선호·균형)의 "유형" 축이다: 신부 단독 / 신랑 단독 / 둘 / 단체. 2026-08-29 갤러리 1 검증에서
확신 라벨 36/36 정답이라 `V2Knobs.subjects_trusted=True` (근거는 config.py 주석).

margin(1위-2위 코사인 차) < 0.01 이면 `unknown` — 검증에서 이 구간의 argmax 는 커플을 신부/신랑 단독으로
오인하는 경우(신랑이 등만 보이는 컷 등)가 84장 중 9장이었다. unknown 은 균형 집계에서 빠진다.
"""

from __future__ import annotations

import numpy as np

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

LABELS_KO = {"bride": "신부 단독", "groom": "신랑 단독", "couple": "두 분", "group": "단체", "unknown": ""}


class SubjectsTagger:
    def __init__(self, laion_runner, min_margin: float = 0.01) -> None:
        self.min_margin = min_margin
        self._labels = list(SUBJECTS)
        # 클래스마다 프롬프트 평균 → 정규화
        vecs = []
        for lab in self._labels:
            t = laion_runner.embed_texts(PROMPTS[lab]).mean(axis=0)
            vecs.append(t / np.linalg.norm(t))
        self._T = np.stack(vecs)   # (4, 768)

    def tag(self, image_emb: np.ndarray) -> tuple[str, float]:
        """(label, margin). image_emb 는 CLIP(정규화) 임베딩 — DINOv3 가 아니다."""
        sims = self._T @ image_emb
        order = np.argsort(-sims)
        margin = float(sims[order[0]] - sims[order[1]])
        if margin < self.min_margin:
            return "unknown", margin
        return self._labels[int(order[0])], margin
