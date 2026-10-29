"""사진 라벨(wes #348) — 이미 계산하는 CLIP 이미지 벡터에 문장을 대어 라벨 8개마다 0~1 점수를 낸다. 추가 비용 ~0.

셀렉 도우미가 "눈 감은 사진 빼줘"·"베일 쓴 컷만" 같은 조건을 거르는 재료다. score 가 피사체 옆에서 계산해 wes `photo_labels`(V46)에
8개 점수를 전부 쓴다 — 임계값 미달도. "라벨이 붙었다"(점수 ≥ 라벨별 임계값)는 wes 가 정한다.

    점수 = 양·음 문장 확률(2026-10-27 결정). 라벨마다 양(눈 감음)·음(눈 뜸) 문장 묶음의 평균 벡터를 두고, 이미지 벡터와의 코사인 둘에
    CLIP 의 logit 배율(100)을 곱해 softmax 한 양쪽 몫이다. 0.5 가 반반. 코사인 그대로는 문장끼리 0.01 단위로만 갈려(피사체 margin 0.01)
    라벨마다 눈금이 달라지고, 웨딩·인물 같은 공통 성분이 점수를 끌어올린다 — 쌍으로 빼면 그 성분이 상쇄된다.
    구도 셋(클로즈업·상반신·전신)은 서로가 음 문장이라 셋 사이 softmax 다(합 1).

문장은 첫 초안이다 — 정밀도는 200장 검수 실험(S2)에서 재고 고친다. 어휘·문장을 바꾸면 `relabel` 이 저장된 벡터로 다시 계산한다
(이미지를 다시 보지 않는다). 라벨 코드는 `domain/labels.py`.
"""

from __future__ import annotations

import numpy as np

from score.domain.labels import LABELS

#: CLIP ViT-L/14(OpenAI) 의 학습된 logit 배율 exp(logit_scale).
LOGIT_SCALE = 100.0

#: 양·음 문장 쌍으로 재는 라벨 — (양, 음).
PAIRS: dict[str, tuple[list[str], list[str]]] = {
    "EYES_CLOSED": (["a photo of a person with their eyes closed",
                     "a wedding portrait where the bride has her eyes closed",
                     "a close-up of a face with closed eyes"],
                    ["a photo of a person with their eyes open",
                     "a wedding portrait where the bride has her eyes open",
                     "a close-up of a face with open eyes"]),
    "SMILE": (["a photo of a smiling person",
               "a bride and groom laughing happily",
               "a close-up of a big smile"],
              ["a photo of a person with a neutral expression",
               "a bride and groom with calm serious faces",
               "a close-up of a face that is not smiling"]),
    "VEIL": (["a bride wearing a wedding veil",
              "a long white veil over the bride's hair"],
             ["a bride without a veil",
              "a bride with her hair uncovered, no veil"]),
    "BOUQUET": (["a bride holding a bouquet of flowers",
                 "a wedding bouquet held in hands"],
                ["a bride with empty hands, holding no flowers",
                 "a wedding photo with no bouquet"]),
    "BACK_VIEW": (["a photo of a person seen from behind with their back to the camera",
                   "the back of a bride walking away from the camera"],
                  ["a photo of a person facing the camera",
                   "a front view of a bride looking toward the camera"]),
}

#: 구도 — 셋 중 하나라 서로가 음 문장이다.
FRAMING: dict[str, list[str]] = {
    "CLOSE_UP": ["a close-up photo of a face",
                 "a tight portrait showing only the head and shoulders"],
    "HALF_BODY": ["a medium shot showing a person from the waist up",
                  "a half-body wedding portrait"],
    "FULL_BODY": ["a full-length photo showing a person from head to toe",
                  "a full-body wedding photo with the whole dress visible"],
}


def _mean_unit(laion_runner, prompts: list[str]) -> np.ndarray:
    t = laion_runner.embed_texts(prompts).mean(axis=0)
    return t / np.linalg.norm(t)


class LabelTagger:
    """문장 벡터를 한 번 만들어 두고, 이미지 벡터의 라벨 점수를 돌려준다. score 는 한 장씩, relabel 은 갤러리를 한 번에."""

    def __init__(self, laion_runner) -> None:
        self._pair_labels = list(PAIRS)
        self._pos = np.stack([_mean_unit(laion_runner, PAIRS[lab][0]) for lab in self._pair_labels])
        self._neg = np.stack([_mean_unit(laion_runner, PAIRS[lab][1]) for lab in self._pair_labels])
        self._framing_labels = list(FRAMING)
        self._framing = np.stack([_mean_unit(laion_runner, FRAMING[lab]) for lab in self._framing_labels])

    def score_many(self, image_embs: np.ndarray) -> list[dict[str, float]]:
        """(n, d) CLIP(정규화) 벡터 → 사진마다 {라벨: 0~1}. float64 로 계산한다 — 배율 100 을 곱한 지수가 float32 를 넘친다."""
        E = np.asarray(image_embs, dtype=np.float64)
        if E.ndim == 1:
            E = E[None, :]
        # 양·음 둘의 softmax 중 양쪽 몫 = sigmoid(배율 × (양 코사인 − 음 코사인))
        pair = 1.0 / (1.0 + np.exp(-LOGIT_SCALE * (E @ self._pos.T - E @ self._neg.T)))
        logits = LOGIT_SCALE * (E @ self._framing.T)
        framing = np.exp(logits - logits.max(axis=1, keepdims=True))
        framing /= framing.sum(axis=1, keepdims=True)
        out = []
        for i in range(E.shape[0]):
            row = {lab: pair[i, j] for j, lab in enumerate(self._pair_labels)}
            row.update({lab: framing[i, j] for j, lab in enumerate(self._framing_labels)})
            out.append({lab: round(float(min(1.0, max(0.0, row[lab]))), 4) for lab in LABELS})
        return out

    def score(self, image_emb: np.ndarray) -> dict[str, float]:
        return self.score_many(image_emb)[0]
