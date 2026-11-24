"""얼굴 값 측정기 — 얼굴 크롭에서 눈 감김 · 웃음을 바로 맞힌다(#170). `face_source` = lora 일 때 MediaPipe 대신 쓴다.

    얼굴 크롭 256 → 가운데 224 → DINOv3 ViT-B/16 + LoRA(q · v 투영, 랭크 8) → [CLS ⊕ 패치 평균](1,536) → 선형 → 시그모이드 2개
      [눈 감김(두 눈 중 큰 쪽, 0~1), 웃음(입꼬리 평균, 0~1)] — MediaPipe 블렌드셰이프와 같은 뜻 · 같은 눈금

학습 정답은 두 가지다: 눈 감김과 정면의 웃음은 MediaPipe 값, 정면이 아닌 얼굴의 웃음은 VLM 판정(웃지 않음 0 · 미소 0.65 · 크게 웃음 0.9).
MediaPipe 는 옆얼굴에서 값을 못 내거나 웃음을 낮게 낸다 — 그러면 웃느라 감은 눈이 결함으로 읽힌다. 입력을 일부러 망가뜨려(작게 · 흐리게 · 가려서) 학습했다.
학습에 안 쓴 갤러리에서 옆얼굴의 웃는 얼굴을 웃는다고 한 비율이 13%(MediaPipe) → 77% 이고, 폴더 추천은 같은 수준이다.
한계: 깨끗한 정면 얼굴에서는 MediaPipe 값과 0.10 쯤 다르다. 얼굴이 아닌 크롭에도 숫자를 낸다. 확인한 갤러리가 하나다.
학습 · 평가: docs/experiments/reco-rubric-2026-11 (k/train_v3.py · k-face-judgement-baseline).

가중치는 두 곳에서 온다:
    백본   Hugging Face `facebook/dinov3-vitb16-pretrain-lvd1689m` (게이트 모델 — GPU 이미지가 빌드 때 받아 굽는다. 임베더와 같은 리비전)
    LoRA   `score/assets/face_values_lora.pt` (1.2MB, 저장소에 둔다) — {"version", "rank", "alpha", "lora", "head"}
`peft` 는 쓰지 않는다. LoRA 는 W += (alpha / rank) · B·A 로 백본 가중치에 한 번 더해 두면 추론은 보통의 forward 다.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image

from score.infrastructure.device import autocast, pick_device

log = logging.getLogger(__name__)

MODEL_ID = "facebook/dinov3-vitb16-pretrain-lvd1689m"
#: 임베더(`embedder/config/settings.py`)와 같은 커밋 — main 이 바뀌어도 같은 가중치를 받는다.
MODEL_REVISION = "5931719e67bbdb9737e363e781fb0c67687896bc"
WEIGHTS = Path(__file__).resolve().parents[2] / "assets" / "face_values_lora.pt"
INPUT_SIDE = 224
BATCH = 64
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#: DINOv3 출력에서 CLS 1개 + 레지스터 토큰 4개 뒤가 패치다.
_PATCH_START = 5


def merge_lora(state: dict, lora: dict, scale: float) -> int:
    """`state`(백본 state_dict)에 LoRA 를 더한다 — 제자리에서. 더한 층 수를 돌려준다.

    LoRA 키는 `scripts/pack_face_lora.py` 가 줄인 모양이다: `layer.3.attention.q_proj.lora_A` ↔ 백본의 `layer.3.attention.q_proj.weight`."""
    merged = 0
    for name, a in lora.items():
        if not name.endswith(".lora_A"):
            continue
        b = lora[name.removesuffix(".lora_A") + ".lora_B"]
        target = name.removesuffix(".lora_A") + ".weight"
        state[target] += scale * (b.to(state[target].dtype) @ a.to(state[target].dtype))
        merged += 1
    return merged


class FaceValueRunner:
    def __init__(self, device: str | None = None, fp16: bool = False, weights: Path = WEIGHTS) -> None:
        import torch
        from transformers import AutoModel

        self.device, self.fp16 = pick_device(device), fp16
        saved = torch.load(weights, map_location="cpu", weights_only=True)
        model = AutoModel.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
        state = model.state_dict()
        merged = merge_lora(state, saved["lora"], saved["alpha"] / saved["rank"])
        if merged != 2 * model.config.num_hidden_layers:
            raise RuntimeError(f"LoRA 가 붙은 층이 {merged}개다 — q · v 투영 {2 * model.config.num_hidden_layers}개여야 한다")
        model.load_state_dict(state)
        self.head = torch.nn.Linear(2 * model.config.hidden_size, 2)
        self.head.load_state_dict(saved["head"])
        self.model = model.to(self.device).eval()
        self.head = self.head.to(self.device).eval()
        self.version = saved["version"]
        log.info("[score] 얼굴 값 측정기(LoRA %s) 로드 · device=%s", self.version, self.device)

    @staticmethod
    def prepare(crop: Image.Image) -> np.ndarray:
        """256 크롭 → 가운데 224 의 정규화된 CHW float32. 학습 때 검증과 같은 자르기다."""
        off = (crop.width - INPUT_SIDE) // 2
        a = np.asarray(crop.convert("RGB"), dtype=np.uint8)[off:off + INPUT_SIDE, off:off + INPUT_SIDE]
        return ((a.astype(np.float32) / 255 - _MEAN) / _STD).transpose(2, 0, 1)

    def predict(self, crops: list[Image.Image]) -> np.ndarray:
        """크롭마다 [눈 감김, 웃음] — (N, 2) float32, 0~1."""
        import torch

        out = []
        with torch.no_grad():
            for start in range(0, len(crops), BATCH):
                x = torch.from_numpy(np.stack([self.prepare(c) for c in crops[start:start + BATCH]])).to(self.device)
                with autocast(self.device, self.fp16):
                    o = self.model(pixel_values=x)
                    feats = torch.cat([o.pooler_output, o.last_hidden_state[:, _PATCH_START:].mean(1)], dim=1)
                    out.append(torch.sigmoid(self.head(feats.float())).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 2), dtype=np.float32)
