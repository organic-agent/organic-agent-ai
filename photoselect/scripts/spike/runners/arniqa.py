"""① 기술 품질 — ARNIQA (WACV 2024, Apache-2.0).

torch.hub로 고정 ref에서 로드 (스파이크 전용 — 채택 시 원본 repo 가중치를
빌드 시 번들). repo 데모(single_image_inference)와 동일하게 전체 이미지와
half-scale 이미지 두 입력을 준다.
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF

from .common import load_image

HUB_REPO = "miccunifi/ARNIQA:main"  # 스파이크 후 커밋 SHA로 고정
REGRESSOR_DATASET = "kadid10k"
VERSION = f"arniqa-{REGRESSOR_DATASET}"

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class ArniqaRunner:
    name = "arniqa"
    version = VERSION

    def __init__(self) -> None:
        self._model = torch.hub.load(
            repo_or_dir=HUB_REPO,
            source="github",
            model="ARNIQA",
            regressor_dataset=REGRESSOR_DATASET,
        )
        self._model.eval()

    @torch.no_grad()
    def score(self, path: str) -> dict[str, float]:
        img = load_image(path)
        x = TF.normalize(TF.to_tensor(img), _MEAN, _STD).unsqueeze(0)
        x_ds = torch.nn.functional.interpolate(
            x, scale_factor=0.5, mode="bilinear", align_corners=False
        )
        score = self._model(x, x_ds, return_embedding=False, scale_score=True)
        return {"technical_score": float(score.item())}
