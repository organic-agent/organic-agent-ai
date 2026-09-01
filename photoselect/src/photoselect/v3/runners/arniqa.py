"""A-2 기술 품질 — ARNIQA (WACV 2024, Apache-2.0). 무참조(no-reference) 화질 평가.

torch.hub 고정 ref에서 로드한다. 채택 시 커밋 SHA 고정 + 가중치 번들(tech-stack.md).
repo 데모와 같이 전체 이미지와 half-scale 두 입력을 준다.

`REGRESSOR_DATASET`: 회귀기를 학습한 데이터셋. **spaq**(실사 스마트폰 사진 왜곡)으로 확정 —
2026-08-25 류지혜(2) 60장 비교(`scripts/arniqa_regressors.py`):
  · hub 기본값 kadid10k(합성 왜곡)는 실사 회귀기 셋과 순위 상관 0.16~0.34 — 다른 것을 잰다
  · koniq10k·spaq·clive는 서로 0.88~0.90으로 일치
  · spaq: 흐림 반응/반전 잡음 23.7배(최고), LAION 미학과 상관 0.10(가장 독립), IQR 0.122
  · koniq10k는 대역이 좁아(IQR 0.051) 백분위가 잡음이 되기 쉽다
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF

from photoselect.v3.runners.common import load_image

HUB_REPO = "miccunifi/ARNIQA:main"  # TODO 채택 시 커밋 SHA로 고정
REGRESSOR_DATASET = "spaq"
VERSION = f"arniqa-{REGRESSOR_DATASET}"

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class ArniqaRunner:
    name = "arniqa"
    version = VERSION

    def __init__(self, regressor_dataset: str = REGRESSOR_DATASET) -> None:
        self.version = f"arniqa-{regressor_dataset}"
        self._model = torch.hub.load(
            repo_or_dir=HUB_REPO, source="github", model="ARNIQA",
            regressor_dataset=regressor_dataset,
        )
        self._model.eval()

    @torch.no_grad()
    def score(self, path: str) -> dict[str, float]:
        img = load_image(path)
        x = TF.normalize(TF.to_tensor(img), _MEAN, _STD).unsqueeze(0)
        x_ds = torch.nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
        s = self._model(x, x_ds, return_embedding=False, scale_score=True)
        return {"technical_score": float(s.item())}
