"""A-2 기술 품질 — ARNIQA (WACV 2024, Apache-2.0). 무참조(no-reference) 화질 평가.

torch.hub 고정 ref에서 로드한다. 채택 시 커밋 SHA 고정 + 가중치 번들(tech-stack.md).
repo 데모와 같이 전체 이미지와 half-scale 두 입력을 준다.

`REGRESSOR_DATASET`: 회귀기를 학습한 데이터셋. hub 기본값 `kadid10k`는 **합성 왜곡**
데이터셋이다. 우리 사진은 실사이므로 `koniq10k`·`spaq`(실사 왜곡)가 맞을 수 있다 —
같은 갤러리에 여러 회귀기를 돌려 변별력을 비교하는 것이 남은 작업이다(study/00 catalog).
"""

from __future__ import annotations

import torch
import torchvision.transforms.functional as TF

from photoselect.analyze.runners.common import load_image

HUB_REPO = "miccunifi/ARNIQA:main"  # TODO 채택 시 커밋 SHA로 고정
REGRESSOR_DATASET = "kadid10k"
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
