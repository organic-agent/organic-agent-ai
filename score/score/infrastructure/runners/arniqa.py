"""A-2 기술 품질 — ARNIQA (WACV 2024, Apache-2.0). 무참조(no-reference) 화질 평가.

torch.hub 고정 ref(커밋 SHA)에서 로드한다. 가중치는 Dockerfile 이 빌드 시 hub 캐시에 굽는다(#35).
repo 데모와 같이 전체 이미지(`long_edge` 로 줄인 것)와 half-scale 두 입력을 준다.

`REGRESSOR_DATASET`: 회귀기를 학습한 데이터셋. **spaq**(실사 스마트폰 사진 왜곡)으로 확정 —
2026-08-25 류지혜(2) 60장 비교(`scripts/arniqa_regressors.py`):
  · hub 기본값 kadid10k(합성 왜곡)는 실사 회귀기 셋과 순위 상관 0.16~0.34 — 다른 것을 잰다
  · koniq10k·spaq·clive는 서로 0.88~0.90으로 일치
  · spaq: 흐림 반응/반전 잡음 23.7배(최고), LAION 미학과 상관 0.10(가장 독립), IQR 0.122
  · koniq10k는 대역이 좁아(IQR 0.051) 백분위가 잡음이 되기 쉽다
"""

from __future__ import annotations

import numpy as np
import torch

from PIL import Image

from score.infrastructure.device import autocast, pick_device
from score.infrastructure.images import as_image

#: 커밋 SHA 고정 — Dockerfile 이 빌드 시 같은 ref 로 hub 캐시(TORCH_HOME)를 채운다. NAT 없는 Lambda 는 런타임에 받을 수 없다.
HUB_REPO = "miccunifi/ARNIQA:66d16eb0ff1e1655872d32c0c233614a3922aaad"
REGRESSOR_DATASET = "spaq"

#: 입력 긴 변. 1600(미리보기 그대로)은 ResNet-50 에 224px 대비 34배 픽셀 + half-scale 한 번 더라 장당 연산의
#: 대부분이었다(#51). 1024 로 낮추면 연산 ~2.4배 감소. 순위 상관·흐림 반응은 #51 리포트.
DEFAULT_LONG_EDGE = 1024

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


class ArniqaRunner:
    """`device`·`fp16`(#68): ResNet-50 인코더와 회귀기가 함께 GPU 로 간다. `score_batch` 는 같은 픽셀 크기끼리만 묶는다 —
    ARNIQA 는 전체 해상도를 그대로 넣는 모델이라 패딩·리사이즈로 맞추면 점수가 바뀐다."""

    def __init__(self, regressor_dataset: str = REGRESSOR_DATASET, long_edge: int = DEFAULT_LONG_EDGE,
                 device: str | None = None, fp16: bool = False) -> None:
        self.long_edge = long_edge
        self.device = pick_device(device)
        self.fp16 = fp16
        self._model = torch.hub.load(
            repo_or_dir=HUB_REPO, source="github", model="ARNIQA",
            regressor_dataset=regressor_dataset,
            skip_validation=True,   # 캐시 히트 뒤 GitHub API 를 건드리지 않게 — Lambda 에는 인터넷이 없다
        )
        self._model.eval().to(self.device)
        self._mean = torch.tensor(_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_STD, device=self.device).view(1, 3, 1, 1)

    def prepare(self, source: str | Image.Image) -> torch.Tensor:
        """CPU 전처리 → (3, H, W) uint8. float 변환·정규화는 GPU 에서 한다 — 1024px 한 장이 uint8 2MB, float 는 8MB 라
        전송도 CPU 연산도 4분의 1 이다(#68). CPU 경로에서는 같은 값을 CPU 에서 계산한다(수치 동일)."""
        arr = np.array(as_image(source, self.long_edge))              # (H, W, 3) uint8 — 복사본(쓰기 가능)이라야 torch 가 경고 없이 받는다
        return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

    @torch.no_grad()
    def score(self, source: str | Image.Image) -> dict[str, float]:
        return self.score_batch([source])[0]

    @torch.no_grad()
    def score_batch(self, sources: list[str | Image.Image]) -> list[dict[str, float]]:
        return self.score_prepared([self.prepare(s) for s in sources])

    @torch.no_grad()
    def score_prepared(self, tensors: list[torch.Tensor]) -> list[dict[str, float]]:
        """`prepare()` 결과들. 순서 유지. 크기가 같은 것끼리 한 forward. 빈 목록이면 []."""
        out: list[dict[str, float] | None] = [None] * len(tensors)
        groups: dict[tuple[int, ...], list[int]] = {}
        for i, t in enumerate(tensors):
            groups.setdefault(tuple(t.shape), []).append(i)
        for idx in groups.values():
            x = torch.stack([tensors[i] for i in idx]).to(self.device, non_blocking=True)
            x = (x.float().div_(255) - self._mean) / self._std
            x_ds = torch.nn.functional.interpolate(x, scale_factor=0.5, mode="bilinear", align_corners=False)
            with autocast(self.device, self.fp16):
                s = self._model(x, x_ds, return_embedding=False, scale_score=True)
            s = s.float().reshape(-1).cpu()
            for i, v in zip(idx, s.tolist()):
                out[i] = {"technical_score": float(v)}
        return out  # type: ignore[return-value]
