"""② 미학 — LAION Aesthetic Predictor v2 (CLIP ViT-L/14 + MLP, Apache-2.0).

CLIP 임베딩은 서비스에서 장면 태그(0-4)·취향 신호(A-2)와 공유되는 자산이라,
이 러너는 임베딩도 함께 반환할 수 있게 만든다 (스파이크에선 점수만 기록).
MLP 가중치는 improved-aesthetic-predictor repo의 고정 커밋에서 받는다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .common import fetch_weight, load_image

MLP_COMMIT = "fe88a163f4661b4ddabba0751ff645e2e620746e"
MLP_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/"
    f"{MLP_COMMIT}/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"
VERSION = f"laion-aesthetic-v2-{MLP_COMMIT[:8]}"


class _MLP(nn.Module):
    """improved-aesthetic-predictor의 MLP 구조 그대로."""

    def __init__(self, input_size: int = 768) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LaionAestheticRunner:
    name = "laion_aesthetic"
    version = VERSION

    def __init__(self) -> None:
        import open_clip

        self._clip, _, self._preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self._clip.eval()

        mlp_path = fetch_weight(MLP_URL, "laion_aesthetic_v2_l14_linearMSE.pth")
        state = torch.load(mlp_path, map_location="cpu", weights_only=True)
        # 원 repo 체크포인트는 "layers.N.*" 키 형태
        self._mlp = _MLP()
        self._mlp.load_state_dict(state)
        self._mlp.eval()

    @torch.no_grad()
    def embed(self, path: str) -> np.ndarray:
        img = self._preprocess(load_image(path)).unsqueeze(0)
        feat = self._clip.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().numpy()

    @torch.no_grad()
    def score(self, path: str) -> dict[str, float]:
        emb = torch.from_numpy(self.embed(path)).unsqueeze(0)
        score = self._mlp(emb)
        return {"aesthetic_score": float(score.item())}
