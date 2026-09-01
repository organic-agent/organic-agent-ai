"""A-3 미학 — LAION Aesthetic Predictor v2 (CLIP ViT-L/14 + MLP, Apache-2.0).

두 층이다: 무거운 CLIP(890MB)이 임베딩을 만들고, 3.7MB MLP가 점수를 낸다.
**임베딩은 클러스터링(A-6)이 함께 쓴다** — 같은 사진을 두 번 CLIP에 넣지 않도록
`embed()`와 `score_from_embedding()`을 갈라 두었다. 설계안의 DINOv3 자리를 지금은 CLIP이
대신한다; embedder가 붙으면 `photos.embedding`을 읽는 쪽으로 바꾼다.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from photoselect.v2.runners.common import fetch_weight, load_image

MLP_COMMIT = "fe88a163f4661b4ddabba0751ff645e2e620746e"
MLP_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/"
    f"{MLP_COMMIT}/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"
VERSION = f"laion-aesthetic-v2-{MLP_COMMIT[:8]}"
EMBED_DIM = 768


class _MLP(nn.Module):
    """improved-aesthetic-predictor의 MLP 구조 그대로."""

    def __init__(self, input_size: int = EMBED_DIM) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024), nn.Dropout(0.2),
            nn.Linear(1024, 128), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.Dropout(0.1),
            nn.Linear(64, 16), nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LaionRunner:
    name = "laion_aesthetic"
    version = VERSION

    def __init__(self) -> None:
        import open_clip

        self._clip, _, self._preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        self._clip.eval()
        state = torch.load(fetch_weight(MLP_URL, "laion_aesthetic_v2_l14_linearMSE.pth"),
                           map_location="cpu", weights_only=True)
        self._mlp = _MLP()
        self._mlp.load_state_dict(state)
        self._mlp.eval()

    @torch.no_grad()
    def embed(self, path: str) -> np.ndarray:
        """L2 정규화된 CLIP 임베딩. 코사인 = 내적."""
        img = self._preprocess(load_image(path)).unsqueeze(0)
        feat = self._clip.encode_image(img)
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.squeeze(0).float().numpy()

    @torch.no_grad()
    def embed_texts(self, prompts: list[str]) -> np.ndarray:
        """(len(prompts), 768) L2 정규화 텍스트 임베딩 — v2 zero-shot 태깅용. 이미지 임베딩과 같은 공간."""
        import open_clip

        tok = open_clip.get_tokenizer(CLIP_MODEL)
        feat = self._clip.encode_text(tok(prompts))
        feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat.float().numpy()

    @torch.no_grad()
    def score_from_embedding(self, emb: np.ndarray) -> float:
        return float(self._mlp(torch.from_numpy(emb).unsqueeze(0)).item())

    def score(self, path: str) -> dict[str, float]:
        return {"aesthetic_score": self.score_from_embedding(self.embed(path))}
