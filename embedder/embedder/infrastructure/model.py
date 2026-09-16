"""DINOv3 장면 임베딩. CLS 토큰을 L2 정규화해 돌려준다.

정규화해 두면 내적이 곧 코사인 유사도라 나중에 인덱스를 바꿔도 값을 다시 만질 필요가 없다.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np
from PIL import Image

from embedder.config.settings import Settings

log = logging.getLogger(__name__)


class DinoEmbedder:
    def __init__(self, model_id: str, model_revision: str, expected_dim: int) -> None:
        # torch import 는 수 초 걸린다. 모델을 안 쓰는 경로가 그 값을 치르지 않게 여기서만 import 한다.
        import torch
        from transformers import AutoImageProcessor, AutoModel

        self._torch = torch
        self.model_id = model_id
        self.model_revision = model_revision
        self.expected_dim = expected_dim

        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        log.info("모델 로드: %s@%s (device=%s)", model_id, model_revision, self.device)
        self.processor = AutoImageProcessor.from_pretrained(model_id, revision=model_revision)
        self.model = AutoModel.from_pretrained(model_id, revision=model_revision).to(self.device).eval()

    def encode(self, images: list[Image.Image]) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            batch = self.processor(images=images, return_tensors="pt").to(self.device)
            cls = self.model(**batch).last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, dim=-1)
            vectors = cls.float().cpu().numpy()

        # DB 컬럼 폭과 어긋나면 UPDATE 에서도 걸리지만, 첫 배치에서 바로 멈추는 편이 낫다.
        if vectors.shape[1] != self.expected_dim:
            raise RuntimeError(
                f"임베딩 차원이 설정과 다릅니다: 모델 {vectors.shape[1]}, 기대값 {self.expected_dim}"
            )
        return vectors


@lru_cache(maxsize=1)
def get_embedder(model_id: str, model_revision: str, expected_dim: int) -> DinoEmbedder:
    """웜 스타트에서 재사용하려고 프로세스당 하나만 만든다."""
    return DinoEmbedder(model_id, model_revision, expected_dim)


def load_from(settings: Settings) -> DinoEmbedder:
    return get_embedder(settings.model_id, settings.model_revision, settings.embed_dim)
