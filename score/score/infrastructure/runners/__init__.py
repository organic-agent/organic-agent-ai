"""torch 러너 — ArniqaRunner(기술) · LaionRunner(CLIP + 미학 MLP). Dockerfile 이 빌드 시 이 둘을 한 번 만들어 가중치를 굽는다."""

from score.infrastructure.runners.arniqa import ArniqaRunner
from score.infrastructure.runners.laion import LaionRunner

__all__ = ["ArniqaRunner", "LaionRunner"]
