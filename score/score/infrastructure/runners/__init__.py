"""torch 러너 — ArniqaRunner(기술) · LaionRunner(CLIP + 미학 MLP). Dockerfile 이 빌드 시 이 둘을 한 번 만들어 가중치를 굽는다.
얼굴 값 러너(`face.FaceRunner`, #164)는 torch 가 없어 여기서 내보내지 않는다 — `runners.face` 에서 직접 가져온다."""

from score.infrastructure.runners.arniqa import ArniqaRunner
from score.infrastructure.runners.laion import LaionRunner

__all__ = ["ArniqaRunner", "LaionRunner"]
