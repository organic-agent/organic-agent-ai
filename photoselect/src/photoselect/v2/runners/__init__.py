"""v2 러너 — v1/runners 의 복제(얼굴 러너 제외). v2 규격이 바뀌면 여기만 고친다."""

from photoselect.v2.runners.arniqa import ArniqaRunner
from photoselect.v2.runners.laion import LaionRunner

__all__ = ["ArniqaRunner", "LaionRunner"]
