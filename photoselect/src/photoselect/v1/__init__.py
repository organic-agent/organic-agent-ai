"""v1 파이프라인 파사드 — 루트 배관(worker·__main__·handler)이 버전을 고를 때 쓰는 유일한 입구.

v1 = VLM(gemma3)·MediaPipe 얼굴·Bradley-Terry 취향 학습·근거 9종의 원본 파이프라인.
공용 계층은 `photoselect.config.Settings`(환경변수)·`db`·`jobs`·`storage` 뿐이다. v2 와는 import 관계가 없다 —
이 버전의 규격이 바뀌면 이 패키지 안만 고친다(중복은 의도한 것).
"""

from __future__ import annotations

from photoselect.config import Settings as BaseSettings
from photoselect.v1 import gallery, store  # noqa: F401 — 파사드 재노출
from photoselect.v1.config import Settings  # noqa: F401

NAME = "v1"


def settings_from(base: BaseSettings) -> Settings:
    """루트 Settings(환경) → 이 버전의 Settings(환경 + 손잡이). pipeline 은 이 버전으로 고정한다."""
    from dataclasses import replace
    return replace(Settings.from_base(base), pipeline=NAME)


def analyze_module():
    from photoselect.v1.analyze import job
    return job


def draft_module():
    from photoselect.v1.draft import job
    return job


def bedrock_client(settings: Settings):
    from photoselect.v1.llm.client import BedrockClient
    return BedrockClient(settings.llm.aws_region, settings.llm.model_id)
