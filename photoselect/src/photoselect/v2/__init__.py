"""v2 파이프라인 파사드 — 루트 배관(worker·__main__·handler)이 버전을 고를 때 쓰는 유일한 입구.

v2 = 슬림 파이프라인 (docs/plan-v2-slim.md): ARNIQA·LAION·고전 지표·연사/컨셉 클러스터, 근거 4종. VLM·얼굴·BT 없음.
공용 계층은 `photoselect.config.Settings`(환경변수)·`db`·`jobs`·`storage` 뿐이다. v1 와는 import 관계가 없다 —
이 버전의 규격이 바뀌면 이 패키지 안만 고친다(중복은 의도한 것).
"""

from __future__ import annotations

from photoselect.config import Settings as BaseSettings
from photoselect.v2 import gallery, store  # noqa: F401 — 파사드 재노출
from photoselect.v2.config import Settings  # noqa: F401

NAME = "v2"


def settings_from(base: BaseSettings) -> Settings:
    """루트 Settings(환경) → 이 버전의 Settings(환경 + 손잡이). pipeline 은 이 버전으로 고정한다."""
    from dataclasses import replace
    return replace(Settings.from_base(base), pipeline=NAME)


def analyze_module():
    from photoselect.v2 import analyze as job
    return job


def draft_module():
    from photoselect.v2 import draft as job
    return job


def bedrock_client(settings: Settings):
    from photoselect.v2.llm.client import BedrockClient
    return BedrockClient(settings.llm.aws_region, settings.llm.model_id)
