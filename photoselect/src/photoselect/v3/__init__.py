"""v3 파이프라인 파사드 — 폴더화 (wes docs/plans/ai-folder-structure.md, V45 스키마).

v3 = FULL(사진별 분석 + 임베딩 그룹) → naming(VLM 이름·배정 → ai_concept_assignments).
추천(draft)은 아직 없다 — 폴더별 추천(docs/plan-v3-folder-compare.md)을 구현할 때 여기 온다.
v2 와는 import 관계가 없다(중복은 의도한 것) — v2 는 V29 스키마의 이전 기능을 그대로 유지한다.
"""

from __future__ import annotations

from photoselect.config import Settings as BaseSettings
from photoselect.v3 import gallery, store  # noqa: F401 — 파사드 재노출
from photoselect.v3.config import Settings  # noqa: F401

NAME = "v3"


def settings_from(base: BaseSettings) -> Settings:
    from dataclasses import replace
    return replace(Settings.from_base(base), pipeline=NAME)


def analyze_module():
    from photoselect.v3 import analyze as job
    return job


def naming_module():
    from photoselect.v3 import naming as job
    return job


def draft_module():
    raise RuntimeError(
        "v3 에는 추천(draft)이 아직 없다 — 폴더별 추천은 docs/plan-v3-folder-compare.md 구현 대기. "
        "추천은 --pipeline v2 로 돌리거나 v3 구현을 기다려라")


def bedrock_client(settings: Settings):
    from photoselect.v3.llm.client import BedrockClient
    return BedrockClient(settings.llm.aws_region, settings.llm.model_id)
