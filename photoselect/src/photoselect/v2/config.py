"""v2 설정 — 슬림 파이프라인 손잡이와 v2 Settings (docs/plan-v2-slim.md). v1 과 공유하는 것 없음."""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from photoselect.config import MODULE_ROOT, Settings as BaseSettings  # noqa: F401

#: v2 파이프라인의 `photo_analysis.model_version`. v1(`MODEL_VERSION`)과 접두사가 달라 서로 재개 대상으로 안 보인다.
V2_MODEL_VERSION = "photoselect-v2-a-0.1"


@dataclass(frozen=True)
class V2Knobs:
    """v2 — 덜어낸 파이프라인 (docs/plan-v2-slim.md). VLM·얼굴·BT 없음."""

    # ── 분석 A ──
    #: 연사 클러스터 임계(코사인)·순서 창. v1과 같다. DINOv3 재임베딩 후 similarityProfile로 재확정.
    burst_threshold: float = 0.96
    burst_window: int = 8
    #: 컨셉 그룹 — 평균연결 계층 클러스터의 코사인 **거리** 임계. 2026-08-29 갤러리 1(DINOv2) 실측
    #: 0.2에서 70그룹 = 촬영 세트 단위(케이크·소파·해변…). DINOv3로 재측정 전까지 잠정값.
    concept_distance: float = 0.2
    #: 컨셉 그룹 수가 이 범위를 벗어나면 임계를 자동 조정한다(갤러리마다 분포가 다르다).
    concept_min_groups: int = 4
    concept_max_share: float = 0.5     # 최대 그룹 ≤ N·share
    #: CLIP zero-shot 피사체 태깅. 정밀도 스파이크 통과 전까지 **계산은 하되 점수·근거에는 안 쓴다**.
    subjects_zero_shot: bool = True
    subjects_trusted: bool = False

    # ── 추천 B ──
    w_technical: float = 0.5
    w_aesthetic: float = 0.5
    lambda_mmr: float = 0.7
    min_per_group: int = 1
    group_cap_ratio: float | None = 0.4
    #: 근거: 백분위를 말해도 되는 상한(상위 15%). 컨셉 그룹이 쿼터를 받고 근거에 쓰이는 최소 크기 —
    #: 그보다 작은 그룹(외톨이 사진)은 쿼터 없이 점수로만 뽑힌다(최하위 사진이 "다양성"으로 끼는 것을 막는다).
    reason_quality_top_pct: float = 15.0
    reason_concept_min_size: int = 3
    #: 4단계(선호·균형) 발동에 필요한 최소 담은 사진 수. subjects_trusted 가 아니면 발동하지 않는다.
    pref_min_selected: int = 5
    w_pref: float = 0.5
    w_balance: float = 0.5
    top_k: int = 30
    target_count: int = 30


@dataclass(frozen=True)
class LlmKnobs:
    """C — Bedrock 텍스트 호출. tech-stack.md §5 · spike-report 08-20 가용성 확인."""

    #: 서울 온디맨드에 Haiku 4.5가 없어 global. 크로스 리전 프로필. 텍스트만 보내므로 무방.
    aws_region: str = "ap-northeast-2"
    model_id: str = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
    #: 이유 문장 한 호출에 넣는 사진 수. 30장 초안이면 1회.
    reasons_batch: int = 40
    reasons_max_tokens: int = 4096
    feedback_max_tokens: int = 512


@dataclass(frozen=True)
class Settings(BaseSettings):
    """v2 Settings = 환경(BaseSettings) + v2 손잡이."""

    v2: V2Knobs = field(default_factory=V2Knobs)
    llm: LlmKnobs = field(default_factory=LlmKnobs)

    @classmethod
    def from_base(cls, base: BaseSettings) -> "Settings":
        return cls(**{f.name: getattr(base, f.name) for f in fields(BaseSettings)})

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.from_base(BaseSettings.from_env())
