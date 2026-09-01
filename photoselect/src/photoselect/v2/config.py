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
    #: CLIP zero-shot 피사체 태깅. 2026-08-29 갤러리 1(822장, DINOv3 재임베딩 후) 사람 검증:
    #: 확신 라벨(margin ≥ 0.01) 무작위 36장 = 36 정답(신부 12·신랑 12·커플 12). 보류(unknown) 84장의
    #: argmax 는 커플 70 정답, 신부/신랑 14 중 9가 실은 커플 → margin 게이트가 애매한 것만 걸러낸다.
    #: 그래서 trusted=True. 다른 갤러리(본식·야외)에서 같은 방식으로 재확인할 것.
    subjects_zero_shot: bool = True
    subjects_trusted: bool = True

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
    """C — Bedrock 호출. tech-stack.md §5 · client.py 머리말."""

    #: 서울 온디맨드에 Sonnet이 없어 global. 크로스 리전 프로필. 이미지도 보내므로 국외 라우팅을 감수한다.
    #: `global.anthropic.claude-sonnet-5` 는 이 계정에 모델 액세스가 아직 없다(2026-08-30 403) — Bedrock 콘솔에서
    #: 열리면 그 id 로 바꾼다. 4.6 은 이미지 + output_config 실호출 확인.
    aws_region: str = "ap-northeast-2"
    model_id: str = "global.anthropic.claude-sonnet-4-6"
    #: 이유 문장 한 호출에 넣는 사진 수. 사진마다 이미지(본인 + 형제)가 붙으므로 작게 — 30장 초안이면 3회.
    reasons_batch: int = 10
    reasons_max_tokens: int = 8192
    #: 사진을 LLM 에 보여 줄지. 끄면 텍스트 재료만 간다(옛 동작).
    reasons_vision: bool = True
    #: LLM 에 보내는 JPEG 의 긴 변. 768이면 장당 ~800 토큰.
    reasons_image_long_edge: int = 768
    #: 비교 대상으로 같이 보여 줄 형제(연사) 사진 최대 수.
    reasons_sibling_images: int = 2
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
