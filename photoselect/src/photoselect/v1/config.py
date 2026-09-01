"""v1 설정 — 점수식·분석·LLM 손잡이와 v1 Settings. 루트 `photoselect.config.Settings`(환경변수)를 상속한다.

embedder와 같은 방식이다: `Settings.from_env()` 하나로 읽고, 코드 어디서도 `os.environ`을
직접 만지지 않는다. 지금은 **로컬 모드**(데이터셋 폴더 → out/ 폴더)만 실제로 동작하고,
DB·S3 필드는 wes V22 스키마가 머지되면 채운다 — 자리는 미리 잡아 둔다.

점수식의 손잡이(λ 곡선·MMR·클러스터 임계값)도 여기 있다. 전부 `study/01-recsys/lab`에서
합성 데이터로 정한 값이고 **실제 갤러리에서 다시 정할 대상**이다. 각 값 옆에 근거를 적었다.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from photoselect.config import MODULE_ROOT, Settings as BaseSettings  # noqa: F401 — MODULE_ROOT 는 러너가 쓴다

#: 축별 VLM 태그 정확도 — 2026-08-25 50장 사람 채점 (spike-report.md). 프롬프트 2차 채점 후 갱신.
#: 정밀도 대용으로 쓴다(축 단위 채점이라 클래스별 정밀도는 없다). 이유 문장이 근거로 삼을 축을 고른다.
AXIS_PRECISION: dict[str, float] = {
    "subjects": 0.98, "framing": 0.92, "expression": 0.90, "lighting": 0.84, "scene": 0.74,
}

#: 이 값이 `photo_analysis.model_version`에 들어간다. 모델·전처리·어휘 중 하나라도 바뀌면 올린다.
MODEL_VERSION = "photoselect-a-0.3"   # 0.2: ARNIQA spaq · 0.3: VLM 프롬프트 1차 원문 복원


@dataclass(frozen=True)
class ScoreKnobs:
    """점수식 상수. 출처는 plan.md §3-B와 study/01-recsys/lab."""

    #: λ(n) = λ_min + (λ_max − λ_min)·n/(n+k). 증거 0개면 사진학 80%.
    lambda_min: float = 0.2
    lambda_max: float = 0.85
    #: k=10은 설계안 명세에서 역산한 값. study/02 step3 [D]는 합성 데이터에서 ~114를 측정했다.
    #: 실제 쌍 비교 데이터가 생기면 같은 절차로 재추정한다. 그전까지는 보수적으로 크게 둔다.
    lambda_k: float = 30.0

    #: 증거 종류별 가중치 — "λ = f(evidence 수)"의 evidence는 종류별 가중합이어야 한다
    #: (study/01 step6 §3: 반응 쌍은 온보딩 쌍보다 정보가 적다).
    w_pair: float = 1.0        # 온보딩 쌍 비교 1건
    w_rating: float = 0.5      # 별점에서 만든 쌍 1건
    w_selection: float = 0.3   # 선택/미선택에서 만든 쌍 1건

    #: 별점·선택으로 쌍을 만들 때 사람당 상한. 무한정 만들면 λ가 부풀려진다.
    max_derived_pairs: int = 60

    #: prior = w_t·technical_pct + w_a·aesthetic_pct (백분위 0~100, 갤러리 내).
    w_technical: float = 0.5
    w_aesthetic: float = 0.5

    #: BT 로지스틱 회귀의 L2. study/02 step2·step9: 편향-분산과 보정 둘 다 1.0 근처가 최적.
    bt_reg: float = 1.0
    #: conf(axis) 베타 사전 강도. 약한 사전이 안전하다(study/02 step3 [C]).
    conf_prior_a: float = 1.0

    #: MMR 다이얼. 오프라인으로 못 박지 않는다(study/01 step4 [B]) — 사람 눈으로 정할 값.
    lambda_mmr: float = 0.7
    #: 장면 커버리지: 갤러리에 존재하는 장면마다 최소 장수.
    min_per_scene: int = 1
    #: 한 장면이 초안의 이 비율을 넘지 못하게 (study/01 step4 [D]: 비례 배분은 snap 68%
    #: 갤러리에서 쏠림을 재생산한다). None이면 상한 없음.
    scene_cap_ratio: float | None = 0.4

    #: 이유 문장에 태그 값을 근거로 쓰려면 그 축의 태그 정밀도(AXIS_PRECISION)가 이 값 이상이어야 한다.
    #: 정밀도가 낮은 축을 근거로 쓰면 "퇴장 행진 컷이라 골랐어요"의 43%가 거짓이 된다(study/02 step8).
    reason_min_precision: float = 0.85
    #: 품질 백분위를 이유로 **말해도 되는** 상한. 상위 15% 안에 들 때만 "미학 상위 6%"처럼 말하고,
    #: 그 밖(예: 상위 68%)은 재료에서 아예 뺀다 — LLM 이 약점을 근거처럼 쓰는 것을 막는다.
    reason_quality_top_pct: float = 15.0
    #: "담으신 사진과 비슷한 컷"을 근거로 쓰려면 담은 사진과의 코사인이 이 값 이상이어야 한다
    #: (DINOv3 ViT-B/16, 정규화 벡터). Yeh & Barsky TOMM 2014 — 예시 기반 근거가 특징 기반보다 선호됨.
    reason_similar_min_cos: float = 0.6

    #: 한 라운드에 제시할 최대 장수.
    top_k: int = 30
    #: 셀렉 목표 장수(작가가 정한다). 담긴 사진이 여기 닿으면 완료. 남은 자리만큼만 제시한다.
    target_count: int = 30


@dataclass(frozen=True)
class AnalyzeKnobs:
    """전수 분석 상수."""

    #: 미리보기 긴 변. embedder 파생본과 같아야 점수 비교가 성립한다(study/00 step2).
    preview_long_edge: int = 1600
    #: 근접 중복 클러스터 코사인 임계값. 2026-08-25 류지혜(2) 200장 CLIP 실측 p90=0.966 → 0.96,
    #: 2026-08-29 갤러리 1 822장 DINOv2 실측 p75=0.951·p90=0.974에서도 같은 위치.
    #: **DINOv3(vitb16)로 재임베딩 후 재측정 필요** — analyze 결과의 similarityProfile을 본다.
    cluster_threshold: float = 0.96
    #: 파일명 순서가 이 거리 안이어야 같은 클러스터 후보로 본다(연사는 연속 촬영이다).
    cluster_window: int = 8
    #: VLM (Ollama) 설정. 프로덕션은 vLLM guided decoding으로 바뀐다.
    vlm_host: str = "http://127.0.0.1:11434"
    vlm_model: str = "gemma3:12b"
    vlm_long_edge: int = 1024
    vlm_timeout: float = 300.0




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
    """v1 Settings = 환경(BaseSettings) + v1 손잡이."""

    score: ScoreKnobs = field(default_factory=ScoreKnobs)
    analyze: AnalyzeKnobs = field(default_factory=AnalyzeKnobs)
    llm: LlmKnobs = field(default_factory=LlmKnobs)

    @classmethod
    def from_base(cls, base: BaseSettings) -> "Settings":
        return cls(**{f.name: getattr(base, f.name) for f in fields(BaseSettings)})

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.from_base(BaseSettings.from_env())
