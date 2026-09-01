"""v3 설정 — 폴더화 파이프라인 (wes docs/plans/ai-folder-structure.md, V45).

v3 = FULL(사진별 분석 + 임베딩 그룹) → NAMING(VLM 이름·배정) → 폴더별 추천(draft,
docs/plan-v3-folder-compare.md §2 — 자식 폴더마다 점수 상위 n_f장, 이유는 2단계).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields

from photoselect.config import MODULE_ROOT, Settings as BaseSettings  # noqa: F401

#: v3 파이프라인의 `photo_analysis.model_version`. v2와 접두사가 달라 서로 재개 대상으로 안 보인다.
V3_MODEL_VERSION = "photoselect-v3-a-0.1"

#: 큰 분류(부모) 고정 목록 — 촬영 종류(galleries.shoot_type)별. ai-folder-structure.md의 초안.
#: '기타'는 목록에 항상 있다(VLM이 목록 밖이라고 판단할 때 + proposed_parent). OTHER 촬영은
#: 리허설 목록을 쓴다 — 데이터로 확정하기 전의 잠정 규칙.
PARENTS: dict[str, list[str]] = {
    "REHEARSAL": ["실내 스튜디오", "하우스·인테리어", "한옥·전통", "야외 정원·건물",
                  "야외 자연", "도심·거리", "기타"],
    "CEREMONY": ["준비·신부대기실", "예식", "단체·가족", "폐백·전통", "피로연·연회",
                 "야외·스냅", "기타"],
}
PARENTS["OTHER"] = PARENTS["REHEARSAL"]

#: 부모 검증(CLIP zero-shot)용 영어 프롬프트. '기타'는 없다 — zero-shot 후보에서 뺀다.
#: 검증 전용이다(판정 아님, 822장 실측 일치 79%): VLM 부모와 다르면 needs_review 근거.
PARENT_PROMPTS: dict[str, list[str]] = {
    "실내 스튜디오": ["an indoor photography studio with a seamless backdrop and studio lighting",
                    "a studio portrait against a plain paper background"],
    "하우스·인테리어": ["an indoor set with furniture, a sofa and house interior decoration",
                     "a cozy room interior with props and furniture"],
    "한옥·전통": ["a traditional Korean hanok house with wooden pillars and tiled roof",
               "people wearing traditional Korean hanbok clothing"],
    "야외 정원·건물": ["outdoors in a landscaped garden next to buildings or architecture",
                   "a garden path, archway or stairs by a building"],
    "야외 자연": ["outdoors in open nature such as a beach, forest, field or lawn",
              "a natural landscape with sea, trees or grass and no buildings"],
    "도심·거리": ["a city street or downtown area with roads, shops and traffic",
              "an urban night street with city lights"],
    "준비·신부대기실": ["a bride getting ready in a waiting room, hair and makeup preparation"],
    "예식": ["a wedding ceremony in a hall, walking down the aisle, exchanging vows"],
    "단체·가족": ["a formal group photo of family and guests at a wedding"],
    "폐백·전통": ["a traditional Korean wedding ritual with hanbok"],
    "피로연·연회": ["a wedding reception or banquet with tables and food"],
    "야외·스냅": ["candid outdoor snapshots outside the wedding venue"],
}


@dataclass(frozen=True)
class V3Knobs:
    """v3 — 분석(A) + 이름·배정(naming). 값의 근거는 ai-folder-structure.md 실측."""

    # ── 분석 A ──
    #: 연사 클러스터 임계(코사인)·순서 창. v2와 같다.
    burst_threshold: float = 0.96
    burst_window: int = 8
    #: 임베딩 그룹(embed_group_id) — concat(DINOv3⊕CLIP) 평균연결 계층 클러스터의 코사인 거리 임계.
    #: 갤러리 1 실측: concat 0.2에서 58그룹, DINOv3 단독과 ARI 0.95, VLM 저신뢰 그룹 9→5.
    group_distance: float = 0.2
    group_min_groups: int = 4
    group_max_share: float = 0.5
    #: 과분할 가드 — 그룹 수가 n×이 값을 넘으면 임계를 올린다 (concept.concept_groups, 대칭 규칙).
    group_frag_share: float = 0.35
    #: CLIP zero-shot 피사체(신부/신랑/커플/단체). v2에서 검증됨(확신 라벨 36/36).
    subjects_zero_shot: bool = True

    # ── naming ──
    #: VLM 대상 선정 — 상수 K 대신 커버리지 목표 (review-v3-design.md (3)). 크기 내림차순으로
    #: 사진 누적 커버리지가 이 값에 닿을 때까지 그룹을 고른다. 갤러리 분포에 자동 적응한다.
    naming_coverage: float = 0.85
    #: 커버리지와 무관한 그룹 수 상한 — 비용의 절대 상한 (⌈이미지 수/naming_chunk⌉+1 회).
    naming_max_groups: int = 120
    #: 그룹 내 평균 중심 거리(spread)가 이보다 크면 대표를 2장(중심 최근접+최원점)으로 —
    #: 이질적 그룹의 이름이 일부만 대표하는 문제 (review-v3-design.md (2)). 잠정값, 홀드아웃 실측 후 조정.
    naming_spread_extra: float = 0.12
    #: Bedrock 한 요청에 넣는 대표 이미지 수 (요청당 이미지 20장 한도로 알려짐 — 실호출 확인 후 조정).
    naming_chunk: int = 15
    naming_max_tokens: int = 4096
    #: VLM에 보내는 대표 JPEG 긴 변.
    naming_image_long_edge: int = 768
    #: K 밖 소그룹 최근접 배정의 코사인 **거리** 상한 — 이보다 멀면 '기타/기타' + needs_review.
    nearest_tau: float = 0.25
    #: VLM confidence가 이보다 낮으면 needs_review.
    review_confidence: float = 0.8

    # ── 추천 B (폴더별, plan-v3-folder-compare.md §2) ──
    #: 점수식 가중치 — v2와 같다 (prior = 0.5·tech_pct + 0.5·aes_pct, 갤러리 내 백분위 유지).
    w_technical: float = 0.5
    w_aesthetic: float = 0.5
    #: 4단계(균형·선호) — subjects_trusted 이고 담은 사진 ≥ pref_min_selected 일 때만.
    subjects_trusted: bool = True
    pref_min_selected: int = 5
    w_balance: float = 0.5
    w_pref: float = 0.5
    #: 폴더 안 MMR — 같은 폴더 안 중복 구도 억제.
    lambda_mmr: float = 0.7
    #: 폴더당 추천 상한 비율 — n_f ≤ ceil(|f|·이 값). 폴더의 절반 넘게 추천하지 않는다.
    folder_cap_ratio: float = 0.5
    #: 목표 장수 폴백 — galleries.max_selectable_photo_count 가 없을 때(로컬).
    target_count: int = 30
    #: 근거: 백분위를 말해도 되는 상한(상위 15%).
    reason_quality_top_pct: float = 15.0
    #: 절대 품질 하한 게이트 (review-v3-design.md (6)) — 원점수가 하한 미만이면 **품질 표현만**
    #: 근거에서 뺀다. 사진을 거르는 데는 쓰지 않는다(하드 필터 금지). 값은 잠정 — 홀드아웃 실측 후 조정.
    quality_floor_technical: float = 0.35   # ARNIQA(0~1)
    quality_floor_aesthetic: float = 4.5    # LAION(0~10)


@dataclass(frozen=True)
class LlmKnobs:
    """Bedrock 호출 — v2 llm/client.py 머리말과 같은 제약 (Mantle 없음, global. 크로스 리전)."""

    aws_region: str = "ap-northeast-2"
    model_id: str = "global.anthropic.claude-sonnet-4-6"
    #: 이유 문장 한 호출에 넣는 사진 수. 사진마다 이미지(본인 + 형제)가 붙으므로 작게.
    reasons_batch: int = 10
    reasons_max_tokens: int = 8192
    #: 사진을 LLM 에 보여 줄지. 끄면 텍스트 재료만 간다.
    reasons_vision: bool = True
    #: LLM 에 보내는 JPEG 의 긴 변. 768이면 장당 ~800 토큰.
    reasons_image_long_edge: int = 768
    #: 비교 대상으로 같이 보여 줄 형제(연사) 사진 최대 수.
    reasons_sibling_images: int = 2


@dataclass(frozen=True)
class Settings(BaseSettings):
    """v3 Settings = 환경(BaseSettings) + v3 손잡이."""

    v3: V3Knobs = field(default_factory=V3Knobs)
    llm: LlmKnobs = field(default_factory=LlmKnobs)

    @classmethod
    def from_base(cls, base: BaseSettings) -> "Settings":
        return cls(**{f.name: getattr(base, f.name) for f in fields(BaseSettings)})

    @classmethod
    def from_env(cls) -> "Settings":
        return cls.from_base(BaseSettings.from_env())
