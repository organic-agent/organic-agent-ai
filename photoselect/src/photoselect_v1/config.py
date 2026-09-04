"""설정 — 환경변수(Settings) + 파이프라인 손잡이(Knobs·LlmKnobs) 한 파일.

photoselect-v1 = AI 클러스터링 폴더화 배치:
    foldering  FULL(사진별 분석 + 임베딩 그룹) → NAMING(VLM 이름·배정). 추천·비교샷은 wes(#25).

embedder 와 같은 방식: `Settings.from_env()` 하나로 읽고 코드 어디서도 `os.environ` 을 직접
만지지 않는다. 값의 근거는 wes docs/plans/ai-folder-structure.md · docs/plan-v3-folder-compare.md 실측.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 모듈 루트 = `photoselect/` (src/photoselect_v1/config.py 기준 두 단계 위). 로컬 산출물(out/)·
# 가중치 캐시(weights/)·데이터셋(../../dataset) 기본 경로의 기준점. 환경변수가 있으면 그쪽이 우선.
MODULE_ROOT = Path(__file__).resolve().parents[2]

#: `photo_analysis.model_version`. 값은 v3 시절 그대로 둔다 — 이미 적재된 행과 재개(스킵) 판정이
#: 이 문자열로 묶여 있어, 바꾸면 전 갤러리가 재분석 대상이 된다.
MODEL_VERSION = "photoselect-v3-a-0.1"

#: 큰 분류(부모) 고정 목록. ai-folder-structure.md의 초안.
#: 서비스 대상은 결혼식 전 앨범·청첩장용 **스튜디오 컨셉 촬영**뿐이다 — 본식·피로연은
#: 다루지 않으므로 촬영 종류(shoot_type) 분기 없이 목록 하나다.
#: '기타'는 목록에 항상 있다(VLM이 목록 밖이라고 판단할 때 + proposed_parent).
PARENTS: list[str] = ["실내 스튜디오", "하우스·인테리어", "한옥·전통", "야외 정원·건물",
                      "야외 자연", "도심·거리", "기타"]

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
}


@dataclass(frozen=True)
class Knobs:
    """분석(FULL) + 이름·배정(naming)의 손잡이. 값의 근거는 실측 문서."""

    # ── 분석 (foldering.analyze) ──
    #: 연사 클러스터 임계(코사인)·순서 창.
    burst_threshold: float = 0.96
    burst_window: int = 8
    #: 임베딩 그룹(embed_group_id) — concat(DINOv3⊕CLIP) 평균연결 계층 클러스터의 코사인 거리 임계.
    #: 갤러리 1 실측: concat 0.2에서 58그룹, DINOv3 단독과 ARI 0.95, VLM 저신뢰 그룹 9→5.
    group_distance: float = 0.2
    group_min_groups: int = 4
    group_max_share: float = 0.5
    #: 과분할 가드 — 그룹 수가 n×이 값을 넘으면 임계를 올린다 (concept.concept_groups, 대칭 규칙).
    group_frag_share: float = 0.35
    #: CLIP zero-shot 피사체(신부/신랑/커플/단체). 확신 라벨 36/36 검증됨.
    subjects_zero_shot: bool = True

    # ── naming (foldering.naming) ──
    #: VLM 대상 선정 — 상수 K 대신 커버리지 목표. 크기 내림차순으로 사진 누적 커버리지가
    #: 이 값에 닿을 때까지 그룹을 고른다. 갤러리 분포에 자동 적응한다.
    naming_coverage: float = 0.85
    #: 커버리지와 무관한 그룹 수 상한 — 비용의 절대 상한 (⌈이미지 수/naming_chunk⌉+1 회).
    naming_max_groups: int = 120
    #: 그룹 내 평균 중심 거리(spread)가 이보다 크면 대표를 2장(중심 최근접+최원점)으로 —
    #: 이질적 그룹의 이름이 일부만 대표하는 문제. 잠정값, 홀드아웃 실측 후 조정.
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


@dataclass(frozen=True)
class LlmKnobs:
    """Bedrock 호출(naming) — llm.py 머리말과 같은 제약 (Mantle 없음, global. 크로스 리전)."""

    aws_region: str = "ap-northeast-2"
    model_id: str = "global.anthropic.claude-sonnet-4-6"


@dataclass(frozen=True)
class Settings:
    """환경(DB·S3·경로) + 손잡이. 파이프라인 선택은 없다 — 이 패키지가 곧 파이프라인 하나다."""

    #: 로컬 모드의 출력 루트. 갤러리마다 하위 폴더가 생긴다.
    out_root: Path
    #: 로컬 모드의 데이터셋 루트 (스파이크와 같은 ../dataset).
    dataset_root: Path

    # ── DB 모드. 환경변수 이름은 embedder·wes scripts/local-ai.sh와 같다. ──
    db_host: str | None = None
    db_port: int = 5432
    db_name: str | None = None
    db_user: str | None = None
    db_password: str | None = None
    #: RDS는 평문 접속을 거부하므로 기본 require. 로컬 docker pg는 DB_SSLMODE=disable.
    db_sslmode: str = "require"
    db_sslrootcert: str | None = None
    #: 미리보기 JPEG가 있는 버킷. score --db 가 여기서 내려받는다.
    s3_bucket: str | None = None
    #: DB 모드에서 미리보기를 내려받는 자리. 갤러리마다 하위 폴더.
    work_dir: Path = Path("/tmp/photoselect")

    knobs: Knobs = field(default_factory=Knobs)
    llm: LlmKnobs = field(default_factory=LlmKnobs)

    @property
    def db_enabled(self) -> bool:
        return bool(self.db_host and self.db_name and self.db_user)

    @classmethod
    def from_env(cls) -> "Settings":
        here = MODULE_ROOT
        return cls(
            out_root=Path(os.environ.get("PHOTOSELECT_OUT", here / "out")),
            dataset_root=Path(os.environ.get("PHOTOSELECT_DATASET", here.parent.parent / "dataset")),
            db_host=os.environ.get("DB_HOST"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=os.environ.get("DB_NAME"),
            db_user=os.environ.get("DB_USER"),
            db_password=os.environ.get("DB_PASSWORD"),
            db_sslmode=os.environ.get("DB_SSLMODE", "require"),
            db_sslrootcert=os.environ.get("DB_SSLROOTCERT"),
            s3_bucket=os.environ.get("S3_BUCKET"),
            work_dir=Path(os.environ.get("PHOTOSELECT_WORK", "/tmp/photoselect")),
        )
