"""설정 — 환경변수(Settings) + CATEGORIZE 손잡이(Knobs·LlmKnobs) 한 파일.

categorize = 갤러리 단위 그룹화·이름 Lambda. torch 없음. score 가 저장한 원점수·CLIP 과 embedder 의 DINOv3 를
읽어 백분위·연사를 만들고, 촬영 시각(없으면 이미지)으로 자른 구간을 Bedrock 이 컨셉(1층)으로 묶은 뒤 1층 안에서
2층 그룹을 만들어 이름을 지어 `concept_assignments` 에 남긴다(#35, 컨셉 구간화 2026-09-30).

embedder 와 같은 방식: `Settings.from_env()` 하나로 읽고 코드 어디서도 `os.environ` 을 직접 만지지 않는다.
값의 근거는 wes docs/plans/ai-folder-structure.md · docs/photoselect/review-v3-design.md 실측.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 모듈 루트 = `categorize/`(config/settings.py 에서 두 단계 위). 로컬 산출물(out/)·데이터셋(../../dataset) 기본 경로의 기준점.
MODULE_ROOT = Path(__file__).resolve().parents[2]

#: `photo_analysis.pipeline_version`. **score 모듈의 같은 상수와 값이 같아야 한다** — 이 값과 같은 행만
#: "점수 있음"으로 읽는다. 값은 v3 시절 그대로: 바꾸면 전 갤러리가 재점수 대상이 된다.
# [GLOSSARY-1 2026-09-27] MODEL_VERSION → PIPELINE_VERSION (용어집 D3: 모델 id가 아니라 파이프라인 버전)
PIPELINE_VERSION = "photoselect-v3-a-0.1"

@dataclass(frozen=True)
class Knobs:
    """그룹화 + naming 의 손잡이. 값의 근거는 실측 문서."""

    # ── 연사 ──
    #: 연사 임계(코사인)·순서 창.
    burst_threshold: float = 0.96
    burst_window: int = 8

    # ── 1층 구간(service/segment.py) ──
    #: 촬영 시각이 있는 사진이 이 비율 이상이고 시각이 둘 이상 다르면 시간으로 자른다. 아니면 이미지(DINOv3 Ward)로.
    time_min_coverage: float = 0.9
    #: 이 초보다 길게 쉰 곳에서 자른다 — 옷·세트를 바꾸는 시간. 갤러리 18 실측(concept-segmentation-2026-09-30):
    #: 5분 공백 → VLM 묶기 ARI 0.938, 2분 공백도 같은 결과.
    segment_gap_s: float = 300.0
    #: 갤러리의 이 비율보다 작은 구간(테스트 컷)은 공백이 짧은 쪽 이웃에 붙인다. 그보다 큰 짧은 구간은 VLM 이 판단한다 —
    #: 2% 로 흡수하면 소품 클로즈업 구간(1.8%)이 공백 규칙대로 엉뚱한 컨셉에 붙었다(갤러리 18: ARI 0.889 → 0.5% 로 0.938).
    segment_min_share: float = 0.005
    #: 시간이 없을 때 DINOv3 Ward 로 나누는 묶음 수. 갤러리 18: 40묶음 순도 98.7%, VLM 묶기 ARI 0.80.
    visual_units: int = 36
    #: VLM 한 호출에 보내는 구간 타일 상한 — 넘으면 공백이 가장 짧은 이웃끼리 합친다.
    concept_max_units: int = 40
    #: 구간 타일(대표 4장 2×2) 한 변.
    concept_tile_edge: int = 768

    # ── 2층(1층 안의 임베딩 그룹) ──
    #: concat(DINOv3⊕CLIP) 평균연결 계층 클러스터의 코사인 거리 임계. 갤러리 1 실측: 0.2 에서 촬영 세트와 일치.
    group_distance: float = 0.2
    #: 과분할 가드 — 1층 안 그룹 수가 n×이 값을 넘으면 임계를 올린다 (grouping.embed_groups).
    group_frag_share: float = 0.35

    # ── 이름(service/naming.py) ──
    #: 2층 VLM 대상 — 1층마다 크기 내림차순으로 사진 커버리지가 이 값에 닿을 때까지. 나머지는 1층 안 최근접.
    naming_coverage: float = 0.85
    #: 2층 그룹 내 평균 중심 거리(spread)가 이보다 크면 대표를 2장(중심 최근접+최원점)으로.
    naming_spread_extra: float = 0.12
    #: 1층 하나에 보내는 2층 대표 이미지 상한 — 1층마다 호출 한 번.
    naming_chunk: int = 15
    #: 1층별 2층 호출을 동시에 보내는 수. 1 이면 직렬(스로틀 때 되돌리는 손잡이).
    naming_parallel: int = 4
    naming_max_tokens: int = 4096
    #: VLM에 보내는 2층 대표 JPEG 긴 변.
    naming_image_long_edge: int = 768


@dataclass(frozen=True)
class LlmKnobs:
    """Bedrock 호출(naming) — infrastructure/bedrock.py 머리말과 같은 제약 (Mantle 없음, global. 크로스 리전)."""

    aws_region: str = "ap-northeast-2"
    model_id: str = "global.anthropic.claude-sonnet-4-6"


@dataclass(frozen=True)
class Settings:
    """환경(DB·S3·경로) + 손잡이."""

    #: 로컬 모드의 출력 루트. score 의 out_root 와 같은 곳을 가리키면 두 CLI 가 이어진다.
    out_root: Path
    #: 로컬 모드의 데이터셋 루트 (../dataset).
    dataset_root: Path

    # ── DB 모드. 환경변수 이름은 embedder·score·wes scripts/local-ai.sh 와 같다. ──
    db_host: str | None = None
    db_port: int = 5432
    db_name: str | None = None
    db_user: str | None = None
    db_password: str | None = None
    db_sslmode: str = "require"
    db_sslrootcert: str | None = None
    #: 미리보기 JPEG 버킷 — naming 의 대표 사진 몇 장만 내려받는다.
    s3_bucket: str | None = None
    #: 대표 사진을 내려받는 자리. Lambda 는 /tmp 만 쓸 수 있다.
    work_dir: Path = Path("/tmp/categorize")

    knobs: Knobs = field(default_factory=Knobs)
    llm: LlmKnobs = field(default_factory=LlmKnobs)

    @property
    def db_enabled(self) -> bool:
        return bool(self.db_host and self.db_name and self.db_user)

    @classmethod
    def from_env(cls) -> "Settings":
        here = MODULE_ROOT
        return cls(
            out_root=Path(os.environ.get("CATEGORIZE_OUT", here / "out")),
            dataset_root=Path(os.environ.get("CATEGORIZE_DATASET", here.parent.parent / "dataset")),
            db_host=os.environ.get("DB_HOST"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=os.environ.get("DB_NAME"),
            db_user=os.environ.get("DB_USER"),
            db_password=os.environ.get("DB_PASSWORD"),
            db_sslmode=os.environ.get("DB_SSLMODE", "require"),
            db_sslrootcert=os.environ.get("DB_SSLROOTCERT"),
            s3_bucket=os.environ.get("S3_BUCKET"),
            work_dir=Path(os.environ.get("CATEGORIZE_WORK", "/tmp/categorize")),
            llm=LlmKnobs(
                aws_region=os.environ.get("BEDROCK_REGION", "ap-northeast-2"),
                model_id=os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-4-6"),
            ),
            knobs=Knobs(naming_parallel=int(os.environ.get("NAMING_PARALLEL", Knobs.naming_parallel))),
        )
