"""설정 — 환경변수(Settings) + CATEGORIZE 손잡이(Knobs·LlmKnobs) 한 파일.

categorize = 갤러리 단위 그룹화·이름 Lambda. torch 없음. score 가 저장한 원점수·CLIP 과 embedder 의 DINOv3 를
읽어 백분위·연사·임베딩 그룹을 만들고, Bedrock 으로 그룹 이름을 지어 `ai_concept_assignments` 에 남긴다(#35).

embedder 와 같은 방식: `Settings.from_env()` 하나로 읽고 코드 어디서도 `os.environ` 을 직접 만지지 않는다.
값의 근거는 wes docs/plans/ai-folder-structure.md · docs/photoselect/review-v3-design.md 실측.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 모듈 루트 = `categorize/`. 로컬 산출물(out/)·데이터셋(../../dataset) 기본 경로의 기준점.
MODULE_ROOT = Path(__file__).resolve().parents[1]

#: `photo_analysis.model_version`. **score 모듈의 같은 상수와 값이 같아야 한다** — 이 값과 같은 행만
#: "점수 있음"으로 읽는다. 값은 v3 시절 그대로: 바꾸면 전 갤러리가 재점수 대상이 된다.
MODEL_VERSION = "photoselect-v3-a-0.1"

#: 큰 분류(부모) 고정 목록 — naming 이 Bedrock 스키마 enum 으로 강제한다. score 의 ParentTagger 가 같은 목록으로
#: 사진마다 clip_parent 를 저장하므로 두 모듈이 같아야 한다. '기타'는 목록에 항상 있다.
PARENTS: list[str] = ["실내 스튜디오", "하우스·인테리어", "한옥·전통", "야외 정원·건물",
                      "야외 자연", "도심·거리", "기타"]


@dataclass(frozen=True)
class Knobs:
    """그룹화 + naming 의 손잡이. 값의 근거는 실측 문서."""

    # ── 그룹화 ──
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

    # ── naming ──
    #: VLM 대상 선정 — 상수 K 대신 커버리지 목표. 크기 내림차순으로 사진 누적 커버리지가
    #: 이 값에 닿을 때까지 그룹을 고른다. 갤러리 분포에 자동 적응한다.
    naming_coverage: float = 0.85
    #: 커버리지와 무관한 그룹 수 상한 — 비용의 절대 상한 (⌈이미지 수/naming_chunk⌉+1 회).
    naming_max_groups: int = 120
    #: 그룹 내 평균 중심 거리(spread)가 이보다 크면 대표를 2장(중심 최근접+최원점)으로.
    naming_spread_extra: float = 0.12
    #: Bedrock 한 요청에 넣는 대표 이미지 수.
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
        )
