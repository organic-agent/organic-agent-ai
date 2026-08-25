"""설정 — 환경변수와 CLI 인자가 여기로 모인다.

embedder와 같은 방식이다: `Settings.from_env()` 하나로 읽고, 코드 어디서도 `os.environ`을
직접 만지지 않는다. 지금은 **로컬 모드**(데이터셋 폴더 → out/ 폴더)만 실제로 동작하고,
DB·S3 필드는 wes V22 스키마가 머지되면 채운다 — 자리는 미리 잡아 둔다.

점수식의 손잡이(λ 곡선·MMR·클러스터 임계값)도 여기 있다. 전부 `study/01-recsys/lab`에서
합성 데이터로 정한 값이고 **실제 갤러리에서 다시 정할 대상**이다. 각 값 옆에 근거를 적었다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

#: 이 값이 `photo_analysis.model_version`에 들어간다. 모델·전처리·어휘 중 하나라도 바뀌면 올린다.
MODEL_VERSION = "photoselect-a-0.1"


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

    #: 한 라운드에 제시할 최대 장수.
    top_k: int = 30
    #: 셀렉 목표 장수(작가가 정한다). 담긴 사진이 여기 닿으면 완료. 남은 자리만큼만 제시한다.
    target_count: int = 30


@dataclass(frozen=True)
class AnalyzeKnobs:
    """전수 분석 상수."""

    #: 미리보기 긴 변. embedder 파생본과 같아야 점수 비교가 성립한다(study/00 step2).
    preview_long_edge: int = 1600
    #: 근접 중복 클러스터 코사인 임계값. 2026-08-25 류지혜(2) 200장 실측: 이웃 유사도 p90=0.966,
    #: 0.96에서 82클러스터(최대 15장·단독 49%). 갤러리마다 다시 볼 것 —
    #: analyze가 유사도 히스토그램을 찍어 준다.
    cluster_threshold: float = 0.96
    #: 파일명 순서가 이 거리 안이어야 같은 클러스터 후보로 본다(연사는 연속 촬영이다).
    cluster_window: int = 8
    #: VLM (Ollama) 설정. 프로덕션은 vLLM guided decoding으로 바뀐다.
    vlm_host: str = "http://127.0.0.1:11434"
    vlm_model: str = "gemma3:12b"
    vlm_long_edge: int = 1024
    vlm_timeout: float = 300.0


@dataclass(frozen=True)
class Settings:
    #: 로컬 모드의 출력 루트. 갤러리마다 하위 폴더가 생긴다.
    out_root: Path
    #: 로컬 모드의 데이터셋 루트 (스파이크와 같은 ../dataset).
    dataset_root: Path

    score: ScoreKnobs = field(default_factory=ScoreKnobs)
    analyze: AnalyzeKnobs = field(default_factory=AnalyzeKnobs)

    # ── 아래는 DB 모드 자리. wes V22 머지 후 embedder의 config.py를 그대로 따라 채운다. ──
    db_host: str | None = None
    db_port: int = 5432
    db_name: str | None = None
    db_user: str | None = None
    db_password: str | None = None
    s3_bucket: str | None = None

    @property
    def db_enabled(self) -> bool:
        return bool(self.db_host and self.db_name and self.db_user)

    @classmethod
    def from_env(cls) -> "Settings":
        here = Path(__file__).resolve().parent
        return cls(
            out_root=Path(os.environ.get("PHOTOSELECT_OUT", here / "out")),
            dataset_root=Path(os.environ.get("PHOTOSELECT_DATASET", here.parent.parent / "dataset")),
            db_host=os.environ.get("DB_HOST"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=os.environ.get("DB_NAME"),
            db_user=os.environ.get("DB_USER"),
            db_password=os.environ.get("DB_PASSWORD"),
            s3_bucket=os.environ.get("S3_BUCKET"),
        )
