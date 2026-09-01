"""공용 설정 — 환경변수만. 점수식·모델 손잡이는 버전별 `v1/config.py`·`v2/config.py` 에 있다.

embedder 와 같은 방식: `Settings.from_env()` 하나로 읽고 코드 어디서도 `os.environ` 을 직접 만지지 않는다.
버전 모듈은 이 Settings 를 상속해 자기 손잡이를 얹는다(`v1.config.Settings`, `v2.config.Settings`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# 모듈 루트 = `photoselect/` (src/photoselect/config.py 기준 두 단계 위). 로컬 산출물(out/)·가중치 캐시(weights/)·
# 데이터셋(../../dataset) 기본 경로의 기준점. 환경변수(PHOTOSELECT_OUT 등)가 있으면 그쪽이 우선.
MODULE_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    #: 로컬 모드의 출력 루트. 갤러리마다 하위 폴더가 생긴다.
    out_root: Path
    #: 로컬 모드의 데이터셋 루트 (스파이크와 같은 ../dataset).
    dataset_root: Path

    # ── DB 모드 (wes V29). 환경변수 이름은 embedder·wes scripts/local-ai.sh와 같다. ──
    db_host: str | None = None
    db_port: int = 5432
    db_name: str | None = None
    db_user: str | None = None
    db_password: str | None = None
    #: RDS는 평문 접속을 거부하므로 기본 require. 로컬 docker pg는 DB_SSLMODE=disable.
    db_sslmode: str = "require"
    db_sslrootcert: str | None = None
    #: 미리보기 JPEG가 있는 버킷. analyze --db 가 여기서 내려받는다.
    s3_bucket: str | None = None
    #: DB 모드에서 미리보기를 내려받는 자리. 갤러리마다 하위 폴더.
    work_dir: Path = Path("/tmp/photoselect")
    #: 어느 파이프라인을 돌릴지 — "v1"(VLM·얼굴·BT) / "v2"(슬림) / "v3"(폴더화, V45).
    #: CLI --pipeline 이 우선, 다음은 환경변수 PHOTOSELECT_PIPELINE, 기본 v2. 워커·Lambda 도 이 값을 본다.
    pipeline: str = "v2"

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
            pipeline=os.environ.get("PHOTOSELECT_PIPELINE", "v2"),
        )
