"""설정 — 환경변수(Settings) + SCORE 손잡이(Knobs) 한 파일.

score = 사진별 점수 Lambda. 갤러리 전수에 CLIP ViT-L/14 · ARNIQA · LAION 미학을 돌려 `photo_analysis`의
원점수·피사체·`clip_embedding`을 적재한다. 그룹·이름은 `categorize` 모듈의 일이다(#35).

embedder 와 같은 방식: `Settings.from_env()` 하나로 읽고 코드 어디서도 `os.environ` 을 직접 만지지 않는다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# 모듈 루트 = `score/`(이 파일은 score/score/config/ 아래라 세 단계 위). 로컬 산출물(out/)·가중치 캐시(weights/)·
# 데이터셋(../../dataset) 기본 경로의 기준점.
MODULE_ROOT = Path(__file__).resolve().parents[2]

#: `photo_analysis.model_version`. **categorize 모듈의 같은 상수와 값이 같아야 한다** — categorize 는
#: 이 값과 같은 행만 "점수 있음"으로 읽는다. 값은 v3 시절 그대로 둔다: 이미 적재된 행과 재개(스킵) 판정이
#: 이 문자열로 묶여 있어, 바꾸면 전 갤러리가 재점수 대상이 된다.
MODEL_VERSION = "photoselect-v3-a-0.1"

#: 큰 분류(부모) 고정 목록. categorize 의 naming 이 같은 목록을 Bedrock 스키마 enum 으로 쓴다 — 두 모듈이 같아야 한다.
#: 서비스 대상은 결혼식 전 앨범·청첩장용 **스튜디오 컨셉 촬영**뿐이다 — 본식·피로연은 다루지 않으므로
#: 촬영 종류 분기 없이 목록 하나다. '기타'는 목록에 항상 있다.
PARENTS: list[str] = ["실내 스튜디오", "하우스·인테리어", "한옥·전통", "야외 정원·건물",
                      "야외 자연", "도심·거리", "기타"]

#: 부모 검증(CLIP zero-shot)용 영어 프롬프트. '기타'는 없다 — zero-shot 후보에서 뺀다.
#: 검증 전용이다(판정 아님, 822장 실측 일치 79%): categorize 의 naming 이 VLM 부모와 다르면 needs_review 근거.
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
    """SCORE 의 손잡이."""

    #: CLIP zero-shot 피사체(신부/신랑/커플/단체). 확신 라벨 36/36 검증됨.
    subjects_zero_shot: bool = True
    #: 이 장수마다 DB 에 쓰고 commit 한다 — 데드라인에 멈추거나 죽어도 그때까지의 점수는 남는다.
    write_batch: int = 32
    #: CLIP 이미지 인코딩을 이 장수씩 한 forward 로 묶는다(#51). write_batch 의 약수가 자연스럽다.
    clip_batch: int = 8
    #: ARNIQA 입력 긴 변(#51). 1600 → 1024 로 연산 ~2.4배 절감. 바꾸면 technical_score 스케일이 바뀐다.
    arniqa_long_edge: int = 1024
    #: ARNIQA 를 이 장수씩 한 forward 로(#68). 같은 픽셀 크기끼리만 묶인다(세로 683×1024 / 가로 1024×683) — 갤러리는
    #: 대개 한 방향이 95% 라 묶음이 거의 그대로 산다. CPU 에서는 1 이 낫다(배치 이득 없음, 메모리만 든다).
    arniqa_batch: int = 1
    #: 연산 장치. "auto" 면 cuda → mps → cpu 순. Lambda 는 cpu 로 떨어지고, SageMaker/EC2 GPU 는 cuda(#68).
    device: str = "auto"
    #: cuda 에서 fp16 autocast. 처리량 ~2배, 점수는 소수점 셋째 자리에서 흔들린다 — CPU 점수와 Spearman ≥ 0.99 검증 뒤 씀.
    #: cpu·mps 에서는 무시된다.
    fp16: bool = True
    #: 디코드·classical(선명도) 을 GPU 추론과 겹치게 하는 스레드 수. 0 이면 지금처럼 한 스레드에서 순서대로.
    #: GPU 는 4 vCPU 의 JPEG 디코드를 기다리는 게 병목이라 GPU 환경에서 켠다. Lambda(CPU) 는 0 — 디코드와 추론이 같은 코어를 다툰다.
    decode_workers: int = 0


@dataclass(frozen=True)
class Settings:
    """환경(DB·S3·경로) + 손잡이."""

    #: 로컬 모드의 출력 루트. 갤러리마다 하위 폴더가 생긴다.
    out_root: Path
    #: 로컬 모드의 데이터셋 루트 (../dataset).
    dataset_root: Path

    # ── DB 모드. 환경변수 이름은 embedder·wes scripts/local-ai.sh 와 같다. ──
    db_host: str | None = None
    db_port: int = 5432
    db_name: str | None = None
    db_user: str | None = None
    #: Lambda 는 인프라 env, GPU 워커는 호스트 env 스크립트가 SSM 에서 읽어 --env-file 로 넘긴다(#91, score/deploy/gpu-worker/).
    #: 컨테이너가 SSM 을 직접 읽는 경로는 없다 — 비밀을 읽는 주체를 호스트 한 곳으로.
    db_password: str | None = None
    #: RDS 는 평문 접속을 거부하므로 기본 require. 로컬 docker pg 는 DB_SSLMODE=disable.
    db_sslmode: str = "require"
    db_sslrootcert: str | None = None
    #: 미리보기 JPEG 가 있는 버킷. DB 모드가 여기서 내려받는다.
    s3_bucket: str | None = None
    #: DB 모드에서 미리보기를 내려받는 자리. 갤러리마다 하위 폴더. Lambda 는 /tmp 만 쓸 수 있다.
    work_dir: Path = Path("/tmp/score")

    #: GPU 워커(#75): 한 번에 집는 장수 · 집을 게 없을 때 대기 초 · 연속 유휴가 이 초를 넘기면 자기 인스턴스를 정지(0 이면 안 함).
    #: 유휴 기본 30초(#81) — 지금은 갤러리를 연달아 처리할 사용자가 없어 켜 둘 이유가 없다. 다중 사용자 운영에서는 600 으로(콜드 16s·로드 22s 를 아낌).
    worker_batch: int = 32
    worker_poll_seconds: float = 3.0
    worker_idle_stop_seconds: int = 30
    #: 배치가 이만큼 연속으로 실패하면 루프를 끝낸다(#81) — 같은 오류로 헛도는 것을 막는다. 종료 코드 1, 인스턴스 정지는 wes 감시 몫.
    worker_max_consecutive_failures: int = 5
    #: 이만큼 배치를 처리하면 루프를 끝낸다(0 = 무한). 검증·벤치마크용.
    worker_max_batches: int = 0

    #: 미리보기를 S3 에서 동시에 내려받는 스레드 수(#68). 장당 왕복이 병목이라 8 이면 한 프로세스가 7,000장을 1분대에 받는다.
    download_workers: int = 8

    #: Lambda 타임아웃 앞에서 멈출 여유(초). "지금까지 가장 오래 걸린 쓰기 배치 + 이 값"보다 남은 시간이
    #: 적으면 배치 경계에서 멈추고 commit 한다(embedder 와 같은 규칙). 로컬 CLI 에는 데드라인이 없다.
    stop_margin_seconds: int = 60

    knobs: Knobs = field(default_factory=Knobs)

    @property
    def db_enabled(self) -> bool:
        return bool(self.db_host and self.db_name and self.db_user)

    @classmethod
    def from_env(cls) -> "Settings":
        here = MODULE_ROOT
        return cls(
            out_root=Path(os.environ.get("SCORE_OUT", here / "out")),
            dataset_root=Path(os.environ.get("SCORE_DATASET", here.parent.parent / "dataset")),
            db_host=os.environ.get("DB_HOST"),
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=os.environ.get("DB_NAME"),
            db_user=os.environ.get("DB_USER"),
            db_password=os.environ.get("DB_PASSWORD"),
            db_sslmode=os.environ.get("DB_SSLMODE", "require"),
            db_sslrootcert=os.environ.get("DB_SSLROOTCERT"),
            s3_bucket=os.environ.get("S3_BUCKET"),
            work_dir=Path(os.environ.get("SCORE_WORK", "/tmp/score")),
            download_workers=int(os.environ.get("SCORE_DOWNLOAD_WORKERS", "8")),
            worker_batch=int(os.environ.get("WORKER_BATCH", "32")),
            worker_poll_seconds=float(os.environ.get("WORKER_POLL_SECONDS", "3")),
            worker_idle_stop_seconds=int(os.environ.get("WORKER_IDLE_STOP_SECONDS", "30")),
            worker_max_consecutive_failures=int(os.environ.get("WORKER_MAX_CONSECUTIVE_FAILURES", "5")),
            worker_max_batches=int(os.environ.get("WORKER_MAX_BATCHES", "0")),
            stop_margin_seconds=int(os.environ.get("STOP_MARGIN_SECONDS", "60")),
            knobs=Knobs(
                clip_batch=int(os.environ.get("CLIP_BATCH", Knobs.clip_batch)),
                arniqa_long_edge=int(os.environ.get("ARNIQA_LONG_EDGE", Knobs.arniqa_long_edge)),
                arniqa_batch=int(os.environ.get("ARNIQA_BATCH", Knobs.arniqa_batch)),
                device=os.environ.get("SCORE_DEVICE", Knobs.device),
                fp16=os.environ.get("SCORE_FP16", "1" if Knobs.fp16 else "0") not in ("0", "false", "no", ""),
                decode_workers=int(os.environ.get("SCORE_DECODE_WORKERS", Knobs.decode_workers)),
            ),
        )
