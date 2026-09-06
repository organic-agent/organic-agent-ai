"""환경변수 하나로 모아 읽는다.

Lambda에는 Terraform이 값을 넣어 주고(인프라 레포의 `modules/embedding`), 로컬 실행은 셸
환경에서 온다.

원래 이 모듈에서 가장 중요한 점은 DB 비밀번호가 **없다**는 것이었다. 접속은 RDS IAM 인증을
쓰고, `db.py`가 매 실행마다 짧은 수명의 토큰을 만들어 비밀번호 자리에 넣었다.

지금은 조직 SCP가 `rds-db:connect`를 계정 전체에서 거부해 그 설계를 쓰지 못한다. 임시로
`DB_PASSWORD`를 받아 쓴다. 이 환경변수는 Terraform이 넣지 않는다 -- 넣으면 state에 평문으로
남기 때문에, apply 밖에서 주입하고 `ignore_changes`가 지켜 준다.

**이건 임시 우회로다.** SCP가 풀리면 `db_password`와 `db.py`의 password 인자를 지우고 토큰
생성으로 되돌린다. 절차는 인프라 레포 `docs/runbook.md`의 "SCP 차단" 절.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    #: 실제로 TCP 연결을 맺을 곳. Lambda에서는 RDS 엔드포인트다.
    db_host: str
    db_port: int
    db_name: str
    db_user: str

    #: embedder 전용 DB 사용자의 비밀번호. 마스터 비밀번호가 아니다.
    #:
    #: 원래는 이 필드가 없었다 -- RDS IAM 인증으로 비밀번호 자체가 필요 없는 설계였다.
    #: 조직 SCP가 rds-db:connect를 거부해 임시로 되돌린 상태다. SCP가 풀리면 이 필드와
    #: db.py의 password 인자를 함께 지우고 토큰 생성으로 돌아간다.
    #: 자세한 경위와 원복 절차는 인프라 레포의 docs/runbook.md "SCP 차단" 절에 있다.
    db_password: str

    #: IAM 인증 토큰에 서명할 호스트. 보통 db_host와 같고, SSM 포트 포워딩으로 로컬에서
    #: 돌릴 때만 갈린다 -- 그때 연결은 localhost로 하지만 토큰은 RDS 엔드포인트로 서명해야
    #: RDS가 받아준다.
    #:
    #: 지금은 비밀번호 인증이라 쓰이지 않는다. 원복할 때 필요하므로 남겨 둔다.
    db_auth_host: str

    #: Lambda 기본값은 verify-full이다. RDS IAM 인증은 TLS를 요구하고, 인증서까지 검증해야
    #: 토큰을 가로챌 중간자가 설 자리가 없어진다. 터널을 쓰는 로컬 실행은 호스트명이 맞지
    #: 않으므로 require로 낮춰야 한다.
    db_sslmode: str
    db_sslrootcert: str

    s3_bucket: str

    embed_dim: int
    batch_size: int
    model_id: str

    #: 임베딩 전에 줄이는 긴 변 길이. DINOv3가 실제로 보는 것은 224px이고 프로세서가 알아서
    #: 줄이므로, 여기서는 디코딩 직후 메모리를 눌러 두는 것이 목적이다. 원본 그대로 배치를
    #: 쌓으면 4천만 화소 몇 장으로 Lambda 메모리가 넘어간다.
    #:
    #: 미리보기 파생본도 이 크기로 나간다. 모델 입력에는 영향이 없으므로(프로세서가 224로
    #: 다시 줄인다) 화질이 아쉬우면 올려도 되지만, 배치 하나가 메모리에 올리는 픽셀 수가
    #: 제곱으로 늘어난다.
    resize_long_edge: int

    #: 파생본 JPEG 품질. 82는 1024px에서 확대하지 않는 한 열화가 눈에 띄지 않으면서
    #: 장당 200KB 안팎으로 떨어지는 지점이다.
    preview_quality: int

    #: Lambda 타임아웃 앞에서 멈출 여유(초). 다음 배치를 시작하기 전에 "지금까지 가장 오래 걸린
    #: 배치 시간 + 이 값"보다 남은 시간이 적으면 배치 경계에서 멈추고 commit한다. 하드 킬은
    #: 진행 중이던 배치를 롤백시키고 아무 결과도 남기지 않으므로, 그 전에 스스로 멈추는 편이
    #: 낫다. 로컬 CLI에는 데드라인이 없어 쓰이지 않는다.
    stop_margin_seconds: int = 60

    #: 원본 GET을 미리 걸어 두는 스레드 수. 한 배치를 CPU가 다듬는 동안 다음 배치의 원본을 이만큼
    #: 동시에 내려받는다. 5~13MB 원본을 한 장씩 순차로 받으면 네트워크를 기다리는 동안 CPU가 놀고,
    #: CPU가 일하는 동안 회선이 논다 -- 2026-09-05 로컬 E2E에서 장당 1.7초의 대부분이 이 대기였다.
    #: 미리 받아 두는 창은 두 배치 앞(job.PREFETCH_BATCHES)이라 메모리의 원본은 세 배치(24장, ~300MB)를 넘지 않는다.
    download_workers: int = 4

    #: 빌드 시 내려받은 모델 snapshot과 런타임 로드를 같은 immutable commit으로 묶는다.
    model_revision: str = "5931719e67bbdb9737e363e781fb0c67687896bc"
    #: 갤러리 샤딩(#56). wes 가 부른 실행(조정자)은 대상 사진 수를 이 값으로 나눈 만큼(최대 max_shards)
    #: 자기 함수를 동시에 띄우고 끝난다. 장당 0.78s 라 150장 = 약 2분. 250 에서 150 으로 낮춘 이유(#59): 벽시계는 가장 긴
    #: 샤드가 정하는데 Lambda 호스트 편차로 한 샤드가 1.7배 느린 일이 있었다 — 샤드가 짧을수록 그 피해 폭이 준다(콜드 스타트 +2회).
    #: 0 이면 샤딩하지 않는다. Lambda 예약 동시성이 max_shards 이상이어야 샤드가 스로틀되지 않는다.
    shard_photos: int = 150
    #: 샤드 상한 32(#63)의 근거는 RDS 커넥션 — 샤드는 세션 advisory lock 으로 커넥션 1개를 끝까지 붙들고, db.t4g.micro(79)에서
    #: 평상시 24 + 32 = 56 이 여유 20 의 한계다. 인프라 예약 동시성(embedder_reserved_concurrent_executions)도 같은 값이어야 한다.
    max_shards: int = 32

    @staticmethod
    def from_env() -> "Settings":
        db_host = _required("DB_HOST")

        return Settings(
            db_host=db_host,
            db_port=int(os.environ.get("DB_PORT", "5432")),
            db_name=_required("DB_NAME"),
            db_user=_required("DB_USER"),
            db_password=_required("DB_PASSWORD"),
            db_auth_host=os.environ.get("DB_AUTH_HOST") or db_host,
            db_sslmode=os.environ.get("DB_SSLMODE", "verify-full"),
            db_sslrootcert=os.environ.get("DB_SSLROOTCERT", "/opt/rds-ca/global-bundle.pem"),
            s3_bucket=_required("S3_BUCKET"),
            embed_dim=int(os.environ.get("EMBED_DIM", "768")),
            batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "8")),
            model_id=os.environ.get("EMBED_MODEL_ID", "facebook/dinov3-vitb16-pretrain-lvd1689m"),
            resize_long_edge=int(os.environ.get("RESIZE_LONG_EDGE", "1024")),
            preview_quality=int(os.environ.get("PREVIEW_QUALITY", "82")),
            stop_margin_seconds=int(os.environ.get("STOP_MARGIN_SECONDS", "60")),
            download_workers=int(os.environ.get("EMBED_DOWNLOAD_WORKERS", "4")),
            model_revision=os.environ.get(
                "EMBED_MODEL_REVISION",
                "5931719e67bbdb9737e363e781fb0c67687896bc",
            ),
            shard_photos=int(os.environ.get("SHARD_PHOTOS", "150")),
            max_shards=int(os.environ.get("MAX_SHARDS", "32")),
        )


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        # 초기화 시점에 죽는 편이 낫다. 늦게 발견하면 이미 사진 절반을 처리한 뒤다.
        raise RuntimeError(f"환경변수 {name}이(가) 비어 있습니다")
    return value
