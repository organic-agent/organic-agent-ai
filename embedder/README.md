# embedder

갤러리 하나의 사진을 DINOv3(ViT-B/16)로 임베딩해 `photo_analysis.embedding`(pgvector `vector(768)`)에
적재한다. 이 벡터는 앱의 클러스터링과 AI 셀렉(`photoselect`)이 함께 읽는다 — 두 소비자가 같은
DINOv3 벡터를 본다.
Lambda로 배포되지만 **로컬에서도 같은 코드가 그대로 돈다** — 진입점만 다르다.

```
controller/handler.py   Lambda      {"galleryId": 1, "photoIds": [101, …]} → controller/embed.py | {"jobId": …} → controller/admin.py
__main__.py             로컬 CLI    python -m embedder --gallery-id 1 [--photo-ids 1,2,3]
     └─────── embed 와 __main__ 둘 다 service/job.run() 하나를 부른다
```

패키지는 층으로 나뉘어 있다(#121). 의존은 한 방향이다 — controller → service → repository, 그리고 모두가 domain 을 본다.
어드민(관리자 사진 교체)과 일반(배치 임베딩)은 층마다 파일이 따로다(#123) — 어드민은 `controller/admin.py → service/admin_job.py →
repository/admin_jobs.py → domain/admin.py` 로 따라가면 된다.

```
embedder/
├── __main__.py        로컬 CLI (python -m 규약상 루트)
├── controller/        handler.py(Lambda 진입·분기만) · embed.py(일반) · admin.py(어드민, 페이로드 검증 → AdminPhotoEvent)
├── service/           job.py(배치 본체) · admin_job.py(한 장 교체) · images.py(다듬기) · metadata.py(EXIF)
├── domain/            photo.py(PhotoRef · PhotoMetadata · EmbeddingResult) · run.py(RunResult) · admin.py(AdminPhotoEvent · AdminJobResult) — 로직 없음
├── repository/        connection.py(접속) · photos.py(일반) · admin_jobs.py(어드민) · storage.py(S3, 공유)
├── infrastructure/    model.py(DINOv3 로드·추론, torch 는 여기서만)
└── config/            settings.py(환경변수 → Settings)
```

## 흐름

```
SELECT id, storage_key FROM photos p
WHERE gallery_id = ? AND status <> 'PENDING'
  AND NOT EXISTS (SELECT 1 FROM photo_analysis a WHERE a.photo_id = p.id AND a.embedding IS NOT NULL)
  → S3 GET(스레드 풀이 두 배치 앞서 미리 받는다) → 원본 열기 → EXIF 읽기(촬영 시각 · 카메라 · 셔터/조리개/ISO · 크기)
           → 축소 디코드(JPEG 1/2·1/4) · 리사이즈 · EXIF 회전 → JPEG 인코딩
  → S3 PUT previews/{원본키}.jpg                                  ← ① 먼저 올린다
  → 올린 JPEG 바이트를 다시 열어 DINOv3(L2 정규화)                    ← ② 그 파일로 임베딩
  → INSERT INTO photo_analysis (photo_id, embedding, embedding_model) ... ON CONFLICT DO UPDATE
    UPDATE photos SET preview_key = ?, taken_at = ?, ...          (같은 트랜잭션, 8장 배치마다 commit — status 는 쓰지 않는다)
```

**순서가 계약이다(#24).** 벡터는 메모리의 중간 이미지가 아니라 **S3에 실제로 올라간 미리보기
파일**에서 계산한다. 그래서 "미리보기는 없는데 벡터는 있는" 사진이 구조적으로 생길 수 없고,
**운영은 이 경로 하나다(#73·#100)**: wes 스위퍼가 `{"galleryId", "photoIds": [50장]}` 으로 부르면 그 목록만 처리한다 —
대상 조회·잠금·샤딩·재호출 없음(배정은 wes 의 `photos.dispatched_at` 이 원자적으로). 위 SELECT 는 로컬 CLI 가 목록 없이
부를 때만 돈다(wes `scripts/local-ai.sh`). `photos.status` 는 쓰지 않는다 — V15 부터 그 값은 PENDING·UPLOADED 뿐이고
임베딩 여부는 `photo_analysis.embedding` 이 말한다(embedder 역할에 status UPDATE 권한도 없다).

photoselect가 보는 픽셀과 벡터가 같은 파일이라는 점도 따라온다. 대가는 1024px JPEG를 한 번 더 디코드하는 장당 수십 ms다.

**속도는 원본 GET이 정한다.** 5~13MB 원본을 한 장씩 받고 다듬기를 번갈아 하면 네트워크와 CPU가 서로를
기다린다 -- 2026-09-05 로컬 E2E(822장)에서 장당 1.7초의 대부분이 이 대기였다(`docs/local-e2e-2026-09-05.md`).
그래서 GET은 `EMBED_DOWNLOAD_WORKERS`(기본 4) 스레드가 두 배치 앞서 미리 받아 두고, 디코드는 JPEG 축소
디코드(`images.prepare`, 긴 변 1536px 이상을 남기는 1/2·1/4)로 원본을 통째로 풀지 않는다. 처리·PUT·commit 순서는
그대로 메인 스레드가 한 장씩 밟는다.

- **벡터는 `photo_analysis`에, 나머지는 `photos`에.** 벡터는 모델을 바꾸면 다시 적는 파생값이라
  업로드 때 정해지는 정체성(EXIF)과 테이블을 나눴다(V29). 같은 행에 AI 분석 배치가
  태그·점수·클러스터를 채우므로 여기서는 `embedding`·`embedding_model`만 갈아 끼운다.
- **재실행이 안전하다.** 기본 조건이 "벡터 없음"이라 중간에 죽어도 다시 부르면 남은
  것만 이어서 한다. 한 장이 실패해도 잡을 죽이지 않고 `failed`에 키만 모아 돌려준다.
  원본을 못 읽었든, 미리보기 PUT이 실패했든, 올린 파일을 못 열었든 전부 같은 `failed`다 —
  어느 쪽이든 벡터가 없으므로 다음 호출이 자연히 다시 집는다. IAM에 `s3:PutObject`가 없으면
  사진마다 `failed`로 떨어지고 벡터도 적재되지 않는다. `failed`가 대상 수와 같으면 임베딩이
  아니라 권한을 봐야 한다.
- **타임아웃 앞에서 스스로 멈춘다.** Lambda는 다음 배치를 시작하기 전에 남은 시간이
  "지금까지 가장 오래 걸린 배치 + `STOP_MARGIN_SECONDS`"보다 적으면 배치 경계에서 멈추고
  commit한다. 결과에 `stopped: true, remaining: N`이 실린다. **자기 재호출은 하지 않는다(#100)** —
  남은 사진은 wes 스위퍼가 `dispatched_at` 을 되돌려 다시 배정한다. 50장 배치는 이 지점에 가지 않는다.
  하드 킬을 당하면 진행 중이던 배치만 롤백된다. 로컬 CLI에는 데드라인이 없다.
- **잠금·샤딩·조정자가 없다(#100).** wes 의 배정(`dispatched_at`)이 원자적이라 같은 사진이 두 번 배정되지 않는다.
  병렬은 wes 가 배치를 동시에 여러 개 띄우는 것으로 얻는다(`embed-max-in-flight` 32, 인프라 예약 동시성과 같은 값).
- **미리보기 파생본도 여기서 만든다.** 임베딩을 하려면 어차피 HEIC를 디코딩하고 EXIF 회전을
  굽고 크기를 줄여야 하는데, 그 결과가 그대로 브라우저가 그릴 수 있는 이미지다. 남은 일은
  JPEG 인코딩과 PUT 하나뿐이라 별도 잡으로 뺄 이유가 없다 — 빼면 같은 이미지를 두 번 받아
  두 번 디코딩하게 된다. 아이폰 원본(HEIC)은 Chrome·Firefox·Edge가 그리지 못하므로
  이 파생본이 미리보기의 유일한 통로다.
- **촬영 정보(EXIF)도 여기서 읽는다.** 앱 서버는 이미지 바이트를 만지지 않으므로 EXIF를 읽을
  방법이 아예 없고, 이 잡은 이미 원본을 열어 두었다. 추가 비용은 태그를 훑는 것뿐이라 벡터·
  파생본과 같은 UPDATE에 얹는다. 읽는 값은 촬영 시각·카메라 제조사와 모델·셔터·조리개·ISO·
  가로세로·바이트 크기이고, 컬럼은 전부 nullable이다 — 스크린샷처럼 EXIF가 없는 파일도 있다.
  회전·축소 **전의** 원본에서 읽는다. 그 뒤에는 Orientation 태그가 지워지고 크기도 원본이 아니다.
- **EXIF 추출 실패는 임베딩을 죽이지 않는다.** 실패한 키만 `metadataFailed`로 나온다. 상세
  화면에 정보가 덜 나올 뿐 사진은 보이고 벡터·미리보기는 적재된다. EXIF 컬럼만 `COALESCE`로
  이전 값을 지킨다(`preview_key`는 이번에 올린 파일이 곧 벡터의 원본이라 그냥 덮어쓴다).
- **재계산은 플래그가 아니다(#100).** 모델·전처리를 바꿔 다시 돌리려면 `photo_analysis` 행을 지운다(관리자 재처리) — 그러면 대상 조회에 다시 걸린다.
- **`PENDING`은 건너뛴다.** 업로드 URL만 발급되고 S3에 객체가 없을 수 있는 상태다.

## 접속: 원래는 비밀번호가 없었다

설계는 **RDS IAM 인증**이었다. `boto3`가 만드는 15분짜리 토큰을 비밀번호 자리에 넣는다.
토큰 생성은 로컬 서명 연산이라 네트워크를 타지 않는데, 이 함수가 붙는 서브넷에는 NAT도
인터페이스 엔드포인트도 없으므로 그 점이 결정적이다 — Parameter Store를 읽으려면 시간당
과금되는 엔드포인트가 필요해진다. 부수 효과로 비밀번호가 어디에도 남지 않았다.

**지금은 못 쓴다.** 조직 SCP가 이 계정 전체에서 `rds-db:connect`를 거부한다. 계정은 조직의
멤버 계정이라 여기서 풀 수 없다(SCP는 관리 계정에 적용되지 않으므로, 관리자인데도 막힌다는
것이 곧 멤버 계정이라는 증거다). 그래서 임시로 `DB_PASSWORD` 환경변수를 쓴다.

이 환경변수는 Terraform이 넣지 않는다 — 넣으면 state에 평문으로 남는다. apply 밖에서 한 번
주입하고 `ignore_changes`가 지켜 준다. 주입 명령은 인프라 레포 런북에 있다.

**`GRANT rds_iam`은 지금 있으면 안 된다.** pg_hba가 `hostssl all +rds_iam pam`을 먼저
매칭해서 비밀번호를 보지도 않고 PAM으로 보낸다. 증상은 `PAM authentication failed`다.

전제 조건은 인프라 레포가 책임진다. 원복 절차(SCP를 푼 뒤)도 거기 있다 —
`docs/runbook.md`의 "SCP 차단" 절.

그리고 **DB 안에 사용자를 한 번 만들어 줘야 한다**(Terraform이 못 하는 부분).
인프라 레포 `docs/runbook.md`의 "임베딩 파이프라인" 절을 볼 것:

```sql
CREATE USER embedder;
GRANT rds_iam TO embedder;
GRANT SELECT (id, deleted_at) ON galleries TO embedder;
GRANT SELECT (id, gallery_id, storage_key, status, deleted_at, preview_key,
    taken_at, camera_make, camera_model, exposure_time, f_number, iso, width, height,
    byte_size, version) ON photos TO embedder;
GRANT UPDATE (status, preview_key, taken_at, camera_make, camera_model,
    exposure_time, f_number, iso, width, height, byte_size, dispatched_at, version, updated_at) ON photos TO embedder;
GRANT SELECT, INSERT, UPDATE ON photo_analysis TO embedder;
GRANT SELECT (id, attempt_count, job_type, target_type, target_id, revision_id, status, payload)
    ON admin_processing_jobs TO embedder;
GRANT UPDATE (status, failure_code, last_run_at, updated_at)
    ON admin_processing_jobs TO embedder;
GRANT SELECT (id, photo_id, storage_key) ON admin_photo_revisions TO embedder;
```

`photo_analysis`는 V45에서 생긴 테이블이다(벡터도 `photos.embedding`이 아니라 여기 있다).
V45를 적용한 뒤 이 GRANT를 빠뜨리면 첫 배치의 INSERT가
`permission denied for table photo_analysis`로 실패하고, 사진은 전부 UPLOADED로 남는다.

`galleries` 읽기는 대상 선별(`fetch_targets`)이 휴지통에 들어간 갤러리를 거르는 데 쓴다.
빠뜨리면 임베딩이 `InsufficientPrivilege: permission denied for table galleries`로
전량 실패한다 (2026-08-14 운영에서 실제로 겪었다 — CREATE USER 때 GRANT를 같이 안 하면
그 테이블을 읽는 코드가 배포되는 날 터진다).

V42 migration은 기존 `embedder` role이 있으면 exact-photo job용 권한을 위와 같이 자동
추가한다. migration 뒤에 role을 새로 만드는 환경만 이 SQL을 직접 실행하면 된다.

### 위 SQL을 실행할 관리자 접속

비밀번호는 저장소에 두지 않는다 — Parameter Store에서 실행 시점에 꺼낸다.

| 항목 | 값 |
|---|---|
| Host / Port | `localhost:15432` (`scripts/db-tunnel.sh`로 터널을 연 상태) |
| Database | `wes_db` |
| User | `wes_admin` |
| Password | `aws ssm get-parameter --region ap-northeast-2 --name /wes/prod/spring.datasource.password --with-decryption --query Parameter.Value --output text` |
| sslmode | `require` (터널 때문에 verify-full은 호스트명 검증에 실패) |

로컬에 psql이 없으면 Testcontainers용으로 이미 있는 이미지의 psql을 쓴다:

```bash
scripts/db-tunnel.sh   # 다른 탭에 띄워 두고
PGPASSWORD=$(aws ssm get-parameter --region ap-northeast-2 \
  --name /wes/prod/spring.datasource.password --with-decryption \
  --query Parameter.Value --output text) \
docker run --rm -e PGPASSWORD pgvector/pgvector:pg16 \
  psql "host=host.docker.internal port=15432 dbname=wes_db user=wes_admin sslmode=require"
```

## 로컬 실행

### 로컬 pg + dev 버킷 (개발 기본)

`scripts/local-ai.sh <galleryId>`가 venv 생성부터 임베딩, 이어지는 AI 분석·추천 CLI까지 한 번에
돌린다(`--only-embed`면 여기까지만). DB는 `docker-compose.local.yml`의 pg, 버킷은 인프라가
`/wes/local/app.storage.bucket`에 기록한 dev 버킷(`wes-dev-photos-*`)이다 — 로컬 wes(local 프로필)도
같은 파라미터를 읽으므로 브라우저가 올린 사진을 이 CLI가 그대로 읽는다. 손으로 돌리려면:

```bash
export DB_HOST=localhost DB_PORT=5432 DB_NAME=wes DB_USER=wes DB_PASSWORD=wes DB_SSLMODE=disable
export S3_BUCKET="$(aws ssm get-parameter --region ap-northeast-2 --name /wes/local/app.storage.bucket --query Parameter.Value --output text)"
python -m embedder --gallery-id 1
```

### 공유 RDS (배포 전 검증)

RDS는 퍼블릭 접근이 없으므로 SSM 포트 포워딩으로 터널을 먼저 연다.

```bash
aws ssm start-session --region ap-northeast-2 \
  --target "$(cd ../../../organic-agent-infra && terraform output -raw ec2_instance_id)" \
  --document-name AWS-StartPortForwardingSessionToRemoteHost \
  --parameters '{"host":["<rds-endpoint>"],"portNumber":["5432"],"localPortNumber":["15432"]}'
```

다른 탭에서:

```bash
python -m venv .venv && source .venv/bin/activate
# torchvision까지 받아야 한다. transformers의 fast 이미지 프로세서가 요구한다.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

export DB_HOST=localhost DB_PORT=15432 DB_NAME=wes_db DB_USER=embedder
export DB_PASSWORD="$(aws ssm get-parameter --region ap-northeast-2 \
  --name /wes/prod/embedder.db.password --with-decryption \
  --query Parameter.Value --output text)"
# 터널을 거치면 인증서의 호스트명이 localhost와 맞지 않아 verify-full이 실패한다.
export DB_SSLMODE=require
export S3_BUCKET="$(cd ../../../organic-agent-infra && terraform output -raw photo_bucket)"

python -m embedder --gallery-id 1
```

맥에서는 `torch.backends.mps`가 잡혀 GPU로 돈다. Lambda는 CPU다.

## 빌드와 배포

```bash
./deploy.sh
```

이미지를 만들어 ECR에 올리고, Lambda가 그 이미지를 집게 한 뒤, 실제로 걸렸는지까지 확인한다.
`torch`와 모델 가중치 레이어는 캐시되므로 파이썬 코드만 고쳤다면 1분이 안 걸린다.

**앱과 배포 경로가 다르다.** `cd.yml`은 `embedder/`를 건드리지 않으므로 main에 머지해도 운영
함수는 그대로다. 따로 안 나가는 것이 둘 더 있다:

- **`terraform apply`로도 코드는 안 나간다.** 인프라의 `modules/embedding`에
  `lifecycle { ignore_changes = [image_uri] }`가 걸려 있다. 저장소·함수·IAM만 관리하고 이미지
  태그는 일부러 손대지 않는다 — 그래야 코드를 고칠 때마다 terraform을 돌리지 않는다.
- **ECR에 `:latest`를 푸시하는 것만으로도 안 나간다.** Lambda는 갱신 시점의 다이제스트를 고정해
  둔다. 태그가 새 이미지를 가리켜도 함수는 옛 다이제스트를 계속 실행한다.

**Hugging Face 토큰이 필요하다.** DINOv3는 게이트 모델이라 가중치를 굽는 단계에서 라이선스를
승인한 계정의 토큰을 요구한다. `HF_TOKEN`을 내보내거나 `hf auth login`을 해 두면 스크립트가
빌드 시크릿(`--secret id=hf_token`)으로 넘긴다 — ARG/ENV가 아니라서 이미지 히스토리에 남지
않는다. 토큰이 없으면 빌드를 시작하기 전에 멈춘다. 로컬 실행(`python -m embedder`)도 같은
토큰으로 처음 한 번 받아 `~/.cache/huggingface`에 캐시한다.

스크립트가 쓰는 세 플래그가 전부 필요하다.

- **`--platform linux/amd64`** — 맥에서 빌드하면 기본이 arm64다. Lambda 함수는 x86_64로
  만들어져 있어서 아키텍처가 다르면 실행 시점에 죽는다. (앱 컨테이너는 Graviton EC2에
  올라가 arm64지만, 이 함수는 별개다.)
- **`--provenance=false --sbom=false`** — 이게 없으면 buildx가 attestation이 붙은 **manifest
  list**를 만들고, Lambda는 그걸 거절한다:

  ```
  InvalidParameterValueException: The image manifest, config or layer media type
  for the source image ... is not supported.
  ```

  이미지 내용과 무관한 포장 형식 문제인데 메시지가 그렇게 읽히지 않는다. Lambda는 단일
  아키텍처 Docker v2 매니페스트만 받는다.

**첫 apply는 순서가 있다.** Lambda는 이미지가 없는 ECR을 상대로 만들어지지 않으므로,
리포지토리만 먼저 만들고 → 푸시 → 전체 apply 한다. 인프라 레포 `docs/deploy-order.md` 참고.

## 환경변수

| 변수 | 기본값 | 비고 |
|---|---|---|
| `DB_HOST` / `DB_PORT` / `DB_NAME` / `DB_USER` | — | 필수(포트 기본 5432). Lambda는 Terraform이 주입 |
| `DB_PASSWORD` | — | 필수. **Terraform이 주입하지 않는다** — apply 밖에서 넣고 `ignore_changes`가 지킨다. SCP 우회로용이라 원복 시 사라진다 |
| `DB_AUTH_HOST` | `DB_HOST` | IAM 토큰에 서명할 호스트. 지금은 안 쓰인다(원복용). 터널로 로컬 실행할 때만 달라진다 |
| `DB_SSLMODE` | `verify-full` | 터널로 로컬 실행할 때는 `require` |
| `DB_SSLROOTCERT` | `/opt/rds-ca/global-bundle.pem` | 이미지에 구워져 있다 |
| `S3_BUCKET` | — | 필수 |
| `EMBED_DIM` | `768` | `vector(n)` 컬럼과 `Photo.EMBEDDING_DIMENSION`과 셋이 같아야 한다 |
| `EMBED_BATCH_SIZE` | `8` | 모델에 한 번에 넣는 장수 |
| `EMBED_MODEL_ID` | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 바꾸면 이미지를 다시 빌드해야 한다(가중치가 구워져 있다). 차원이 다른 모델(ViT-S 384, ViT-L 1024)은 `EMBED_DIM`·`vector(n)`·`EMBEDDING_DIMENSION`도 같이 바꿔야 한다 |
| `RESIZE_LONG_EDGE` | `1024` | 디코딩 직후 메모리를 누르는 용도. 미리보기 파생본도 이 크기로 나간다 |
| `PREVIEW_QUALITY` | `82` | 파생본 JPEG 품질. 1024px에서 장당 200KB 안팎 |
| `STOP_MARGIN_SECONDS` | `60` | 타임아웃 앞에서 멈출 여유. 남은 시간 < (가장 긴 배치 + 이 값)이면 배치 경계에서 멈추고 commit — 남은 장은 wes 가 재배정 |
