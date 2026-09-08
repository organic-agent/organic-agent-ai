# GPU 워커 인스턴스 전달본 (#87) — 인프라 `modules/score-gpu` · wes W6 합류용

파이프라인 v2 의 score 실행 모양은 **EC2 g6.xlarge 2대의 워커 루프**(`python -m score worker --gpu`, #75·#79·#81)다. 이 디렉토리는
인프라(`organic-agent-infra/docs/pipeline-v2-infra-plan.md` §4.4·§8)가 Image Builder 컴포넌트에 그대로 넣을 **유닛·스크립트 본문**과,
인스턴스 역할·wes 계약을 한 곳에 모은 것이다. 워커 코드 자체는 `score/gpu_worker.py`, 이미지는 `Dockerfile.gpu`(CI 가 `gpu`·`gpu-<sha>` 태그 push).

| 파일 | 설치 경로 | 역할 |
|---|---|---|
| `wes-score-env.sh` | `/usr/local/bin/` (0755) | SSM 파라미터 3개 → `/run/wes-score.env`(tmpfs, 0600). JDBC URL 에서 host·port·db 를 자른다 |
| `wes-score-gpu-run.sh` | `/usr/local/bin/` (0755) | env → ECR 로그인 → `docker pull …:gpu` → `exec docker run --gpus all --env-file …` |
| `wes-score-gpu.service` | `/etc/systemd/system/` | 부팅마다 run 스크립트. `Restart=on-failure`, 10분 3회 상한, 끝나면 env 파일 삭제 |

인프라 계획 §4.4 골격과 다른 점 — 실제 코드의 이름·값으로 맞췄다:

- 유휴 정지 env 는 `IDLE_STOP_SECONDS` 가 아니라 **`WORKER_IDLE_STOP_SECONDS`** 이고 기본 **30**(#81, 사용자 결정 — 지금은 갤러리를 연달아 처리할
  사용자가 없다). 다중 사용자 운영이 되면 600 으로(인스턴스 `Environment=WORKER_IDLE_STOP_SECONDS=600` 한 줄, AMI 재빌드 불필요).
- `ExecStartPre` 둘 대신 run 스크립트 하나가 env → pull → `exec docker run` 을 한다. 같은 활성화 안에서 `EnvironmentFile` 을 만들고 읽을 수 없어서다.
- 컨테이너 env 에 GPU 손잡이(`SCORE_DEVICE=cuda`, fp16, CLIP 32 · ARNIQA 8 배치, 디코드 4 · 다운로드 16 스레드)를 명시한다 — SageMaker
  ml.g6.xlarge 검증값(장당 0.033s). `SCORE_DEVICE=cuda` 는 GPU 가 안 잡히면 러너 로드에서 일찍 죽으라는 뜻(CPU 로 조용히 떨어지지 않게).
- ECR 레지스트리는 `sts get-caller-identity` 로 계산한다(`ecr:DescribeRepositories` 불필요).

## 인스턴스 요구사항

| 항목 | 값 | 이유 |
|---|---|---|
| AMI | AL2023 + NVIDIA 드라이버 **≥ 525** + docker + nvidia-container-toolkit(`nvidia-ctk runtime configure --runtime=docker`) | 이미지의 torch cu121 이 드라이버만 요구한다. 코드 이미지는 AMI 에 굽지 않는다(결정 B) |
| 부팅 지연 | DLAMI 를 쓴다면 `systemctl mask dkms.service update-motd.service` | 벤치마크 실측: dkms 56s + motd 30s 가 매 부팅에 붙었다(콜드 66s → 16s) |
| IMDSv2 | `http_tokens=required`, **`http_put_response_hop_limit=2`** | 컨테이너 안에서 역할 자격증명·자기 인스턴스 id(#75 `instance_id()`)를 얻는다. hop 1 이면 컨테이너가 IMDS 를 못 본다 |
| 태그 | `Name=wes-score-gpu` | 워커 역할 `StopSelf` 조건 · wes W6 탐색 키 |
| 루트 볼륨 | gp3 30GB | 이미지 4.2GB + docker 레이어 캐시 |
| 네트워크 | 퍼블릭 서브넷 + 공인 IP, 인바운드 0, RDS SG 에 5432 인그레스 | ECR·SSM·S3 는 IGW, RDS 는 VPC 내부 |
| 접속 | SSM Session Manager 만 | 키·SSH 없음 |

## 인스턴스 역할 권한 (인프라 계획 §4.2 와 대조 — 일치, 추가 없음)

| 무엇 | 액션 | 리소스 · 조건 | 누가 쓰나 |
|---|---|---|---|
| 미리보기 읽기 | `s3:GetObject` | `arn:aws:s3:::<photo_bucket>/previews/*` | 컨테이너(`storage.PreviewStorage`) |
| 이미지 pull | `ecr:GetAuthorizationToken`(`*`) · `ecr:BatchGetImage` · `ecr:GetDownloadUrlForLayer` · `ecr:BatchCheckLayerAvailability` | `repository/wes-score` | run 스크립트 |
| 설정 읽기 | `ssm:GetParameter` | `/wes/prod/photoselect.db.password` · `/wes/prod/spring.datasource.url` · `/wes/prod/app.storage.bucket` **세 ARN 만** | env 스크립트(호스트). 컨테이너는 SSM 을 읽지 않는다 |
| 복호화 | `kms:Decrypt` | `*` + `kms:ViaService = ssm.ap-northeast-2.amazonaws.com` | 위 SecureString(지금은 기본 키라 엄밀히 불필요, 고객 키로 바꿔도 안 깨지게) |
| 자기 정지 | `ec2:StopInstances` | `instance/*` + `ec2:ResourceTag/Name = wes-score-gpu` | 컨테이너(`gpu_worker.stop_self`, IMDS 로 자기 id) |
| 접속 | 관리형 `AmazonSSMManagedInstanceCore` | | SSM 세션 |

필요 **없는** 것: `lambda:GetFunctionConfiguration`(벤치마크 시절 env 복사 — 폐기), `ecr:DescribeRepositories`, `ec2:Describe*`, S3 쓰기.
`sts:GetCallerIdentity` 는 항상 허용된다.

컨테이너의 DB 비밀번호는 `--env-file` 로 받는다(인프라 계획 §8). 컨테이너가 SSM 을 직접 읽는 경로(#75 의 `DB_PASSWORD_SSM_PARAM`)는 #91 에서
지웠다 — 비밀을 읽는 주체는 호스트 env 스크립트 한 곳이고, 컨테이너 env 모양은 Lambda 와 같다(`DB_PASSWORD`).

## 켜고 끄는 책임 (인프라 계획 §4.5 네 층 중 이 repo 몫)

- 워커는 **켜지면 일하고 없으면 끈다**. 갤러리를 배정받지 않는다 — `photo_analysis.embedding IS NOT NULL AND clip_embedding IS NULL AND error IS NULL`
  이면 누구 것이든 32장씩 `FOR UPDATE OF photo_analysis SKIP LOCKED` 로 집는다. 2대가 한 갤러리를 나눠 먹어도 중복 0(RDS 에서 확인).
- 유휴 30초(`WORKER_IDLE_STOP_SECONDS`) → 루프 종료. 운영은 그 전에 IMDSv2 로 자기 id 를 얻어 `StopInstances`(#103: 종료와 정지는 별개 — `--no-idle-stop` 은 정지만 건너뛴다) → exit 0. StopInstances 가 실패하면 로그만 남기고 exit 0 — 유닛은
  재시작하지 않고, wes `GpuController`(`idle-stop-after: PT2M`)·CloudWatch 알람(CPU 30분 < 5%)이 끈다.
- wes 쪽 실제 값(#168 머지, `application-variable.yml`): `gpu.enabled`(운영은 SSM `/wes/prod/app.analysis.gpu.enabled`, **지금 false**) ·
  `tag: wes-score-gpu` · `start-grace: PT5M`(이 안에는 유휴로 안 본다) · `idle-stop-after: PT2M` · `fallback-after: PT10M` · `fallback-interval: PT10M`.
  wes 주석도 "워커의 유휴 30초 자기 정지가 1차"로 우리 기본값을 전제한다 — 이 값을 바꾸면 wes 에 알린다.
- 연속 5배치 실패(#81) → exit 1 → 유닛이 30초 뒤 재시작, 10분 3회면 포기. 그 뒤는 알람 몫.
- 켜는 것(`StartInstances`)·폴백(Lambda `{galleryId, photoIds}`) 결정은 wes. **`gpu.enabled=false` 면 워커를 켜지 않고 폴백만 돈다** —
  GPU 배포 전인 지금이 그 상태다(`GpuController.isFallbackDue`: GPU 가 없으면 유예 없이 바로 폴백).

## 검증 절차 (인프라 I2 인스턴스가 나온 날, NEXT.md §1-B)

1. **콜드**: 정지 인스턴스 → `StartInstances` → `journalctl -u wes-score-gpu -f` 로 `[worker] 시작` 까지. 목표 Start → 첫 배치 **≤ 40s**
   (부팅 16 + pull 확인 2 + 러너 로드 9 + 워밍 <1). 대상은 갤러리 8 의 `clip_embedding` 일부를 NULL 로 만들어 준비(`scripts/snapshot_scores.py` 로 원복).
2. **처리**: 로그 `score worker batch=32 photos=N failed=F seconds=S` 가 배치마다 한 줄, 장당 ≈ 0.033s. 끝나면 DB 에 미점수 0.
3. **정지**: 마지막 배치 뒤 30초 ± 5초에 `StopInstances`(CloudTrail 또는 `describe-instances` 상태 `stopping`). env 파일 `/run/wes-score.env` 가 없어졌는지.
4. **2대 동시**: 같은 갤러리 → `photo_analysis` 중복 쓰기 0, 처리 합 = 대상 수. `SELECT count(*) … WHERE clip_embedding IS NULL AND error IS NULL` = 0.
5. **자기 정지 실패**: 역할에서 `StopSelf` 를 잠시 빼고 → 워커가 `StopInstances 실패 … wes 감시에 맡긴다` 로그 뒤 exit 0 → wes W6(있을 때) 30분,
   없으면 CloudWatch 알람이 끄는지. 실측 CPU 사용률(점수 중·유휴)을 기록해 알람 임계값(5%) 확정(인프라 §4.5 보정 항목).
6. **이미지 갱신**: AI main 에 push → CI `gpu` 태그 이동 → 다음 기동의 `pull` 로그가 새 다이제스트를 보이는지(AMI 재빌드 없음).

## wes 에 넘기는 계약 (2026-09-08 현재 AI 쪽 상태)

| 항목 | 상태 |
|---|---|
| embedder 페이로드 `{"galleryId","jobId","photoIds":[…]}` — 목록만 임베딩, 잠금·fan-out 없음 | #74 배포. V15 뒤 status 를 쓰지 않는다(#84, 기본값) |
| score Lambda 폴백 `{"galleryId","photoIds":[…]}` — 점수만, 잡·체인 없음 | #76 배포. 미리보기 없음·실패는 `photo_analysis.error`(#86) |
| GPU 워커 집기 `embedding ∧ ¬clip_embedding ∧ error IS NULL` + `FOR UPDATE OF photo_analysis SKIP LOCKED` | #76·#86. wes 부분 인덱스 `idx_photo_analysis_unscored` 와 같은 조건 |
| 사진 단위 실패 표시 `photo_analysis.error` — `PREVIEW_MISSING` · `SCORE_FAILED` | #86 (score). embedder 의 결정적 실패(디코드 불가) → `error`, 일시 실패 → `dispatched_at=NULL` 은 **아직**(정리 이슈에서, wes 스위퍼 10분 재배정이 그동안 덮는다) |
| 로그 `key=value` 한 줄 | score 워커·폴백 #86. embedder `embedder gallery=G photos=N ok=K failed=F seconds=S` 는 아직 |
| `ai_analysis_jobs` — V16 뒤 Lambda 는 `error` 만 쓴다 | **아직**: score `jobs.record*`(갤러리 경로)·categorize `jobs.py` 가 status·result 를 쓴다. V16 배포 전에 정리 이슈 필요(NEXT.md §1-D) |
| embedder 갤러리 경로(fan-out·자기 재호출) | V15 뒤에도 코드에 남아 있다(옛 오케스트레이터 호환). wes PR-B 배포 뒤 삭제 → 그때 인프라 `ReinvokeSelf` 제거 가능(결정 K) |
