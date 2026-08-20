# photoselect 기술 스택 — 모델·인프라·서버

`feature-design.md`의 각 단계를 구현하는 데 필요한 ML 모델, 패키지, 인프라, 서버 구성의
확정안. 전제는 루트 `CLAUDE.md`의 확정 스택 (Python 3.12, embedder 패턴, ECR 컨테이너
Lambda, 공유 RDS 직접 접근, Bedrock).

---

## 1. 전체 그림 — 컨테이너 하나, 실행 모양 셋

```
photoselect ECR 컨테이너 이미지 (단일)
 ├─ 배치 A: 점수 파이프라인   Lambda EVENT  {"galleryId", "jobId"}   ← 0단계 (갤러리당 1회)
 ├─ 배치 B: 추천 채우기 잡    Lambda EVENT  {"galleryId", "jobId"}   ← 기능 A (잡당 1회)
 └─ 동기 C: 사진 진단 API     Lambda 동기 호출 (wes 경유)            ← 기능 B (요청당)
```

- ML 모델(가중치 포함)은 전부 이미지에 번들 → 세 모양이 같은 이미지를 쓰고 핸들러
  엔트리만 다르다. 배치 B·동기 C는 ML 추론이 없어 모델 로드를 건너뛴다(지연 로드).
- wes(Kotlin/Spring)가 실행 조건 검증·트리거·인증을 소유. 이 서버는 HTTP를 외부에 직접
  노출하지 않는다 (동기 C도 wes 경유).

## 2. ML 모델 스택 (단계 → 모델 → 조달)

| 단계 | 모델 | 패키지 | 가중치 조달 | 라이선스 | CPU 예상* |
|---|---|---|---|---|---|
| 0-1 얼굴/눈/미소 | MediaPipe Face Landmarker | `mediapipe` | `.task` 파일을 이미지에 번들 (공식 배포) | Apache-2.0 (모델 포함) | ~10ms/장 |
| 0-2 기술 품질 | ARNIQA (ResNet-50 + 선형회귀) | `torch`(CPU) + vendored 모델 코드 | [miccunifi/ARNIQA](https://github.com/miccunifi/ARNIQA) 가중치를 빌드 시 고정 커밋으로 다운로드 → 이미지 번들 | Apache-2.0 | ~0.3s/장 |
| 0-3 미학 | LAION Aesthetic v2 (CLIP ViT-L/14 + MLP) | `open_clip_torch` + MLP 가중치 vendored | CLIP: open_clip 공식 / MLP: [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor) in-repo `.pth` | Apache-2.0 | ~0.5s/장 (임베딩 포함) |
| 0-4 장면 태그 | CLIP 제로샷 (0-3 임베딩 재사용) | 상동 | 텍스트 프롬프트 임베딩은 빌드 시 사전 계산해 `.npy`로 번들 → 런타임 텍스트 인코더 불필요 | — | ~0 (내적만) |
| 0-5 클러스터링 | pgvector 임베딩 union-find | `psycopg` + 순수 Python | 모델 없음 (embedder 임베딩 재사용) | — | — |
| 0-6 클러스터 순위 | 결정적 합성 규칙 | 순수 Python (`numpy`) | 모델 없음 | — | — |
| A-2 취향 신호 | CLIP 임베딩 유사도 | `numpy` (DB에서 읽은 벡터) | 모델 없음 | — | — |
| A-4 슬롯 채우기 | 그리디 (후속: DPP) | 순수 Python | — | — | — |
| A-5 / B-2 근거·진단 | Claude (Bedrock) | `anthropic[bedrock]` | API | — | 호출당 |
| (스파이크 예비) | Charm / HSEmotion / 6DRepNet | 각 원본 repo | 골든셋 상관이 부족한 축에만 추가 | Apache-2.0 / MIT | — |

\* 예상치는 가설 — 스파이크에서 실측해 이 표를 갱신한다.

가중치 패키징 원칙:

- **런타임 다운로드 금지** (torch.hub 자동 다운로드 포함) — Lambda 콜드스타트·재현성·공급망
  이유. Dockerfile에서 고정 커밋/체크섬으로 받아 `models/` 디렉토리에 번들.
- 모델 파일마다 `MODEL_VERSIONS` 상수로 버전 명시 → `photo_analysis.model_version`에 기록.

## 3. Python 스택 (requirements)

| 용도 | 패키지 | 비고 |
|---|---|---|
| 런타임 | Python **3.12** | Lambda 컨테이너 베이스 `public.ecr.aws/lambda/python:3.12` |
| 추론 | `torch` (CPU 휠), `open_clip_torch`, `mediapipe`, `numpy`, `pillow` | torch는 `--index-url` CPU 전용 휠로 이미지 슬림화 |
| DB | `psycopg[binary]`, `pgvector` | 공유 RDS 직접 읽기/쓰기 |
| AWS | `boto3` | S3(미리보기)·Parameter Store |
| LLM | `anthropic[bedrock]` | `AnthropicBedrockMantle(aws_region="ap-northeast-2")` |
| 테스트 | `pytest` | 골든셋 리그레션 포함 |
| (최적화 예비) | `onnxruntime` | §7 — torch 추론이 15분 예산을 위협할 때만 |

- 버전은 requirements.txt에 전부 핀 고정. 이미지 크기 목표 < 4GB (Lambda 한도 10GB).

## 4. 인프라 (Terraform — `../organic-agent-infra`)

| 리소스 | 구성 | 비고 |
|---|---|---|
| Lambda (배치 A) | ECR 이미지, **메모리 8~10GB** (vCPU 비례 확보 — CPU 추론이라 메모리=연산력), timeout 15분, VPC 내 | 3,000장 × ~0.8s/장 직렬이면 초과 위험 → 프로세스 풀 병렬 + 실측 |
| Lambda (배치 B) | 같은 이미지, 메모리 1~2GB, timeout 5분 | ML 없음 — DB 조립 + LLM 1회 |
| Lambda (동기 C) | 같은 이미지, 메모리 1~2GB, **provisioned concurrency 소수** | 사용자 대기 — 콜드스타트 회피 |
| S3 | 기존 previews 버킷 읽기 전용 | 원본 접근 불필요 (HEIC 디코드는 embedder 몫) |
| RDS (공유 Postgres) | VPC 보안그룹에 Lambda 인바운드 추가 | 스키마 변경은 전부 wes Flyway |
| Parameter Store | DB 접속 정보 등 시크릿 | `config.py` 지연 로드 단일 통로 |
| IAM | `bedrock:InvokeModel`, `s3:GetObject`(previews), `ssm:GetParameter`, VPC ENI | 최소 권한 |
| ECR | 단일 리포지토리, 태그 = git SHA | |
| CloudWatch | 잡당 구조화 로그(JSON) + 배치 A 처리량 메트릭 | 15분 예산 감시 알람 |

15분 한도 초과 시 대응 순서: ① 병렬화·해상도 조정 → ② ONNX Runtime 전환 → ③ 갤러리
분할 자기 재호출(체크포인트는 `photo_analysis` 자체 — 이미 처리한 사진은 건너뛰는 멱등
설계라 재호출이 곧 재개) → ④ ECS Fargate 이관 (최후).

## 5. LLM (Bedrock) 구성

| 용도 | 모델 | 이유 |
|---|---|---|
| A-5 근거 문장화 (배치) | `anthropic.claude-opus-5` | 품질 우선, 사용자 비대기 — 갤러리당 1회라 원가 허용 |
| B-2 사진 진단 (동기) | 저지연 티어 (예: `anthropic.claude-haiku-*`) — 스파이크에서 품질 확인 후 확정 | 사용자 대기 2~3초 목표 |

- 리전: `ap-northeast-2` 우선, 미제공 모델은 크로스 리전 프로필 `apac.anthropic.…` (스파이크
  최우선 확인 사항).
- 공통: **structured outputs 필수** (photo_id 검증 가드레일의 전제), 시스템 프롬프트(톤
  규칙·근거 코드 사전)는 **prompt caching**으로 고정, 재시도는 멱등 (같은 잡 재실행 시
  기존 근거 덮어쓰기).

## 6. 서버/모듈 구조 (embedder 패턴 준수)

```
photoselect/
├── handler.py        # Lambda 엔트리 3종 분기 (score / draft / diagnose)
├── __main__.py       # 로컬 CLI: python -m photoselect score --gallery-id 1 --job-id 1
├── job.py            # 잡 라이프사이클 (PENDING→RUNNING→COMPLETED/FAILED, 단일 트랜잭션)
├── pipeline/
│   ├── faces.py      # 0-1 MediaPipe
│   ├── quality.py    # 0-2 ARNIQA
│   ├── aesthetic.py  # 0-3/0-4 CLIP + LAION MLP + 장면 태그
│   ├── cluster.py    # 0-5/0-6 union-find + 순위 합성
│   ├── draft.py      # A-1~A-4 그리디 선택
│   └── reasons.py    # A-5 LLM 근거 (+ 템플릿 폴백)
├── diagnose.py       # 기능 B 조회 + LLM
├── db.py             # psycopg, photo_analysis/photo_selection_items 접근 (접근 규칙 강제)
├── config.py         # env/Parameter Store 지연 로드 단일 통로
├── bedrock.py        # AnthropicBedrockMantle 래퍼, structured outputs, 재시도
├── models/           # 번들된 가중치 (.task/.pth/.npy) + MODEL_VERSIONS
├── tests/
├── requirements.txt
└── Dockerfile
```

- `db.py`가 접근 규칙을 코드로 강제한다: `photo_selections.status` 미접근,
  `photo_selection_items`는 `source='AI'`만 쓰기, `photo_ratings` 쿼리 부재.

## 7. 로컬 개발·테스트

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
../organic-agent-server/wes/scripts/db-tunnel.sh        # SSM 포트 포워딩 (기본 15432)
python -m photoselect score --gallery-id 1 --job-id 1   # 0단계 로컬 실행
python -m photoselect draft --gallery-id 1 --job-id 2   # 기능 A 로컬 실행
pytest                                                   # 단위 + 골든셋 리그레션
```

- 골든셋(`../dataset`)은 CI가 아닌 로컬/스파이크에서 실행하는 평가 스크립트로 분리
  (recall@K, 클러스터 대표 일치율, 장면 커버리지).
- LLM 테스트는 recorded fixture 우선, 실호출은 스파이크·검수 시에만.

## 8. 확정 전 확인 목록 (스파이크와 연동)

1. ap-northeast-2 Bedrock 모델 가용성 (진단용 저지연 티어 포함)
2. 배치 A 실측: 3,000장 처리 시간 vs 15분 (메모리 10GB 기준)
3. torch CPU 이미지 크기와 콜드스타트 (동기 C의 provisioned concurrency 산정)
4. ARNIQA·LAION 점수의 골든셋 상관 (모델 표 갱신, 필요 시 Charm/HSEmotion 투입)
5. MediaPipe `.task` 모델의 Lambda(리눅스 arm64/x86_64) 동작 확인 — 아키텍처는 torch 휠
   호환성 기준으로 x86_64 우선
