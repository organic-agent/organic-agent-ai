# score — 사진별 점수 Lambda

갤러리 하나의 **모든 사진**에 CLIP ViT-L/14 · ARNIQA · LAION 미학을 돌려 `photo_analysis`에 원점수 ·
피사체 · `clip_embedding` 을 적재한다. torch 가 필요한 갤러리 전수 추론이라 이 repo(Lambda)의 일이다.
그룹·이름은 다음 칸인 [`categorize/`](../categorize/README.md), 미리보기·DINOv3 는 앞 칸인 [`embedder/`](../embedder/README.md).

```
[GPU 워커]  photo_analysis SKIP LOCKED 32장 집기 → 점수 → commit          ← 운영 기본 (인스턴스가 있을 때)
[Lambda ]  wes ──EVENT {galleryId, photoIds}──▶ [score]                   ← 폴백 (워커가 없거나 못 따라갈 때)
```

잡 상태·categorize 호출·재시도는 **wes 가 소유한다**(#98, wes V16). score 는 어느 경로로 불려도 점수만 쓴다.

## 무엇을 계산하나 (`service/pipeline.py`)

```
사진마다 (embedder 가 만든 미리보기 JPEG 입력)
  LaionRunner.embed         → CLIP 768d (저장: photo_analysis.clip_embedding — categorize 가 재계산 없이 읽는다)
     ├─ score_from_embedding → aesthetic_score (LAION MLP)
     └─ SubjectsTagger.tag   → subjects (bride|groom|couple|group|unknown, margin < 0.01 → unknown)
  ArniqaRunner.score        → technical_score (spaq 회귀기)
  classical.measure         → sharpness · highlight_clip · shadow_clip · mean_luma · is_bw
  FaceRunner.measure        → face_n · face_measured · eye_blink · eye_smile · smile (주인공 얼굴의 눈 감김 · 웃음, #164)
  FaceValueRunner.predict   → GPU 이미지는 같은 값을 MediaPipe 대신 DINOv3 + LoRA 측정기로 잰다 · face_src · faces (#170)
→ store.write_scores  (subjects · sub_scores · clip_embedding · pipeline_version 만)
```

CLIP 벡터 하나를 뽑아 미학·피사체·컨셉 라벨 세 가지에 쓴다(텍스트 프롬프트는 추가 비용 0). 백분위·연사·그룹은
categorize 의 컬럼이라 UPSERT 의 SET 절에 없다 — 이 경계가 곧 두 Lambda 의 경계다.

## 실행 모양 — embedder 와 같다

| | |
|---|---|
| 진입점 | `controller/handler.py`(Lambda EVENT `{"galleryId", "photoIds"}` — 다른 페이로드는 에러) / `__main__.py`(CLI, 갤러리 전체 가능) → `service/job.py` 의 `run()` |
| 단위 · 재개 | 사진. 같은 `PIPELINE_VERSION` 이고 CLIP 이 저장된 사진은 건너뛴다. `write_batch`(32)장마다 commit |
| 속도 손잡이 | 한 장은 한 번만 디코드해 세 러너에 넘긴다. CLIP 은 `CLIP_BATCH`(8)장씩 한 forward, ARNIQA 입력 긴 변은 `ARNIQA_LONG_EDGE`(1024 — 1600 대비 연산 1/2.4, 순위 상관 0.93) (#51) |
| 데드라인 | 15분 앞에서 배치 경계에서 멈추고(`STOP_MARGIN_SECONDS`) commit 한다. 남은 사진은 wes 스윕이 다시 보낸다 |
| 실패 | 사진 단위 결정적 실패는 `photo_analysis.error` — 미리보기 없음(S3 404) `PREVIEW_MISSING`, 점수 실패 `SCORE_FAILED`(#85). wes 가 기대 장수에서 뺀다 |
| 잡·체인·샤딩 | **없다**(#98). wes 가 `analysis_jobs` 를 소유하고 categorize 를 직접 부른다. 갤러리 advisory lock·자기 재호출·조정자도 함께 사라졌다 |
| 접속 | `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE`, `S3_BUCKET`(미리보기), Lambda 는 `/tmp` 만 쓴다(`SCORE_WORK`) |

## 구조

패키지는 층으로 나뉘어 있다(#130, embedder #121 과 같은 모양). 의존은 한 방향이다 — controller → service → repository,
service → infrastructure, 모두가 domain·config 를 본다. torch 는 infrastructure 에서만 import 한다.

```
score/
├── score/
│   ├── __main__.py            CLI (python -m 규약상 루트): --gallery-id N [--photo-ids] [--force] | --local "갤러리" | --list | worker --gpu | train
│   ├── controller/            handler.py(Lambda: {galleryId, photoIds} 하나, 데드라인 앞 배치 경계 정지) · sagemaker.py(SageMaker train — GPU 벤치마크 전용)
│   ├── service/               job.py(photoIds 경로 = 운영 폴백 / 갤러리 경로 = 로컬·벤치마크, 잡·체인 없음 #98)
│   │                          · pipeline.py(SCORE 본체 — 위 그림. Scorer 러너 1회 로드 · 디코드 1회 · CLIP/ARNIQA 배치 · 배치 쓰기 · 데드라인 정지)
│   │                          · worker.py(GPU 워커 루프 #75: photo_analysis SKIP LOCKED 32장 집기 → 점수 → commit, 유휴면 자기 정지)
│   │                          · classical.py(Laplacian 선명도 · 노출 클립 · 평균 밝기 · 흑백 여부) · subjects.py(CLIP zero-shot — SubjectsTagger)
│   ├── domain/                photo.py(PhotoRef · PhotoAnalysis) · run.py(ScoreResult) · errors.py(PREVIEW_MISSING · SCORE_FAILED) — 로직 없음
│   ├── repository/            connection.py(Postgres 접속) · photos.py(대상 조회: load_db · load_by_ids) · store.py(LocalStore(out/v3/) · DbStore — write_scores · claim_batch · write_errors)
│   │                          · storage.py(S3 미리보기 · download_previews, 404 → missing) · dataset.py(로컬 데이터셋 폴더: list_galleries · load_local)
│   ├── infrastructure/        runners/(ArniqaRunner — torch.hub SHA 고정 · LaionRunner — open_clip + MLP · FaceRunner — YuNet + MediaPipe, torch 없음 · FaceValueRunner — DINOv3 + LoRA) · device.py(cuda → mps → cpu · fp16 autocast)
│   │                          · images.py(PIL 로드 — load_image · fit_long_edge · as_image) · ec2.py(IMDS · StopInstances) · gpu.py(NVML 사용률 표본)
│   ├── assets/                face_values_lora.pt(얼굴 값 측정기의 LoRA · 선형층, 1.2MB — `scripts/pack_face_lora.py` 가 만든다)
│   └── config/settings.py     Settings · Knobs · PIPELINE_VERSION · MODULE_ROOT
├── tests/test_score.py   pytest 59 — 재개 · 컬럼 경계 · 배치/데드라인 · CLIP/ARNIQA 배치·실패 격리 · 프리페치 · 집기(SKIP LOCKED·error) · GPU 워커 루프 · photoIds 폴백 · handler 계약 · categorize 와의 상수 일치
├── Dockerfile · deploy.sh   컨테이너 Lambda (가중치 빌드 시 번들, CMD `score.controller.handler.handler`) · ECR 푸시 + update-function-code
├── Dockerfile.gpu           GPU 워커·벤치마크 이미지 (cu121 torch). main 의 score/** 변경마다 CI 가 ECR :gpu(이동) + :gpu-<sha>(불변) 로 민다(#77)
├── scripts/sagemaker_benchmark.py   SageMaker training job 제출·대기·로그 요약 · ec2_benchmark.py  EC2 stop/start 실측 · snapshot_scores.py  점수 스냅샷·비교(fp16 검증)
├── scripts/compare_local.py         CPU 경로 회귀(#89): 두 커밋을 같은 venv 로 로컬 데이터셋 N장 돌려 점수·CLIP 벡터 비트 동일 검사 — `check "dataset1/데이터셋1" --rev main --limit 16`
└── requirements.txt · pyproject.toml
```

## 로컬 실행

```bash
cd score && uv venv --python 3.12 .venv                       # Python 3.12 — torch 2.4.1 핀은 3.13+ 에 없다
uv pip install --python .venv/bin/python -r requirements.txt -r ../categorize/requirements.txt   # 맥은 torch 가 MPS 빌드
# 리눅스는 CUDA 를 피해 먼저: uv pip install --python .venv/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python --no-deps -e . -e ../categorize   # 한 venv 에 둘 다 (체인용)
.venv/bin/python -m pytest -q

.venv/bin/python -m score --list                                   # 로컬 데이터셋 갤러리 목록
.venv/bin/python -m score --local "dataset1/데이터셋1" [--limit 50]  # out/v3/<갤러리>/ 에 점수 (categorize --local 이 이어 읽는다)

export DB_HOST=localhost DB_PORT=5432 DB_NAME=wes DB_USER=wes DB_PASSWORD=wes DB_SSLMODE=disable S3_BUCKET=<버킷>
.venv/bin/python -m score --gallery-id 12 [--force] [--limit N]     # 갤러리 전체 (로컬·벤치마크)
.venv/bin/python -m score --gallery-id 12 --photo-ids 1,2,3        # 그 목록만 (운영 Lambda 와 같은 경로)
.venv/bin/python -m score worker --gpu --no-idle-stop               # GPU 집기 워커: 큐를 다 비우고 유휴 30s 뒤 종료(EC2 정지 안 함)
```

wes 쪽 스크립트가 이걸 감싼다: `../organic-agent-server/wes/scripts/local-ai.sh <galleryId>`(임베딩 → 점수 → 카테고리), `scripts/gpu/score-worker.sh`(GPU 워커 대역).

- **embedder 가 먼저다.** `gallery.load_db` 는 `preview_key` 가 있는 사진만 고른다(#75 — v2 에서 `status` 는 보지 않는다).
- **`PIPELINE_VERSION` 이 재개 키다.** 러너·전처리를 바꾸면 올린다 — 전 갤러리가 재점수 대상이 된다. categorize 의
  같은 상수와 값이 같아야 하며 테스트가 고정한다.
- `photo_ratings` · `photo_selection_items` 는 읽지 않는다(정책).

## GPU 워커 — 파이프라인 v2 의 운영 모양 (#75)

Lambda 32 샤드 대신 GPU 인스턴스 한 대(또는 몇 대)가 **사진 단위로 집어서** 점수를 낸다. 갤러리를 배정받지 않는다.

```bash
python -m score worker --gpu                               # 컨테이너 기본 CMD (운영)
python -m score worker --gpu --no-idle-stop                # 로컬: 같은 루프인데 EC2 를 정지하지 않고 종료
python -m score worker --gpu --once                        # 배치 하나만 (디버깅)
```

- 집기: `photo_analysis.embedding IS NOT NULL AND clip_embedding IS NULL AND error IS NULL`(+ 미리보기 있음·휴지통 아님) 32장을 `FOR UPDATE OF photo_analysis SKIP LOCKED`
  로 잠근 채 미리보기 다운로드(16 스레드) → CLIP·ARNIQA·classical → `write_scores`(UPSERT + commit = 잠금 해제). 워커가 죽으면 롤백으로 행이 자동 반환된다.
  여러 대가 같은 사진을 집을 수 없고(RDS 에서 확인), 한 갤러리를 나눠 먹어도 된다. status 는 보지 않는다(v2 에서 EMBEDDED 가 사라진다).
- 유휴: 집을 게 없으면 `WORKER_POLL_SECONDS`(3) 대기, 연속 `WORKER_IDLE_STOP_SECONDS`(**30**, 다중 사용자 운영이면 600) 를 넘기면 **루프가 끝난다**.
  그때 인스턴스를 정지할지는 별개다(#103) — 기본은 IMDSv2 로 자기 id·리전을 얻어 `StopInstances`(컨테이너에 AWS_REGION 이 없어 리전을 명시해야 한다), `--no-idle-stop` 이면 정지 없이 종료(EC2 밖).
  `WORKER_IDLE_STOP_SECONDS=0` 이면 끝나지 않는다. 켜는 것·폴백은 wes.
- 실패: 배치가 `WORKER_MAX_CONSECUTIVE_FAILURES`(5)회 연속 실패하면 루프를 끝내고 exit 1 — 같은 오류로 헛돌지 않는다(#81).
- 잡 테이블은 건드리지 않는다 — 완료는 wes 가 데이터로 관측.
- **두 단계(wes #274 2물결, `SCORE_SPLIT`, 기본 끔):** 켜면 1단계(축소 디코드 + CLIP·피사체·미학 — 폴더가 기다리는 것)를 먼저 집고, 1단계 대기가
  비면 2단계(1024 디코드 + ARNIQA·화질 지표 — 추천만 쓰는 것)를 집는다(`claim_quality_batch`, `clip_embedding IS NOT NULL AND quality_scored_at IS NULL`,
  wes 부분 인덱스 `idx_photo_analysis_quality_unscored`). 2단계 집기는 **찜 표시**다(wes V41 `quality_claimed_at`, R-2-2 방식 B): 찜을 찍고 곧바로
  commit 해 계산 중에는 행을 잠그지 않고, 결과는 `FOR UPDATE SKIP LOCKED` 한 문장 배치로 지금 잠글 수 있는 행에만 쓴다(categorize 가 폴더 묶음을
  쓰는 행은 건너뛰고 3초 뒤 다시, 5번까지). 찜이 `WORKER_QUALITY_LEASE_SECONDS`(120)보다 오래되면 다시 집힌다. R-2-1에서는 잠근 채 계산해
  categorize 적재가 16초 → 95초였다. 1단계는 `technical_score` 키를 쓰지 않는다 — categorize 가 그 키·`quality_scored_at` 으로
  "화질 점수 있음"을 읽는다. 2단계는 `write_quality`(sub_scores 병합 + `quality_scored_at`)이고, 실패해도 `error` 에 쓰지 않고 빈 점수로 끝 표시만
  남긴다(쓰면 폴더 대상에서 빠진다). 유휴는 두 대기가 다 비었을 때. 끄면(또는 Lambda 폴백) 한 번에 다 계산하고 `quality_scored_at` 도 찍는다.
  `sub_scores` 는 categorize 와 키를 나눠 쓰므로 모든 쓰기가 병합(`||`)이다.
- 사진 단위 결정적 실패(#85, wes V15): 미리보기가 S3 에 없으면(404) 그 장만 빼고 `photo_analysis.error='PREVIEW_MISSING'`, 점수 계산에서
  한 장이 실패하면 `'SCORE_FAILED'`. wes 는 그 장을 기대 장수에서 빼고, 집기가 `error IS NULL` 이라 다시 안 집는다(부분 인덱스
  `idx_photo_analysis_unscored` 와 같은 조건). 그 외 다운로드 오류(접속·스로틀)는 배치 rollback 뒤 재시도. Lambda `photoIds` 폴백도 같은 표시.
- 로그: 배치마다 `score worker batch=32 photos=N failed=F seconds=S` 한 줄(wes 합의 key=value 형식).
- 비밀번호: `DB_PASSWORD` env 하나(#91). 인스턴스에서는 호스트의 `deploy/gpu-worker/wes-score-env.sh` 가 SSM 에서 읽어 `--env-file` 로 넘긴다 — 컨테이너는 SSM 을 읽지 않는다.
- Lambda 폴백: `{"galleryId", "photoIds": [...]}` 페이로드는 점수만 쓰고 끝난다(잡·재호출·체인 없음). CLI `--photo-ids 1,2,3`.

## GPU 벤치마크 (#68, 운영 경로 아님)

Lambda 32 샤드 대신 GPU 한 대로 돌리면 얼마나 빠르고 얼마인지 재는 도구. 계획·결과는 `docs/gpu-benchmark-*.md`(로컬 문서).

```bash
# 이미지: GitHub Actions "Build score GPU image" → ECR wes-score:gpu (컨테이너 CMD 는 워커, 벤치마크는 `train` 인자)
.venv/bin/python scripts/sagemaker_benchmark.py setup                               # 실행 역할 (한 번)
.venv/bin/python scripts/sagemaker_benchmark.py run --gallery-id 7 --force          # ml.g4dn.xlarge, fp16, clip 32 · arniqa 8 · decode 4
.venv/bin/python scripts/sagemaker_benchmark.py run --gallery-id 7 --force --no-fp16 --instance ml.g6.xlarge
.venv/bin/python scripts/ec2_benchmark.py setup && .venv/bin/python scripts/ec2_benchmark.py launch --instance g6.xlarge   # DLAMI + 이미지 pull → 정지
.venv/bin/python scripts/ec2_benchmark.py run --gallery-id 8 --force        # StartInstances → 잡 → StopInstances 타임라인 (2026-09-07: 233s, 콜드 16s)
DB_HOST=localhost DB_PORT=15432 … .venv/bin/python scripts/snapshot_scores.py dump 7 cpu-g7.json     # force 전에 CPU 점수 보관
.venv/bin/python scripts/snapshot_scores.py compare cpu-g7.json gpu-g7.json         # Spearman ≥ 0.99 면 같은 모델
```

손잡이(환경변수): `SCORE_DEVICE`(auto) · `SCORE_FP16`(cuda 만) · `CLIP_BATCH` · `ARNIQA_BATCH`(같은 크기끼리만 묶임) ·
`SCORE_DECODE_WORKERS`(디코드·classical 프리페치 스레드, Lambda 는 0) · `SCORE_DOWNLOAD_WORKERS`(S3 동시 다운로드, 기본 8) ·
`SCORE_FACE_VALUES`(얼굴 값. GPU 워커 · 로컬은 기본 켬, **Lambda 는 기본 끔** — MediaPipe 가 Lambda 안에서 프로세스를 강제 종료한다(#166).
0 이면 얼굴 키를 쓰지 않는다. 그래서 폴백으로 점수를 낸 사진은 얼굴 값이 빈다) ·
`SCORE_FACE_SOURCE`(얼굴 값을 재는 모델. 기본 `mediapipe`, GPU 이미지는 `lora`).
CPU 경로는 이 손잡이가 기본값일 때 이전과 비트 단위로 같다(16장 검증).

## 배포

`score/deploy.sh` — 이미지를 빌드해 ECR(`wes-score`)에 밀고 Lambda(`wes-score`) 코드를 갱신한다. 가중치(CLIP · LAION MLP ·
ARNIQA hub)는 빌드 시 `/opt` 아래에 굽는다 — NAT 없는 서브넷이라 런타임 다운로드가 안 된다. **인프라(함수·ECR·VPC
엔드포인트)는 아직 없다** — `../organic-agent-infra` 후속. 실행 역할에 필요한 것: S3 읽기, `lambda:InvokeFunction`(자기
함수 + categorize), Lambda 인터페이스 VPC 엔드포인트. 메모리는 CLIP-L + ARNIQA 로 6–8GB 권장, 실측 전.

## wes 가 읽는 계약

`photo_analysis.subjects` · `sub_scores{technical_score, aesthetic_score, sharpness, highlight_clip, shadow_clip, subjects_margin}` · `clip_embedding` · `pipeline_version`. 키 이름을 바꾸면 wes·categorize 와 함께 바꾼다.

얼굴 값과 흑백 여부(#164)도 `sub_scores` 에 쌓는다. categorize 가 추천 기본 점수(`reco_base`)와 연사 대표를 정할 때 읽는다(categorize #168).

| 키 | 뜻 |
|---|---|
| `face_n` | 주인공 얼굴 수(0~2). 키 큰 순 둘 중 가장 큰 얼굴의 절반 이상인 것 |
| `face_measured` | 그중 값을 잰 얼굴 수. 옆얼굴 · 작은 얼굴은 재지 못한다 |
| `eye_blink` | 가장 많이 감긴 얼굴의 눈 감김(0~1) |
| `eye_smile` | 그 얼굴의 웃음(0~1) |
| `smile` | 주인공 얼굴의 웃음 평균(0~1) |
| `is_bw` | 흑백 사진이면 1 |

- 값은 화질 점수와 같은 단계에서 잰다. 두 단계로 나눠 돌 때(GPU 워커)는 2단계다 — 폴더가 기다리는 1단계에는 넣지 않았다.
### 얼굴 값을 재는 모델 — MediaPipe → DINOv3 + LoRA 측정기 (#170)

얼굴 값은 처음에 MediaPipe 로 쟀다(#164). MediaPipe 는 옆얼굴에서 두 가지로 틀린다 — 랜드마크를 못 찍어 값을 못 내거나(갤러리에 따라
주인공 얼굴의 12~32%), 값을 내더라도 웃음이 낮게 나온다(크게 웃는 옆얼굴이 0.2 대). 웃음 값이 낮으면 웃느라 감은 눈이 결함으로 읽힌다.
그래서 **우리 측정기를 학습했고, GPU 워커는 그 측정기 하나로 잰다.**

| | MediaPipe (`face_source=mediapipe`) | LoRA 측정기 (`face_source=lora`) |
|---|---|---|
| 어디서 | 로컬 기본값 | GPU 이미지 (운영) |
| 방법 | 얼굴 크롭 → 랜드마크 478개 → 블렌드셰이프 | 얼굴 크롭 → DINOv3 + LoRA → 값 2개 |
| 옆얼굴 · 가린 얼굴 | 값 없음이거나 웃음이 낮게 나온다 | 값을 낸다 |
| 옆얼굴의 웃는 얼굴을 웃는다고 함 | 13% | 77% |
| 깨끗한 정면 얼굴 | 기준(교사) | 교사와 0.10 쯤 다르다 |
| 도는 곳 | CPU (디코드 스레드) | GPU (묶음 추론) |

- 학습 정답: 눈 감김과 정면의 웃음은 MediaPipe 값이다. 정면이 아닌 얼굴의 웃음은 VLM 판정이다(웃지 않음 0 · 미소 0.65 · 크게 웃음 0.9, 얼굴 2,000개).
  판정 기준서는 사람이 얼굴 100개를 직접 판정한 것과 맞췄다 — 웃는지 아닌지가 94% 같다.
- 입력을 일부러 망가뜨려(작게 · 흐리게 · 가려서) 학습했다. 작거나 가린 얼굴에서는 MediaPipe 가 겨우 낸 값보다 정확하다.
- 확인(학습에 안 쓴 갤러리): 웃는 얼굴을 웃는다고 한 비율이 정면 75% → 82%, 옆얼굴 13% → 77%. 안 웃는 얼굴을 웃는다고 한 것은 31개 중 1개.
  폴더 추천은 같은 수준이다(추천 중 애매하게 감긴 눈 9.7% 대 9.8%, 맞힌 장면 30/44 동일).
- 키는 그대로다(`eye_blink` · `eye_smile` · `smile` — 같은 뜻 · 같은 눈금. 옆얼굴의 웃음만 제 크기로 나온다). categorize 와 wes 는 바꿀 것이 없다.

| 키 | 뜻 |
|---|---|
| `face_src` | 값을 잰 모델 — `mp` · `lora` · `null`(값 없음) |
| `faces` | 얼굴마다 `{"box": [x, y, w, h], "blink", "smile", "src"}`. 상자는 사진 대비 비율이다 |

- `faces` 를 남기는 이유: 사진을 다시 검출하지 않고 얼굴 크롭을 만들어 측정기를 다시 학습할 수 있다.
- GPU 이미지만 `lora` 다(백본을 빌드 때 굽는다). 측정기를 못 올리면 MediaPipe 로 물러난다. Lambda 폴백은 얼굴 값을 재지 않는다(#166).
- 추론이 죽으면 그 묶음의 얼굴은 값 없이 남는다. 사진은 실패가 아니다.
- 다시 학습하면 `scripts/pack_face_lora.py <산출물.pt> <버전>` 으로 `assets/face_values_lora.pt` 를 갈아 끼운다. 러너는 `peft` 없이
  백본 가중치에 LoRA 를 직접 더한다(실험의 peft 추론과 값이 같다).
- 한계: 얼굴이 아닌 크롭에도 숫자를 낸다. 확인한 갤러리가 하나다. "웃지 않는데 반쯤 감긴 눈"을 찾는 비율은 재지 못했다(시험 세트에 거의 없다).
- MediaPipe 는 물러날 자리로 이미지에 남겨 둔다. 운영에서 측정기가 자리 잡으면 의존성을 뺀다.

- 재지 못한 값은 `null` 로 쓴다(키를 빼면 병합 때 이전 값이 남는다). `face_n` 이 `null` 이면 러너가 죽은 것, 0 이면 얼굴이 없는 것.
- 이 변경 전에 점수를 낸 사진에는 키가 없다. `PIPELINE_VERSION` 은 올리지 않았다 — 올리면 전 갤러리가 다시 계산된다.
- 값은 미리보기를 줄이는 방식에 민감하다. 같은 사진을 조금 다르게 줄이면 사진의 5~9% 에서 값이 0.05 넘게 달라진다(실측).
설계 근거·역사는 `docs/photoselect/`(review-v3-design.md, pipeline-history.md).
