# score — 사진별 점수 Lambda

갤러리 하나의 **모든 사진**에 CLIP ViT-L/14 · ARNIQA · LAION 미학을 돌려 `photo_analysis`에 원점수 ·
피사체 · `clip_embedding` 을 적재한다. torch 가 필요한 갤러리 전수 추론이라 이 repo(Lambda)의 일이다.
그룹·이름은 다음 칸인 [`categorize/`](../categorize/README.md), 미리보기·DINOv3 는 앞 칸인 [`embedder/`](../embedder/README.md).

```
[GPU 워커]  photo_analysis SKIP LOCKED 32장 집기 → 점수 → commit          ← 운영 기본 (인스턴스가 있을 때)
[Lambda ]  wes ──EVENT {galleryId, photoIds}──▶ [score]                   ← 폴백 (워커가 없거나 못 따라갈 때)
```

잡 상태·categorize 호출·재시도는 **wes 가 소유한다**(#98, wes V16). score 는 어느 경로로 불려도 점수만 쓴다.

## 무엇을 계산하나 (`pipeline.py`)

```
사진마다 (embedder 가 만든 미리보기 JPEG 입력)
  LaionRunner.embed         → CLIP 768d (저장: photo_analysis.clip_embedding — categorize 가 재계산 없이 읽는다)
     ├─ score_from_embedding → aesthetic_score (LAION MLP)
     ├─ SubjectsTagger.tag   → subjects (bride|groom|couple|group|unknown, margin < 0.01 → unknown)
     └─ ParentTagger.tag     → sub_scores.clip_parent (부모 고정 목록 argmax — categorize naming 의 검증용)
  ArniqaRunner.score        → technical_score (spaq 회귀기)
  classical.measure         → sharpness · highlight_clip · shadow_clip · mean_luma
→ store.write_scores  (subjects · sub_scores · clip_embedding · model_version 만)
```

CLIP 벡터 하나를 뽑아 미학·피사체·부모 라벨 세 가지에 쓴다(텍스트 프롬프트는 추가 비용 0). 백분위·연사·그룹은
categorize 의 컬럼이라 UPSERT 의 SET 절에 없다 — 이 경계가 곧 두 Lambda 의 경계다.

## 실행 모양 — embedder 와 같다

| | |
|---|---|
| 진입점 | `handler.py`(Lambda EVENT `{"galleryId", "photoIds"}` — 다른 페이로드는 에러) / `__main__.py`(CLI, 갤러리 전체 가능) → `job.run()` |
| 단위 · 재개 | 사진. 같은 `MODEL_VERSION` 이고 CLIP 이 저장된 사진은 건너뛴다. `write_batch`(32)장마다 commit |
| 속도 손잡이 | 한 장은 한 번만 디코드해 세 러너에 넘긴다. CLIP 은 `CLIP_BATCH`(8)장씩 한 forward, ARNIQA 입력 긴 변은 `ARNIQA_LONG_EDGE`(1024 — 1600 대비 연산 1/2.4, 순위 상관 0.93) (#51) |
| 데드라인 | 15분 앞에서 배치 경계에서 멈추고(`STOP_MARGIN_SECONDS`) commit 한다. 남은 사진은 wes 스윕이 다시 보낸다 |
| 실패 | 사진 단위 결정적 실패는 `photo_analysis.error` — 미리보기 없음(S3 404) `PREVIEW_MISSING`, 점수 실패 `SCORE_FAILED`(#85). wes 가 기대 장수에서 뺀다 |
| 잡·체인·샤딩 | **없다**(#98). wes 가 `ai_analysis_jobs` 를 소유하고 categorize 를 직접 부른다. 갤러리 advisory lock·자기 재호출·조정자도 함께 사라졌다 |
| 접속 | `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE`, `S3_BUCKET`(미리보기), Lambda 는 `/tmp` 만 쓴다(`SCORE_WORK`) |

## 구조

```
score/
├── score/
│   ├── handler.py      Lambda: {galleryId, photoIds} 하나. 데드라인 앞 배치 경계 정지
│   ├── __main__.py     CLI: --gallery-id N [--job-id J] [--force] | --local "갤러리" | --list | worker
│   ├── job.py          갤러리 잡: lock → 잡 RUNNING → EMBEDDED 사진 + 미리보기 다운로드 → pipeline → result 기록
│   ├── pipeline.py     SCORE 본체 (위 그림). Scorer(러너 1회 로드, #75) · 디코드 1회 · CLIP/ARNIQA 배치 · 배치 쓰기 · 데드라인 정지
│   ├── images.py       이미지 로드 (torch 없음) — load_image · fit_long_edge · as_image
│   ├── subjects.py     CLIP zero-shot — SubjectsTagger · ParentTagger
│   ├── classical.py    Laplacian 선명도 · 노출 클립
│   ├── runners/        ArniqaRunner(torch.hub, SHA 고정, score_batch) · LaionRunner(open_clip + MLP) — torch 는 여기만
│   ├── gpu_worker.py   v2 GPU 워커 루프(#75): photo_analysis SKIP LOCKED 32장 집기 → 점수 → commit, 유휴면 자기 정지
│   ├── device.py       cuda → mps → cpu 선택 · cuda fp16 autocast (#68)
│   ├── sagemaker.py    SageMaker training 진입점 — GPU 벤치마크 전용, GPU 사용률 표본
│   ├── config.py       Settings · Knobs · MODEL_VERSION · PARENTS · PARENT_PROMPTS
│   ├── store.py        LocalStore(out/v3/) · DbStore — write_scores 하나
│   ├── gallery.py      PhotoRef — 로컬 폴더 / DB(EMBEDDED + preview_key)
│   ├── storage.py      S3 미리보기 다운로드    ├── db.py  접속    ├── device.py  cuda|mps|cpu
├── tests/test_score.py   pytest 29 — 재개 · 컬럼 경계 · 배치/데드라인 · CLIP/ARNIQA 배치·실패 격리 · 프리페치 · 집기(SKIP LOCKED·error) · GPU 워커 루프 · photoIds 폴백 · handler 계약 · categorize 와의 상수 일치
├── Dockerfile · deploy.sh   컨테이너 Lambda (가중치 빌드 시 번들) · ECR 푸시 + update-function-code
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
.venv/bin/python -m score worker --gpu --once --no-idle-stop       # GPU 집기 워커 한 배치
```

wes 쪽 스크립트가 이걸 감싼다: `../organic-agent-server/wes/scripts/local-ai.sh <galleryId>`(임베딩 → 점수 → 카테고리), `scripts/gpu/score-worker.sh`(GPU 워커 대역).

- **embedder 가 먼저다.** `gallery.load_db` 는 `preview_key` 가 있는 사진만 고른다(#75 — v2 에서 `status` 는 보지 않는다).
- **`MODEL_VERSION` 이 재개 키다.** 러너·전처리를 바꾸면 올린다 — 전 갤러리가 재점수 대상이 된다. categorize 의
  같은 상수와 값이 같아야 하며 테스트가 고정한다.
- `photo_ratings` · `photo_selection_items` 는 읽지 않는다(정책).

## GPU 워커 — 파이프라인 v2 의 운영 모양 (#75)

Lambda 32 샤드 대신 GPU 인스턴스 한 대(또는 몇 대)가 **사진 단위로 집어서** 점수를 낸다. 갤러리를 배정받지 않는다.

```bash
python -m score worker --gpu [--once] [--no-idle-stop]     # 컨테이너 기본 CMD. 로컬에서는 --no-idle-stop
```

- 집기: `photo_analysis.embedding IS NOT NULL AND clip_embedding IS NULL AND error IS NULL`(+ 미리보기 있음·휴지통 아님) 32장을 `FOR UPDATE OF photo_analysis SKIP LOCKED`
  로 잠근 채 미리보기 다운로드(16 스레드) → CLIP·ARNIQA·classical → `write_scores`(UPSERT + commit = 잠금 해제). 워커가 죽으면 롤백으로 행이 자동 반환된다.
  여러 대가 같은 사진을 집을 수 없고(RDS 에서 확인), 한 갤러리를 나눠 먹어도 된다. status 는 보지 않는다(v2 에서 EMBEDDED 가 사라진다).
- 유휴: 집을 게 없으면 `WORKER_POLL_SECONDS`(3) 대기, 연속 `WORKER_IDLE_STOP_SECONDS`(**30**, 다중 사용자 운영이면 600) 를 넘기면 IMDSv2 로 자기 인스턴스를 `StopInstances`. 켜는 것·폴백은 wes.
- 실패: 배치가 `WORKER_MAX_CONSECUTIVE_FAILURES`(5)회 연속 실패하면 루프를 끝내고 exit 1 — 같은 오류로 헛돌지 않는다(#81).
- 잡 테이블은 건드리지 않는다 — 완료는 wes 가 데이터로 관측.
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
`SCORE_DECODE_WORKERS`(디코드·classical 프리페치 스레드, Lambda 는 0) · `SCORE_DOWNLOAD_WORKERS`(S3 동시 다운로드, 기본 8).
CPU 경로는 이 손잡이가 기본값일 때 이전과 비트 단위로 같다(16장 검증).

## 배포

`score/deploy.sh` — 이미지를 빌드해 ECR(`wes-score`)에 밀고 Lambda(`wes-score`) 코드를 갱신한다. 가중치(CLIP · LAION MLP ·
ARNIQA hub)는 빌드 시 `/opt` 아래에 굽는다 — NAT 없는 서브넷이라 런타임 다운로드가 안 된다. **인프라(함수·ECR·VPC
엔드포인트)는 아직 없다** — `../organic-agent-infra` 후속. 실행 역할에 필요한 것: S3 읽기, `lambda:InvokeFunction`(자기
함수 + categorize), Lambda 인터페이스 VPC 엔드포인트. 메모리는 CLIP-L + ARNIQA 로 6–8GB 권장, 실측 전.

## wes 가 읽는 계약

`photo_analysis.subjects` · `sub_scores{technical_score, aesthetic_score, sharpness, highlight_clip, shadow_clip, subjects_margin,
clip_parent}` · `clip_embedding` · `model_version`. 키 이름을 바꾸면 wes·categorize 와 함께 바꾼다.
설계 근거·역사는 `docs/photoselect/`(review-v3-design.md, pipeline-history.md).
