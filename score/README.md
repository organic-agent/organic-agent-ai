# score — 사진별 점수 Lambda

갤러리 하나의 **모든 사진**에 CLIP ViT-L/14 · ARNIQA · LAION 미학을 돌려 `photo_analysis`에 원점수 ·
피사체 · `clip_embedding` 을 적재한다. torch 가 필요한 갤러리 전수 추론이라 이 repo(Lambda)의 일이다.
그룹·이름은 다음 칸인 [`categorize/`](../categorize/README.md), 미리보기·DINOv3 는 앞 칸인 [`embedder/`](../embedder/README.md).

```
wes "AI 분석"(FULL) ──▶ ai_analysis_jobs PENDING ──EVENT {galleryId, jobId}──▶ [score] ──EVENT──▶ [categorize] ──▶ DONE
                                                                              사진마다        갤러리 한 번
```

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
| 진입점 | `handler.py`(Lambda EVENT `{"galleryId", "jobId"?, "force"?}`) / `__main__.py`(CLI) → `job.run()` |
| 단위 · 재개 | 사진. 같은 `MODEL_VERSION` 이고 CLIP 이 저장된 사진은 건너뛴다. `write_batch`(32)장마다 commit |
| 속도 손잡이 | 한 장은 한 번만 디코드해 세 러너에 넘긴다. CLIP 은 `CLIP_BATCH`(8)장씩 한 forward, ARNIQA 입력 긴 변은 `ARNIQA_LONG_EDGE`(1024 — 1600 대비 연산 1/2.4, 순위 상관 0.93) (#51) |
| 데드라인 | 15분 앞에서 배치 경계에서 멈추고(`STOP_MARGIN_SECONDS`) 처리분이 있으면 **자기 재호출** |
| 샤딩 (#54) | wes 호출은 **조정자** — 사진 수 / `SHARD_PHOTOS`(150, #58 — 짧은 샤드가 느린 호스트 편차를 줄인다) 만큼(최대 `MAX_SHARDS` 8) 자기 함수를 `shard:{index,total}` 로 동시에 띄우고 끝난다. 샤드는 `index % total` 인 사진만, 잠금은 (갤러리, 샤드). `result.scoreShards.done` 카운터가 total 에 닿은 마지막 샤드가 categorize 를 연다. 로컬은 `--shards N` 순차. **전제: Lambda 예약 동시성 ≥ `MAX_SHARDS`**(인프라 `score_reserved_concurrent_executions`, 2026-09-06 부터 8) — 낮으면 샤드가 스로틀돼 라운드가 늘어난다 |
| force | `runStartedAt`(시작 시각) 을 샤드·재호출에 넘겨 그 이후 점수만 "있음" — 재호출이 force 를 잃어도 옛 점수가 남지 않는다 |
| 잠금 | 갤러리 advisory lock (`pg_try_advisory_lock(0x53434F, gallery_id)`) — 연타·재호출 겹침 방지 |
| 잡 | `ai_analysis_jobs` 를 RUNNING 으로 열고 `result.score` 를 기록. **DONE 은 categorize 가 찍는다** |
| 체인 | 끝나면 categorize 를 깨운다 — `CATEGORIZE_FUNCTION_NAME`(Lambda EVENT) 또는 `CATEGORIZE_COMMAND`(로컬 서브프로세스). 잡인데 둘 다 없으면 시작 전에 FAILED |
| 접속 | `DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE`, `S3_BUCKET`(미리보기), Lambda 는 `/tmp` 만 쓴다(`SCORE_WORK`) |

## 구조

```
score/
├── score/
│   ├── handler.py      Lambda: 데드라인 → 자기 재호출 / 끝나면 chain / 체인 실패면 잡 FAILED
│   ├── __main__.py     CLI: --gallery-id N [--job-id J] [--force] | --local "갤러리" | --list | worker
│   ├── job.py          갤러리 잡: lock → 잡 RUNNING → EMBEDDED 사진 + 미리보기 다운로드 → pipeline → result 기록
│   ├── pipeline.py     SCORE 본체 (위 그림). 디코드 1회 · CLIP 배치 · 배치 쓰기 · 데드라인 정지
│   ├── images.py       이미지 로드 (torch 없음) — load_image · fit_long_edge · as_image
│   ├── chain.py        categorize 호출 — Lambda EVENT | 서브프로세스
│   ├── worker.py       로컬 폴링 워커 (운영 없음 — wes 에 invoker 가 생기면 삭제)
│   ├── subjects.py     CLIP zero-shot — SubjectsTagger · ParentTagger
│   ├── classical.py    Laplacian 선명도 · 노출 클립
│   ├── runners/        ArniqaRunner(torch.hub, SHA 고정) · LaionRunner(open_clip + MLP) — torch 는 여기만
│   ├── config.py       Settings · Knobs · MODEL_VERSION · PARENTS · PARENT_PROMPTS
│   ├── store.py        LocalStore(out/v3/) · DbStore — write_scores 하나
│   ├── gallery.py      PhotoRef — 로컬 폴더 / DB(EMBEDDED + preview_key)
│   ├── storage.py      S3 미리보기 다운로드    ├── db.py  접속    ├── jobs.py  start · record · fail · claim_next
├── tests/test_score.py   pytest 27 — 재개 · 컬럼 경계 · 배치/데드라인 · CLIP 배치/실패 격리 · 샤딩(분배·조정자·마지막 체인·since) · chain · handler · categorize 와의 상수 일치
├── Dockerfile · deploy.sh   컨테이너 Lambda (가중치 빌드 시 번들) · ECR 푸시 + update-function-code
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
.venv/bin/python -m score --gallery-id 12 [--force]                # 점수만 (잡 계약 밖)
CATEGORIZE_COMMAND=".venv/bin/python -m categorize" .venv/bin/python -m score --gallery-id 12 --job-id 34   # 잡 + 체인
.venv/bin/python -m score worker [--once]                          # 웹 "AI 분석" 버튼의 잡을 폴링 (로컬 대용)
```

wes 쪽 스크립트가 이걸 감싼다: `../organic-agent-server/wes/scripts/local-worker.sh`(워커), `local-ai.sh <galleryId>`(임베딩 → 점수 → 카테고리).

- **embedder 가 먼저다.** `gallery.load_db` 는 `status='EMBEDDED'` 이고 `preview_key` 가 있는 사진만 고른다.
- **`MODEL_VERSION` 이 재개 키다.** 러너·전처리를 바꾸면 올린다 — 전 갤러리가 재점수 대상이 된다. categorize 의
  같은 상수와 값이 같아야 하며 테스트가 고정한다.
- `photo_ratings` · `photo_selection_items` 는 읽지 않는다(정책).

## 배포

`score/deploy.sh` — 이미지를 빌드해 ECR(`wes-score`)에 밀고 Lambda(`wes-score`) 코드를 갱신한다. 가중치(CLIP · LAION MLP ·
ARNIQA hub)는 빌드 시 `/opt` 아래에 굽는다 — NAT 없는 서브넷이라 런타임 다운로드가 안 된다. **인프라(함수·ECR·VPC
엔드포인트)는 아직 없다** — `../organic-agent-infra` 후속. 실행 역할에 필요한 것: S3 읽기, `lambda:InvokeFunction`(자기
함수 + categorize), Lambda 인터페이스 VPC 엔드포인트. 메모리는 CLIP-L + ARNIQA 로 6–8GB 권장, 실측 전.

## wes 가 읽는 계약

`photo_analysis.subjects` · `sub_scores{technical_score, aesthetic_score, sharpness, highlight_clip, shadow_clip, subjects_margin,
clip_parent}` · `clip_embedding` · `model_version`. 키 이름을 바꾸면 wes·categorize 와 함께 바꾼다.
설계 근거·역사는 `docs/photoselect/`(review-v3-design.md, pipeline-history.md).
