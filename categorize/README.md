# categorize — 갤러리 그룹화·이름 Lambda (torch 없음)

갤러리 하나를 **한 번에** 읽어 백분위 · 연사 클러스터 · 임베딩 그룹을 만들고, 그룹에 (큰 분류, 컨셉) 이름을 붙여
`photo_analysis`(pct · cluster · group) 와 `ai_concept_assignments` 에 적재한다. 입력은 전부 DB 에 저장된 것이다 —
[`embedder/`](../embedder/README.md) 의 DINOv3, [`score/`](../score/README.md) 의 원점수·CLIP·clip_parent. 그래서
numpy · scipy · Bedrock 만으로 돌고 torch 가 없다(테스트가 고정). 실제 폴더(`concept_folders` · `detail_folders` ·
`photo_category_assignments`)는 wes 가 배정을 읽어 만든다(`POST /concept-folders/ai`).

```
wes ──EVENT {galleryId, jobId}──▶ [categorize] ──▶ ai_concept_assignments · photo_analysis 백분위·그룹
                                                   실패하면 ai_analysis_jobs.error 한 컬럼만 (#95)
```

## 무엇을 계산하나 (`service/pipeline.py` → `service/naming.py`)

```
갤러리 한 번 (E = DINOv3, C = CLIP, 원점수 — 전부 DB)
  백분위          technical/aesthetic_score · sharpness → *_pct (NaN → 50)
  연사            E, 카메라 파티션 ∧ taken_at 순 창 ≤ 8 ∧ cos ≥ 0.96 → cluster_id · cluster_rank(대표 0, rank_reason)
  임베딩 그룹     X = normalize([normalize(E) ⊕ normalize(C)]) → 평균연결 계층 클러스터, 거리 0.2 적응
                    과병합(그룹<4 또는 최대>50%)이면 0.05씩 하강 · 과분할(그룹>n×0.35)이면 0.05씩 상승
→ store.write_groups  (pct · cluster · group · sub_scores 만 — subjects · clip_embedding · model_version 은 score 의 것)
→ naming.run
  ② 크기순 커버리지 85%(상한 120그룹)까지 대표 1~2장(spread > 0.12 면 최원점 추가)을 15장 청크로 Sonnet
     → {parent(enum 강제), concept(열린 한국어), confidence}. 청크는 동시에 보내고(`NAMING_PARALLEL` 4, 1 이면 직렬) 둘 이상이면 통합 텍스트 호출 1회
  ③ VLM 안 간 소그룹 → concat 공간 최근접 이름 상속. 거리 > 0.25 → 기타/기타 + needs_review
  ④ 저장된 clip_parent 그룹 다수결 ≠ VLM 부모, 또는 confidence < 0.8 → needs_review
→ store.write_assignments (ai_concept_assignments, job_id 에 매달림)
```

Bedrock 이미지 호출 수 ≤ ⌈대표 수/15⌉ + 1 로 비용 상한이 구조로 잡혀 있다. 항상 갤러리 전체를 다시 계산한다 —
결정적이고 싸다(822장 수 초). 재개는 score 의 일이다.

## 실행 모양

| | |
|---|---|
| 진입점 | `controller/handler.py`(Lambda EVENT `{"galleryId", "jobId"?}`) / `__main__.py`(CLI) → `service/job.run()` |
| 단위 | 갤러리. 데드라인·잠금 없음 (수 초 + Bedrock 몇 번) |
| 잡 | **상태를 쓰지 않는다**(#95, wes V16). wes 가 ANALYZING→CATEGORIZING 으로 옮기며 부르고, 배정 행·백분위를 관측해 DONE 을 찍는다. 여기서 쓰는 것은 실패 시 `error` 하나 — photoselect 역할에도 `UPDATE (error, updated_at)` 만 있다 |
| LLM | 잡(job_id)은 naming 까지가 산출물이라 Bedrock 없이 시작하지 않는다. `BEDROCK_REGION`·`BEDROCK_MODEL_ID`(기본 `global.anthropic.claude-sonnet-4-6`) |
| 접속 | `DB_*`, `S3_BUCKET`(대표 사진 몇 장만 내려받는다), `CATEGORIZE_WORK`(Lambda 는 `/tmp`) |

## 구조

패키지는 층으로 나뉘어 있다(#131, embedder #121 과 같은 모양). 의존은 한 방향이다 — controller → service → repository · infrastructure,
그리고 모두가 domain 을 본다. `python -m categorize`(CLI)와 `controller/handler.py`(Lambda)는 같은 `service/job.run()` 을 부른다.

```
categorize/
├── categorize/
│   ├── __main__.py          CLI: --gallery-id N --job-id J | --gallery-id N [--llm] | --local "갤러리" [--llm] (python -m 규약상 루트)
│   ├── controller/          handler.py — Lambda {galleryId, jobId} → service.job.run (Bedrock 클라이언트 주입)
│   ├── service/             job.py(잡: 대상 조회 → pipeline → 실패면 error. 상태 전이는 wes #95)
│   │                        pipeline.py(백분위 · 연사 · 임베딩 그룹 → write_groups → naming)
│   │                        naming.py(Bedrock 이름 · 소그룹 최근접 · clip_parent 다수결 · 배경 검증 → write_assignments)
│   │                        cluster.py(연사 union-find — 카메라 파티션 ∧ 순서 창 ∧ 코사인) · concept.py(평균연결 계층 클러스터, 적응 임계)
│   ├── domain/              photo.py(PhotoRef) · analysis.py(PhotoAnalysis · ConceptAssignment · GalleryRead · Store 프로토콜)
│   │                        run.py(Grouped · CategorizeResult) — 로직 없음
│   ├── repository/          connection.py(접속) · analysis.py(DbStore — read_gallery 한 쿼리 · write_groups · write_assignments ·
│   │                        preview_paths 배치 SELECT+병렬 다운로드) · local.py(LocalStore, out/v3/) · photos.py(load_db · load_local)
│   │                        jobs.py(fail 하나 — ai_analysis_jobs.error) · storage.py(S3 미리보기, 풀 = 스레드 수)
│   ├── infrastructure/      bedrock.py(LlmClient 프로토콜 · BedrockClient.complete_json — JSON 스키마 강제, 텍스트+이미지 블록 · jpeg_bytes)
│   └── config/              settings.py(Settings · Knobs · LlmKnobs · MODEL_VERSION · PARENTS)
├── tests/                   층별 파일(test_service_* · test_repository_* · test_controller_handler · test_boundaries) — pytest 41.
│                            helpers.py(합성 갤러리 · FakeLlm) · db_fakes.py(커넥션 가짜). 모델 없음 · torch 미import 를 테스트로 고정
├── Dockerfile · deploy.sh   컨테이너 Lambda (torch 없음, 작다) · ECR 푸시 + update-function-code
└── requirements.txt · pyproject.toml
```

## 로컬 실행

```bash
cd categorize && uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt --no-deps -e .
# score/.venv 에 이미 같이 깔려 있으면 그걸 써도 된다 (../score/.venv/bin/python -m categorize)
.venv/bin/python -m pytest -q

.venv/bin/python -m categorize --local "dataset1/데이터셋1" [--llm]     # score --local 뒤에. 임베더가 없어 CLIP 이 E 를 겸한다(embeddingsSource: clip)

export DB_HOST=localhost DB_PORT=5432 DB_NAME=wes DB_USER=wes DB_PASSWORD=wes DB_SSLMODE=disable S3_BUCKET=<버킷>
.venv/bin/python -m categorize --gallery-id 12 --job-id 34             # wes 가 부르는 것과 같음 (Bedrock 필수)
.venv/bin/python -m categorize --gallery-id 12 [--llm]                 # 잡 없이 그룹화 확인 (배정은 저장 안 함)
```

- **embedder · score 가 먼저다.** 점수(`model_version == MODEL_VERSION`)와 벡터(E·C)가 모두 있는 사진만 대상이고,
  `embedding_model` 이 섞여 있으면 거부한다.
- 로컬 모드의 `out_root` 를 score 와 같은 곳으로 두면 두 CLI 가 파일로 이어진다(`SCORE_OUT` = `CATEGORIZE_OUT`).
- `photo_ratings` · `photo_selection_items` 는 읽지 않는다(정책).

## 배포

`categorize/deploy.sh` — ECR(`wes-categorize`) + Lambda(`wes-categorize`). **인프라는 아직 없다** — `../organic-agent-infra` 후속.
실행 역할에 `bedrock:InvokeModel`(크로스 리전 프로필 포함), 서브넷에 `bedrock-runtime` 인터페이스 VPC 엔드포인트가 있어야
naming 이 닿는다. 메모리 2–3GB 면 7,000장(거리행렬 ~200MB)까지 여유.

## wes 가 읽는 계약 — 바꾸면 wes 와 함께 바꾼다

- `photo_analysis`: `technical_pct` · `aesthetic_pct` · `cluster_id` · `cluster_rank` · `embed_group_id` ·
  `sub_scores{sharpness_pct, rank_reason}`(score 의 키에 더해서)
- `ai_concept_assignments`: `job_id` · `embed_group_id` · `parent_name` · `concept_name` · `confidence` · `assigned_by` ·
  `proposed_parent` · `clip_parent` · `needs_review`
- 용어: 이 repo 의 `parent_name`(큰 분류)이 wes `ConceptFolder`, `concept_name`(컨셉)이 wes `DetailFolder` 다.
- 손잡이(`config/settings.py`): 연사 0.96, 그룹 거리 0.2, 최근접 τ 0.25, 커버리지 0.85, review confidence 0.8. 연사·그룹 값은
  CLIP/DINOv2 시절 실측이라 DINOv3 기준 재측정 대상 — 결과의 `similarityProfile` 이 근거.

설계 근거·역사는 `docs/photoselect/`(review-v3-design.md, plan-v3-folder-compare.md, pipeline-history.md).
