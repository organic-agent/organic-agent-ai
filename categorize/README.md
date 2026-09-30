# categorize — 갤러리 컨셉 폴더(1층·2층) Lambda (torch 없음)

갤러리 하나를 **한 번에** 읽어 백분위 · 연사를 만들고, 촬영 시각(없으면 이미지)으로 자른 구간을 Bedrock 이 **컨셉(1층)**으로 묶은 뒤
1층 안에서 **세부(2층)** 그룹을 만들어 이름을 붙인다. `photo_analysis`(pct · burst · embed_group) 와 `concept_assignments` 에 적재한다.
입력은 전부 DB 에 저장된 것이다 — [`embedder/`](../embedder/README.md) 의 DINOv3 · 촬영 시각, [`score/`](../score/README.md) 의 원점수 · CLIP ·
`bg_luma`. 그래서 numpy · scipy · Bedrock 만으로 돌고 torch 가 없다(테스트가 고정). 실제 폴더는 wes 가 배정을 읽어 만든다.

```
wes ──EVENT {galleryId, jobId, conceptCount?}──▶ [categorize] ──▶ concept_assignments · photo_analysis 백분위·연사·그룹
                                                                  실패하면 analysis_jobs.error 한 컬럼만 (#95)
```

스튜디오는 컨셉 하나를 다 찍고 쉬었다가 다음 컨셉으로 넘어간다 — 그 쉬는 시간이 1층의 경계다. 1층 이름은 고정 목록이 아니라
VLM 이 짓는 자유 이름이고 사용자가 고친다. 근거: `docs/experiments/concept-segmentation-2026-09-30.md`
(갤러리 18 · 멘토 라벨 · 스튜디오 컨셉 4개: 옛 고정 목록 1층 ARI 0.32 → 이 파이프라인 0.94).

## 무엇을 계산하나 (`service/pipeline.py`)

```
갤러리 한 번 (E = DINOv3, C = CLIP, 원점수 · bg_luma, taken_at · camera — 전부 DB)
  백분위     technical/aesthetic_score · sharpness → *_pct (NaN → 50)
  연사       E, 카메라 파티션 ∧ taken_at 순 창 ≤ 8 ∧ cos ≥ 0.96 → burst_id · burst_rank(연사 대표 0, rank_reason)
  구간       (service/segment.py) 시각이 90% 이상 있고 서로 다르면 → 시각순, 5분 넘게 쉰 곳에서 자름.
             0.5% 미만 구간(테스트 컷)만 공백이 짧은 이웃에 붙임. 시각이 없으면 → DINOv3 Ward 36묶음. 연사는 경계를 넘지 않음
  1층        (naming.name_concepts) 구간마다 대표 4장 2×2 타일 → Sonnet 한 호출로 "같은 컨셉끼리 묶고 이름"
             시간 모드 + conceptCount → "정확히 K개". 이미지 모드 + conceptCount → "K개라고 기억한다, 애매하면 나눠라"
             이름은 갤러리 안에서 서로 다르게(겹치면 " 2") — wes 가 1층을 이름으로 묶는다
  2층        1층마다 X = normalize([normalize(E) ⊕ normalize(C)]) 평균연결 계층 클러스터, 거리 0.2 (과분할이면 올림)
             embed_group_id 는 갤러리 전체에서 유일, 1층을 넘지 않음
→ store.write_groups  (pct · burst · embed_group · sub_scores 만 — subjects · clip_embedding · pipeline_version 은 score 의 것)
→ naming.name_details  1층마다 커버리지 85% 까지 대표 1~2장 + 1층 이름 → 2층 이름(1층당 한 호출, 동시에)
                       나머지는 같은 1층 안 최근접. 그룹이 하나뿐인 1층은 호출 없이 "전체"
                       needs_review: confidence < 0.8 · 최근접 거리 > 0.25 · 배경 밝기 차 > 80
→ store.write_assignments (concept_assignments, job_id 에 매달림)
```

Bedrock 호출 = 1층 1회(이미지 ≤ 40) + 2층 (그룹이 둘 이상인 1층 수)회(1층당 이미지 ≤ 15). 항상 갤러리 전체를 다시 계산한다.

## 실행 모양

| | |
|---|---|
| 진입점 | `controller/handler.py`(Lambda EVENT `{"galleryId", "jobId"?, "conceptCount"?}`) / `__main__.py`(CLI) → `service/job.run()` |
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
│   │                        pipeline.py(백분위 · 연사 · 구간 → 1층 → 2층 그룹 → write_groups → 2층 이름)
│   │                        segment.py(1층 구간 — 시각 공백, 없으면 DINOv3 Ward)
│   │                        naming.py(① 구간 타일 → 컨셉 묶기·이름 ② 1층별 2층 이름 · 최근접 · 검증 → write_assignments)
│   │                        burst.py(연사 union-find — 카메라 파티션 ∧ 순서 창 ∧ 코사인) · grouping.py(1층별 평균연결 계층 클러스터)
│   ├── domain/              photo.py(PhotoRef) · analysis.py(PhotoAnalysis · ConceptAssignment · GalleryRead · Store 프로토콜)
│   │                        run.py(Grouped · CategorizeResult) — 로직 없음
│   ├── repository/          connection.py(접속) · analysis.py(DbStore — read_gallery 한 쿼리 · write_groups · write_assignments ·
│   │                        preview_paths 배치 SELECT+병렬 다운로드) · local.py(LocalStore, out/v3/) · photos.py(load_db · load_local)
│   │                        jobs.py(fail 하나 — analysis_jobs.error) · storage.py(S3 미리보기, 풀 = 스레드 수)
│   ├── infrastructure/      bedrock.py(LlmClient 프로토콜 · BedrockClient.complete_json — JSON 스키마 강제, 텍스트+이미지 블록 · jpeg_bytes)
│   └── config/              settings.py(Settings · Knobs · LlmKnobs · PIPELINE_VERSION)
├── tests/                   층별 파일(test_service_* · test_repository_* · test_controller_handler · test_boundaries) — pytest 52.
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
.venv/bin/python -m categorize --gallery-id 12 --job-id 34 [--concept-count 10]   # wes 가 부르는 것과 같음 (Bedrock 필수)
.venv/bin/python -m categorize --gallery-id 12 [--llm]                 # 잡 없이 그룹화 확인 (배정은 저장 안 함)
```

- **embedder · score 가 먼저다.** 점수(`pipeline_version == PIPELINE_VERSION`)와 벡터(E·C)가 모두 있는 사진만 대상이고,
  `embedding_model` 이 섞여 있으면 거부한다.
- 로컬 모드의 `out_root` 를 score 와 같은 곳으로 두면 두 CLI 가 파일로 이어진다(`SCORE_OUT` = `CATEGORIZE_OUT`).
- `photo_ratings` · `photo_selection_items` 는 읽지 않는다(정책).

## 배포

`categorize/deploy.sh` — ECR(`wes-categorize`) + Lambda(`wes-categorize`). **인프라는 아직 없다** — `../organic-agent-infra` 후속.
실행 역할에 `bedrock:InvokeModel`(크로스 리전 프로필 포함), 서브넷에 `bedrock-runtime` 인터페이스 VPC 엔드포인트가 있어야
naming 이 닿는다. 메모리 2–3GB 면 7,000장(거리행렬 ~200MB)까지 여유.

## wes 가 읽는 계약 — 바꾸면 wes 와 함께 바꾼다

- `photo_analysis`: `technical_pct` · `aesthetic_pct` · `burst_id` · `burst_rank` · `embed_group_id` ·
  `sub_scores{sharpness_pct, rank_reason}`(score 의 키에 더해서)
- `concept_assignments`: `job_id` · `embed_group_id` · `concept_name` · `detail_name` · `confidence` · `assigned_by` · `needs_review`.
  `concept_name` 은 자유 이름이고 갤러리 안에서 유일하다. `proposed_concept_name` · `clip_concept_name` 은 더 쓰지 않는다(NULL, 컬럼 삭제는 wes)
- 페이로드 `conceptCount`(선택): 사용자가 기억하는 컨셉 수. wes 가 넘긴다
- 용어: 이름의 정본은 용어집(WES-DOCS `docs/glossary.md`)이다. 필드 이름 = DB 컬럼 이름 = wes 필드 이름(wes V23) —
  1층 `concept_name` = wes `ConceptFolder`, 2층 `detail_name` = wes `DetailFolder`, 연사 `burst_id`·`burst_rank`, 임베딩 그룹 `embed_group_id`,
  파이프라인 버전 `pipeline_version`·`PIPELINE_VERSION`. Bedrock 프롬프트의 JSON 키는 `concepts[].segments` · `groups[].detail` 이다.
  로컬 캐시(`out/`)의 옛 키는 읽을 때 옮기고, 없어진 필드는 버린다(`repository/local.py`).
- 손잡이(`config/settings.py`): 연사 0.96, 구간 공백 300초 · 흡수 0.5% · 시각 비율 90% · 이미지 묶음 36, 그룹 거리 0.2, 최근접 τ 0.25,
  커버리지 0.85, review confidence 0.8. 연사·그룹 값은
  CLIP/DINOv2 시절 실측이라 DINOv3 기준 재측정 대상 — 결과의 `similarityProfile` 이 근거.

설계 근거·역사는 `docs/photoselect/`(review-v3-design.md, plan-v3-folder-compare.md, pipeline-history.md).
