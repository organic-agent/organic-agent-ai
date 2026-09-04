# embedder · score · categorize 아키텍처 문서

> 갱신: 2026-09-04 (#24 · #25 · #26 · #31 · #35 반영) · 대상 코드: `embedder/embedder/`, `score/score/`, `categorize/categorize/`
> 목적: 배치 모듈 셋의 **현재 구성**을 학습·설명·개선 검토용으로 한 장에 정리한다. 설계 근거는
> wes `docs/plans/ai-folder-structure.md`·`docs/plans/ai-feature-migration-to-wes.md`,
> `docs/photoselect/review-v3-design.md`. 이 문서는 코드가 지금 무엇을 하는지만 적는다.
> (파일 이름의 photoselect 는 역사다 — #35 에서 `photoselect/` 가 `score/`·`categorize/` 로 갈렸다.)

## 0. 한 문단 요약

웨딩 사진 갤러리를 놓고 Lambda 셋이 체인으로 돈다. **디렉토리 하나 = 함수 하나**다(#35). **embedder**는
갤러리당 1회, 원본을 열어 미리보기 JPEG를 S3에 올리고 **그 파일로** DINOv3 벡터를 계산해 EXIF와 함께 wes 공유
Postgres에 적는다. **score**는 미리보기만 읽어 사진별 점수(CLIP·ARNIQA·고전 지표)와 CLIP 벡터·zero-shot 라벨을
적고, 끝나면 **categorize**를 EVENT로 깨운다. categorize는 저장된 벡터와 점수만으로(torch 없이) 연사·임베딩
그룹을 만들고 Bedrock으로 이름을 붙여 `ai_concept_assignments`에 남긴다. 폴더 실체와 폴더별 추천·비교샷은
**wes**가 한다 — 경계는 "갤러리 전수에 torch 모델 추론이 필요한가"이고, wes도 미리보기 몇 장을 읽어 Bedrock에
보낸다. 두 저장소는 HTTP로 데이터를 나르지 않는다. **DB 테이블이 곧 인터페이스**다.

```
              ┌──────────────────── wes (Kotlin/Spring) ─────────────────────────┐
              │ 업로드 → "임베딩 실행" → "AI 분석"(FULL·NAMING) → 폴더 세트 구체화     │
              │        추천(ai_selection_jobs 자기 실행) · 비교샷(동기) · Bedrock      │
              └──┬──────────────┬───────────────────────┬─────────────────┬───────┘
                 │EVENT          │EVENT {galleryId,jobId}  │EVENT (NAMING)     │읽기
                 ▼               ▼  (FULL)                 ▼                   │
  S3 원본 ──▶ [embedder] ─previews/*.jpg─▶ S3 ──▶ [score] ──EVENT──▶ [categorize] ──▶ naming(Bedrock)
                 │                                   │사진마다             │갤러리 한 번        │
                 ▼                                   ▼                   ▼                   ▼
   photos(preview_key, EXIF, EMBEDDED)     photo_analysis(subjects·     photo_analysis(pct·연사·그룹)   concept_folders / detail_folders
   photo_analysis(embedding = DINOv3)        sub_scores·CLIP·model_version)  ai_concept_assignments      photo_category_assignments
                                            ai_analysis_jobs RUNNING       ai_analysis_jobs DONE        ai_recommendations · ai_pair_verdicts
```

## 1. 공통 설계 원칙 (세 모듈이 같은 모양인 이유)

| 원칙 | embedder | score | categorize | 왜 |
|---|---|---|---|---|
| **평탄 패키지, 진입점 둘, 본체 하나** | `handler.py`(Lambda) / `__main__.py`(CLI) → `job.run()` | 동일 | 동일 | 배포 전에 실제 S3·RDS 상대로 같은 코드를 로컬 검증. 실행 환경이 바뀌어도 진입점만 바뀐다 |
| **모듈은 자기 완결** | 자기 `requirements` · `Dockerfile` · `deploy.sh` · `tests/` | 동일 (torch 컨테이너) | 동일 (torch 없는 작은 이미지) | 함수마다 이미지·의존이 다르다. 공용 코드(db·jobs·store)는 각자 필요한 만큼만 갖는다 |
| **설정은 `Settings.from_env()` 하나** | `config.Settings` | + `Knobs` | + `Knobs` + `LlmKnobs` | 코드 어디서도 `os.environ`을 직접 만지지 않는다 |
| **스키마는 wes 소유** | INSERT/UPDATE만 | 동일 | 동일 | 이 repo는 Flyway를 만들지 않는다. 컬럼이 없으면 명확한 에러로 죽는다 |
| **한 장 실패가 잡을 죽이지 않는다** | `failed` / `metadataFailed` | `failed` | — (갤러리 단위) | 재실행 시 남은 것만 이어서 처리한다 |
| **재실행 안전(멱등)** | "벡터 없음"만 대상 | 같은 `MODEL_VERSION` + CLIP 저장된 사진 스킵, 32장마다 commit | 항상 전체 재계산 | 중간에 죽어도 다시 부르면 된다 |
| **데드라인·재호출** | 배치 경계에서 멈춤 → 자기 재호출 | 동일 (같은 잡으로) | 없음 (수 초) | 15분 Lambda 한도 |
| **torch 경계가 디렉토리로 보인다** | `model.py` 함수 안 | `runners/` 뿐 | **torch 미import** (테스트로 고정) | 콜드 스타트·이미지 크기·실행 환경 선택 |
| **접근 규칙을 인터페이스로 강제** | 벡터 두 컬럼만 `ON CONFLICT DO UPDATE` | `write_scores` 하나 | `write_groups` · `write_assignments` | CLAUDE.md 규칙이 리뷰가 아니라 타입으로 지켜진다 |
| **모듈 간 계약은 상수 + 테스트** | `embedding_model` | `MODEL_VERSION` · `PARENTS` | 같은 값 — 서로의 `config.py`를 파싱해 비교하는 테스트 | 두 함수가 따로 배포돼도 어긋나면 테스트가 잡는다 |

## 2. embedder — 갤러리당 1회, 원본을 여는 유일한 곳

### 2.1 파일 지도

```
embedder/embedder/
├── handler.py      Lambda 진입점. {"galleryId"} → job.run / {"jobId"} → admin_job.run. 멈췄으면 자기 재호출
├── __main__.py     CLI: python -m embedder --gallery-id N [--force]
├── config.py       Settings (DB·S3·모델 id·리사이즈·JPEG 품질·STOP_MARGIN_SECONDS). 모델 revision 고정
├── job.py          본체 — 잠금 → 대상 조회 → 8장 배치 루프(데드라인 검사) → 배치 단위 commit
├── db.py           fetch_targets / store_embeddings / try_lock_gallery + admin 잡용 CAS SQL
├── images.py       open_original / prepare(회전·RGB·1024) / to_jpeg / open_preview(올린 JPEG 재디코드) / preview_key_for
├── model.py        DinoEmbedder — CLS 토큰 L2 정규화, 프로세스당 1개(lru_cache)
├── metadata.py     EXIF → PhotoMetadata (회전·축소 전 원본에서 읽는다)
├── storage.py      S3 get/put — 게이트웨이 엔드포인트 경유, NAT 없음
├── quality.py      기술 품질 신호 0..100 (admin QUALITY_ANALYSIS 잡 전용)
├── admin_event.py  관리자 사진 교체 이벤트 계약 (DERIVATIVE | EMBEDDING | QUALITY_ANALYSIS)
└── admin_job.py    한 사진·한 리비전 exact 처리 — 검증 → 계산 → CAS commit
```

### 2.2 갤러리 잡 흐름 (`job.run`) — 순서가 계약이다 (#24)

```
try_lock_gallery   pg_try_advisory_lock(gallery) — 잡혀 있으면 {"skipped": "already running"} (모델 로드 전)
fetch_targets      photos WHERE status<>'PENDING' AND deleted_at IS NULL AND NOT EXISTS(embedding)   ← force 면 조건 제거
   │
   ▼ batch_size(8)씩 — 배치 시작 전: 남은 시간 < (가장 긴 배치 + STOP_MARGIN 60s) 이면 멈추고 stopped=true
   S3 GET 원본 ─▶ open_original ─┬─▶ metadata.extract   (회전·축소 전 원본, best-effort)
                                └─▶ images.prepare ─▶ to_jpeg ─▶ ① S3 PUT previews/{원본키}.jpg
                                                                 ② open_preview(올린 바이트) ─▶ DinoEmbedder.encode
   │                                                             PUT 실패 = 그 사진 failed (encode 안 감)
   ▼ store_embeddings  (한 트랜잭션, 배치 단위 commit)
   INSERT photo_analysis(photo_id, embedding, embedding_model) ON CONFLICT DO UPDATE   ← 벡터 두 컬럼만
   UPDATE photos SET preview_key=?, taken_at=COALESCE(..), status='EMBEDDED'
          WHERE id=? AND storage_key=? AND deleted_at IS NULL                          ← CAS
handler: stopped 이고 processed>0 이면 같은 payload(force=false)로 자기 EVENT 재호출 (best-effort)
```

설명할 때 짚을 포인트:

- **벡터는 S3에 실제로 올라간 미리보기에서 나온다.** "미리보기 없는 벡터"가 구조적으로 생길 수 없고,
  `status='EMBEDDED'`는 곧 "벡터와 미리보기가 둘 다 있다"는 뜻이다. score·categorize가 보는 픽셀과 벡터가
  같은 파일이다. 대가는 1024px JPEG 재디코드 장당 수십 ms.
- **실패는 두 종류뿐이다.** 원본을 못 읽었든 PUT이 실패했든 벡터가 없으면 `failed` — 다음 호출이
  자연히 다시 집는다. EXIF만 실패하면 `metadataFailed` — 사진은 멀쩡히 보인다. `failed`가 대상 수와
  같으면 임베딩이 아니라 IAM(`s3:PutObject`)을 봐야 한다.
- **타임아웃 앞에서 스스로 멈춘다.** 하드 킬은 진행 중인 배치를 롤백시키므로, 배치 경계에서 commit하고
  자기 재호출로 잇는다. 처리 0장이면 재호출하지 않는다(같은 사진이 계속 실패하는 갤러리에서 무한 루프
  방지). 재호출은 Lambda 인터페이스 VPC 엔드포인트와 자기 함수 `lambda:InvokeFunction`이 있어야
  실제로 동작한다 — 없으면 `reinvoked: false`로 끝나고 앱이 다시 부르면 이어진다.
- **벡터는 `photo_analysis`, 정체성은 `photos`.** 벡터는 모델을 바꾸면 다시 적는 파생값이고 EXIF는
  업로드 때 정해지는 값이라 테이블을 나눴다. 같은 행에 score·categorize가 점수·그룹을 채우므로 embedder는
  `embedding`·`embedding_model`만 건드린다.

### 2.3 admin exact-photo 잡 (`admin_job.run`)

관리자 사진 교체 outbox가 `{"jobId", "attemptCount", "jobType", "photoId", "revisionId", …}`를
보내면 **한 사진·한 리비전만** 처리한다. 검증 SELECT는 짧은 트랜잭션으로 끝내고 긴 S3·CPU 구간에는
DB 커넥션을 잡지 않는다. 결과 저장은 새 커넥션에서 사진 UPDATE와 `admin_processing_jobs` SUCCEEDED
전이를 한 트랜잭션에 CAS한다. rowcount가 1이 아니면 `AdminJobClaimLost` → 롤백. 응답은
`SUCCEEDED | FAILED | IGNORED`.

### 2.4 배포·접속에서 알아야 할 것

- Lambda 컨테이너 이미지(zip 250MB 한도를 torch가 넘음). 가중치는 **빌드 시** HF 토큰(빌드 시크릿)으로
  굽고 런타임은 오프라인. `deploy.sh`가 다이제스트로 `update-function-code`까지 하고 `CodeSha256`을
  검증한다. `terraform apply`나 `:latest` 푸시만으로는 배포되지 않는다.
- DB 접속은 원래 RDS IAM 토큰 설계였으나 조직 SCP가 `rds-db:connect`를 막아 임시로 `DB_PASSWORD`.
- 서브넷에 NAT가 없다. S3는 게이트웨이 엔드포인트. HF·Parameter Store·Lambda API는 접근 불가.

### 2.5 산출 계약 (score·categorize·wes가 읽는 것)

| 테이블.컬럼 | 값 | 소비처 |
|---|---|---|
| `photo_analysis.embedding` (vector 768) | DINOv3 ViT-B/16 CLS, L2 정규화 | CATEGORIZE 연사·그룹, wes 추천 MMR |
| `photo_analysis.embedding_model` | 모델 id | 갤러리 안에 둘 이상 섞이면 categorize가 분석을 거부 |
| `photos.preview_key` | `previews/{원본키}.jpg`, 긴 변 1024 | score·categorize·wes가 여는 **유일한** 이미지 |
| `photos.taken_at`, `camera_make/model` | EXIF | 연사 클러스터의 정렬·파티션 키 |
| `photos.status = 'EMBEDDED'` | 벡터 AND 미리보기 | wes "AI 분석" 버튼 활성 조건 · SCORE 대상 선별 |

## 3. score — 미리보기만 읽어 사진마다 점수를 낸다 (torch)

### 3.1 파일 지도

```
score/score/
├── handler.py      Lambda: {galleryId, jobId?, force?} → job.run. 멈췄으면 자기 재호출, 끝났고 잡이면 chain, 체인 실패면 잡 FAILED
├── __main__.py     CLI: --gallery-id N [--job-id J] [--force] | --local "갤러리" | --list | worker(로컬 폴링)
├── job.py          갤러리 잡: advisory lock(0x53434F) → 잡 RUNNING → EMBEDDED 사진 + 미리보기 다운로드 → pipeline → result.score 기록
├── pipeline.py     SCORE 본체 — 사진마다 CLIP·미학·피사체·부모 라벨·ARNIQA·고전 지표. write_batch 마다 commit, 데드라인 정지
├── chain.py        categorize 호출 — CATEGORIZE_FUNCTION_NAME(Lambda EVENT) | CATEGORIZE_COMMAND(서브프로세스)
├── worker.py       로컬 폴링 워커 — wes 에 분석 invoker 가 생기면 삭제
├── subjects.py     CLIP zero-shot — SubjectsTagger(피사체) · ParentTagger(부모 라벨)
├── classical.py    Laplacian 선명도 · 하이라이트/섀도 클립
├── runners/        ArniqaRunner(torch.hub, 커밋 SHA 고정) · LaionRunner(open_clip ViT-L/14 + 미학 MLP) — torch 는 여기만
├── config.py       Settings · Knobs · MODEL_VERSION · PARENTS · PARENT_PROMPTS
├── store.py        LocalStore · DbStore — write_scores (subjects · sub_scores · clip_embedding · model_version)
├── gallery.py · storage.py · db.py · jobs.py(start · record · fail · claim_next)
```

### 3.2 잡 흐름 (`job.run`)

```
① lock  pg_try_advisory_lock('SCO', gallery_id) — 겹치면 skipped="already running"
② 잡    jobs.start: PENDING→RUNNING (재호출이면 이미 RUNNING). chain 미설정이면 여기서 FAILED
③ 대상  photos EMBEDDED + preview_key → S3 미리보기 다운로드(/tmp)
④ 재개  model_version == MODEL_VERSION 이고 clip_embedding 있는 사진은 건너뛴다 (force 면 전부)
⑤ 사진마다
     LaionRunner.embed         → CLIP 768d (저장)
        ├─ score_from_embedding → aesthetic_score
        ├─ SubjectsTagger.tag   → subjects (margin < 0.01 → unknown)
        └─ ParentTagger.tag     → sub_scores.clip_parent
     ArniqaRunner.score        → technical_score
     classical.measure         → sharpness · highlight_clip · shadow_clip · mean_luma
   32장마다 write_scores + commit → 배치 경계에서 남은 시간 < (가장 긴 배치 + 60s) 면 stopped
⑥ 잡    jobs.record: result.score 합치기 (상태는 RUNNING 그대로)
⑦ handler: stopped·processed>0 → 자기 재호출 {galleryId, jobId} / 다 끝났고 잡 → chain.invoke_categorize
```

짚을 포인트: CLIP 벡터 하나를 뽑아 미학·피사체·부모 라벨 세 가지에 쓴다(텍스트 프롬프트는 추가 비용 0). 부모
라벨을 사진 단위로 저장해 두는 덕에 categorize가 CLIP 텍스트 인코더 없이 검증한다. DONE은 여기서 찍지 않는다 —
체인의 끝(categorize)이 찍는다.

### 3.3 배포에서 알아야 할 것

- 컨테이너 Lambda (torch). 가중치 셋(CLIP openai · LAION MLP · ARNIQA hub)을 **빌드 시** `/opt` 아래에 굽는다 — NAT
  없는 서브넷이라 런타임 다운로드가 전부 실패한다. ARNIQA는 `HUB_REPO`의 커밋 SHA로 고정하고 `skip_validation`.
- 실행 역할: S3 읽기, `lambda:InvokeFunction`(자기 함수 + categorize). 인터페이스 VPC 엔드포인트 `lambda` 필요.
- 메모리 6–8GB 권장(CLIP-L 1.7GB + ARNIQA + torch), 장당 ~0.8s CPU → 호출당 ~1,000장. **인프라·실측 아직 없음.**

## 4. categorize — 저장된 것만 읽어 갤러리 한 번에 그룹·이름 (torch 없음)

### 4.1 파일 지도

```
categorize/categorize/
├── handler.py      Lambda: {galleryId, jobId?} → job.run (Bedrock 클라이언트 주입). score 가 체인으로(FULL) / wes 가 직접(NAMING)
├── __main__.py     CLI: --gallery-id N --job-id J | --gallery-id N [--llm] | --local "갤러리" [--llm]
├── job.py          잡: RUNNING(이어받거나 열거나) → EMBEDDED 사진 목록(다운로드 없음) → pipeline → DONE/FAILED
├── pipeline.py     백분위 · 연사 · 임베딩 그룹 → write_groups → naming.run
├── naming.py       Bedrock 이름 · 소그룹 최근접 · 저장된 clip_parent 다수결(majority) → write_assignments
├── cluster.py      연사 union-find (카메라 파티션 ∧ 순서 창 ∧ cos ≥ 0.96)
├── concept.py      임베딩 그룹 — 평균연결 계층 클러스터, 적응 임계
├── llm.py          BedrockClient.complete_json — JSON 스키마 강제, 텍스트+이미지 블록
├── config.py       Settings · Knobs · LlmKnobs · MODEL_VERSION · PARENTS
├── store.py        LocalStore · DbStore — read_analysis/embeddings/clip · write_groups · write_assignments · preview_path(대표 사진 지연 다운로드)
├── gallery.py · storage.py · db.py · jobs.py(start · finish · fail)
```

### 4.2 잡 흐름 (`job.run` → `pipeline.run` → `naming.run`)

```
갤러리 한 번 (입력은 전부 DB: DINOv3 E, CLIP C, 원점수 — model_version == MODEL_VERSION 인 행만)
  백분위          technical/aesthetic_score · sharpness → *_pct (NaN → 50)
  연사            E, 카메라 파티션 ∧ taken_at 순 창 ≤ 8 ∧ cos ≥ 0.96 → cluster_id · cluster_rank(대표 0, rank_reason)
  임베딩 그룹     X = normalize([normalize(E) ⊕ normalize(C)]) → 평균연결 계층 클러스터, 거리 0.2 적응
                    과병합(그룹<4 또는 최대>50%)이면 0.05씩 하강 · 과분할(그룹>n×0.35)이면 0.05씩 상승
→ store.write_groups  (pct · cluster · group · sub_scores 만)
→ naming.run
  ② 크기순 커버리지 85%(상한 120그룹)까지 대표 1~2장(spread > 0.12 면 최원점 추가)을 15장 청크로 Sonnet
     → {parent(enum 강제), concept(열린 한국어), confidence}. 청크 둘 이상이면 통합 텍스트 호출 1회
  ③ VLM 안 간 소그룹 → concat 공간 최근접 이름 상속. 거리 > 0.25 → 기타/기타 + needs_review
  ④ 저장된 clip_parent 그룹 다수결 ≠ VLM 부모, 또는 confidence < 0.8 → needs_review
→ store.write_assignments (ai_concept_assignments, job_id에 매달림)
→ jobs.finish: DONE, result = score 의 것 || {"categorize": …}
```

짚을 포인트: Bedrock 이미지 호출 수 ≤ ⌈대표 수/15⌉ + 1로 비용 상한이 구조로 잡혀 있다. 잡(job_id)은 naming까지가
산출물이라 Bedrock 없이 시작하지 않는다. 로컬 데이터셋 모드는 임베더가 없어 CLIP을 E 자리에 쓴다
(`embeddingsSource: clip`). 연사 0.96·그룹 0.2는 CLIP/DINOv2 시절 실측이라 DINOv3 기준 재측정 대상이다.

### 4.3 배포에서 알아야 할 것

- torch 없음 → 작은 컨테이너(zip도 가능하지만 배포 경로를 셋이 같이 쓴다). 메모리 2–3GB.
- 실행 역할: `bedrock:InvokeModel`(`global.` 크로스 리전 프로필 포함). 인터페이스 VPC 엔드포인트 `bedrock-runtime` 필요 —
  지금 인프라에는 S3 게이트웨이 엔드포인트뿐이다. **인프라 아직 없음.**

## 5. 데이터 계약 — 누가 어떤 컬럼을 쓰나

| 테이블 | 쓰는 쪽 | 컬럼 | 읽는 쪽 |
|---|---|---|---|
| `photos` | embedder | preview_key · EXIF · status | score·categorize(대상·정렬), wes |
| `photo_analysis` | embedder | embedding · embedding_model | categorize, wes 추천 |
| | score | subjects · sub_scores · clip_embedding · model_version | categorize, wes |
| | categorize | technical_pct · aesthetic_pct · cluster_id · cluster_rank · embed_group_id · sub_scores(+sharpness_pct·rank_reason) | naming, wes |
| `ai_concept_assignments` | categorize(naming) | job_id · embed_group_id · parent_name · concept_name · confidence · assigned_by · proposed_parent · clip_parent · needs_review | wes `AiCategoryFolderService` |
| `ai_analysis_jobs` | wes(PENDING) → score(RUNNING, result.score) → categorize(DONE/FAILED, result.categorize) | status · result · error | wes 폴링 |
| `concept_folders` · `detail_folders` · `photo_category_assignments` | wes | — | wes 추천·비교샷 |
| `ai_selection_jobs` · `ai_recommendations` · `ai_pair_verdicts` | wes | — | wes |
| `photo_ratings` | 접근 금지 | | |

**wes가 `sub_scores`에서 읽는 키**(암묵 계약, 이름을 바꾸면 wes와 함께 바꾼다): `sharpness` · `sharpness_pct` ·
`highlight_clip` · `shadow_clip` · `technical_score` · `aesthetic_score`.
**score ↔ categorize 계약**: `MODEL_VERSION`("photoselect-v3-a-0.1")과 `PARENTS`가 두 `config.py`에 같은 값으로 있고,
각 모듈의 테스트가 상대 파일을 파싱해 비교한다. 이벤트는 `{galleryId, jobId}`.
**용어 뒤집힘**: 이 repo의 `parent_name`(큰 분류)이 wes `ConceptFolder`, `concept_name`(컨셉)이 wes `DetailFolder`다.
**`isAnalyzed`**: wes #138이 `model_version` + 백분위까지 보도록 고쳤다 — score→categorize 사이 창을 막는다.

### 5.1 로컬 모드와 DB 모드

| | LocalStore (`out/v3/<gallery>/`, score·categorize 가 같은 규약) | DbStore (wes Postgres) |
|---|---|---|
| 사진 목록 | `../dataset` 폴더, id = 상대 경로 | `photos` EMBEDDED, id = `photos.id`, 미리보기 S3 다운로드(score) / 대표만(categorize) |
| 임베딩 E | **CLIP이 겸임** | DINOv3 |
| 용도 | 데이터셋 회귀 · 프롬프트 실험 | E2E · 운영 |

## 6. 테스트 현황

| 모듈 | 파일 | 건수 | 무엇을 고정하나 |
|---|---|---|---|
| embedder | `tests/` 6파일 | 34 | fetch_targets 조건, PUT→encode 순서, PUT 실패 격리, 데드라인 중단·재호출, advisory lock, admin CAS, quality |
| score | `tests/test_score.py` | 13 | 재개(MODEL_VERSION+CLIP), write_scores 가 categorize 컬럼 보존, 배치 쓰기·데드라인 정지, chain 인자, handler(재호출·체인·체인 실패 시 FAILED), categorize 와 상수 일치 |
| categorize | `tests/test_categorize.py` | 23 | concat 공간 성질, 카메라 파티션, 적응 임계 대칭, naming 4단계, write_groups 가 score 컬럼 보존, CLIP 폴백, **torch 미import**, score 와 상수 일치, handler |

LLM·러너는 전부 Fake다. 실제 Bedrock·DB를 치는 테스트는 없다. Dockerfile 빌드는 아직 돌려 보지 않았다.

## 7. 개선을 검토할 때 볼 자리

1. **인프라.** score·categorize 함수·ECR, 인터페이스 VPC 엔드포인트 `lambda`(embedder 재호출·score 체인)와
   `bedrock-runtime`(categorize naming), IAM 체인 권한. 없으면 로컬 워커(`score worker`)가 그 자리를 대신한다.
2. **wes 분석 invoker.** wes는 아직 PENDING만 만들고 깨우지 않는다. `EmbeddingInvoker`와 같은 `AnalysisInvoker`
   (Lambda EVENT / 로컬 서브프로세스)가 생기면 `score/worker.py`를 지운다.
3. **`MODEL_VERSION` 하나가 SCORE 재개 키.** 러너·임계값을 바꿔도 이 문자열을 안 올리면 기존 갤러리는
   재점수되지 않고, 올리면 전 갤러리가 대상이 된다. 변경 규칙이 문서화돼 있지 않다.
4. **로컬 러너 리사이즈 1600 vs 미리보기 1024.** DB 모드는 축소만 하니 무해하지만 로컬 실험값을 운영
   임계로 옮길 때 점수 스케일이 다르다.
5. **러너를 호출마다 다시 올린다.** `LaionRunner()`(890MB)·`ArniqaRunner()`를 `job.run` 마다 생성한다. embedder의
   `lru_cache` 패턴으로 웜 컨테이너에서 재사용하는 것이 후보다.
6. **연사·그룹 임계 재측정** — `categorize` 결과의 `similarityProfile`·`groupDistance`가 근거다.
7. **Dockerfile 미검증.** score 의 가중치 굽기(HOME·TORCH_HOME 캐시 위치)는 실제 빌드로 확인해야 한다.

## 8. 읽는 순서 추천

1. `embedder/embedder/job.py` → `db.py`의 `store_embeddings` — 배치 패턴의 원형과 #24 순서
2. `score/score/job.py` → `pipeline.py` → `handler.py` — 재개·데드라인·체인이 어떻게 이어지나
3. `categorize/categorize/store.py`의 `Store` 프로토콜 — 데이터 경계 전부
4. `categorize/categorize/pipeline.py` → `naming.py` — 그룹 → 이름
5. wes `docs/category-architecture.md` — 배정이 폴더가 되는 쪽
6. wes `docs/plans/ai-feature-migration-to-wes.md` — 추천·비교샷이 wes에 있는 이유와 경계
