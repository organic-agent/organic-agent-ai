# AI 셀렉 E2E 테스트 계획 — 실제 사용자 흐름으로 돌려보기

작성 2026-08-27. 목표: **부부가 프론트에서 "AI 추천" 버튼을 누르면 → wes → AI 서버 → DB → 프론트에
추천 사진과 이유가 뜨고, 담기/거절 반응이 다음 라운드에 반영되는 것**을 실제 배포 환경에서 확인한다.
성능(정확도) 검증은 범위 밖 — 배관이 끝까지 이어지는지만 본다.

관련: `STATUS.md`(기능 진행), `schema-proposal.md`(테이블), `tech-stack.md` §4(인프라 목표).

---

## 0. 현재 인프라·서버 실태 (2026-08-27 조사)

| 층 | 있는 것 | 없는 것 |
|---|---|---|
| **인프라** (`../organic-agent-infra`) | 단일 스택, ap-northeast-2. VPC(퍼블릭 + DB 서브넷, NAT 없음, S3 엔드포인트만), RDS pg16 `wes-db`(private, t4g.micro), S3 `wes-photos-*`, EC2 `wes-app`(t4g.micro arm64), ALB `api.easyselect.kr`, **임베더 Lambda `wes-embedder`**(ECR 이미지, 3GB, 15분, DB 서브넷), Loki/Grafana | dev/prod 분리, GPU EC2, vLLM, Bedrock IAM, photoselect용 ECR/Lambda, CloudWatch 알람 |
| **wes** (`../organic-agent-server/wes`) | Flyway **V28**까지. `photo_selections`·`photo_selection_items`·`photo_ratings`·`photos.embedding(768)`. 임베더 트리거 `LambdaEmbeddingInvoker`(EVENT, `{"galleryId","force"}`), `POST /galleries/{id}/embeddings/run`. 프로필 `local`/`prod`만 | `ai_*` 테이블·엔티티·API 전무. 업로드 완료 시 자동 트리거 없음(프론트가 호출) |
| **프론트** (`../organic-agent-test-web`) | Vite+React, 갤러리 상세 탭(photos/clusters/selection/…), 담기·제출·별점 API 연동. `VITE_API_BASE_URL=https://api.easyselect.kr` | AI 추천 UI, 배포 파이프라인(로컬 빌드 → 수동) |
| **AI repo** (여기) | 분석·초안·취향·이유 전부 로컬 동작. `handler.py` 골격. `Store` 인터페이스 | `DbStore`, Dockerfile, 배포 스크립트 |
| **데이터** | RDS 실갤러리 1(743장)·2(57장) — 테스트 계정(user 3, 부부 역할) 접근 가능. `../dataset` 8,000장 | — |

**판단 — 2단 구조.** 클라우드에 두 번째 스택은 만들지 않는다. 대신:

| 층 | 용도 | 왜 |
|---|---|---|
| **로컬** (`docker-compose.local.yml` pg + `bootRun local` + AI CLI `--db` + 프론트 `localhost:8080`) | **스키마·API·UI 시험.** V29를 고치고 DB를 날리고 다시 올리는 곳 | Flyway는 forward-only — 공유 RDS에 한 번 올린 V29는 못 고치고 V30을 쌓아야 한다. 설계 시험은 부술 수 있는 DB에서 |
| **공유 스택** (`api.easyselect.kr`, README상 disposable) | **배포 검증.** Lambda↔RDS 접속·IAM·SG·이미지 크기·콜드스타트 | 로컬로 못 보는 것만. V29는 확정 후 main 머지 → CD로 1회 적용 |

**전제 확인 필요**: `api.easyselect.kr`을 실제 고객이 쓰고 있지 않아야 한다(갤러리 1·2가 테스트 계정
소유인지). 고객이 있다면 Terraform 환경 분리(모듈 변수화 + 상태 키 분리, RDS·ALB·EC2 월 $40~50)가
선행 이슈가 된다.

**제약 (2026-08-27 결정)**: 지금은 infra(Terraform) 수정이 어렵다. 따라서 **Lambda·ECR·IAM을 새로
만들지 않는다.** prod S3 버킷(previews)·RDS는 그대로 쓴다. AI 실행은 §2-0의 워커 방식.

---

## 1. 목표 흐름 (사용자 시나리오)

```
[작가] 갤러리 업로드 → embeddings/run (기존)
                      → wes: ai_analysis_jobs INSERT (PENDING)                 ← 신규
                      → 워커: PENDING claim → 분석 A → photo_analysis 적재 → DONE ← 신규 (§2-0)
[부부] 셀렉 탭에서 "AI 추천 받기"
                      → wes: ai_selection_jobs INSERT (PENDING) — 호출 없음     ← 신규
                      → 워커: claim → DbStore로 photo_analysis·evidence 읽기
                              → ai_recommendations round 1 INSERT (이유 포함) → DONE
[부부] 추천 목록 조회 (폴링) → 담기(기존 API) / 거절                              ← 조회·거절 신규
[부부] "더 추천" → wes: refine 잡 → 워커 round 2 (담긴 것·거절 반영)
```

---

## 2. 결정 사항 — 범위를 줄이는 네 가지

E2E를 **가장 빨리** 잇기 위한 결정. 각각 나중에 되돌릴 수 있다.

### 2-0. AI 실행은 **DB 폴링 워커** — Lambda·Invoker 없음 (infra 동결 대응)
schema-proposal의 "DB가 곧 큐" 설계를 A·B 모두에 적용한다. wes는 `ai_analysis_jobs`·`ai_selection_jobs`에
PENDING을 INSERT만 하고, `python -m photoselect worker`가 두 테이블을 폴링해 잡을 집어(`UPDATE … SET status='RUNNING'
WHERE status='PENDING' … FOR UPDATE SKIP LOCKED`) 실행한다. wes 쪽 `LambdaRecommendationInvoker`·`function-name`
프로퍼티·EC2 IAM 변경이 전부 사라진다.

| 워커 위치 | 언제 | 비고 |
|---|---|---|
| **노트북** (`db-tunnel.sh` + AWS 자격) | 첫 E2E·파일럿 | A(VLM 포함)·B 모두. 사람이 붙어 있을 때만 |
| **EC2 `wes-app` 사이드카** (`docker-compose.prod.yml`, wes repo 소유) | 무인 운영 필요 시 | B만(경량 이미지, arm64, 메모리 한도 384m). t4g.micro 1GiB라 빠듯 — 스왑 의존. A는 계속 노트북/GPU |
| Lambda / GPU EC2 | infra 열리면 | 워커 코드 그대로, 진입점만 바꿈 |

**wes가 잡 생성 후 응답에 넣는 것**: jobId. 프론트는 `GET …/recommendations/jobs/{jobId}` 폴링. 워커가 안 떠 있으면
PENDING에 머문다 → 프론트에 "처리 대기 중" 표시, wes는 20분 넘은 RUNNING을 PENDING으로 되돌린다(schema-proposal §2).

### 2-1. 분석 A는 노트북 워커가 prod 버킷 previews를 읽어 RDS에 쓴다 (GPU EC2 보류)
VLM(gemma3 12B, 6s/장)은 Lambda에서 못 돈다. GPU EC2 + systemd 루프는 Terraform·AMI·비용이
붙는 별개 작업이다. 갤러리 2개(800장)면 노트북 Ollama로 1~2시간이면 끝난다.
→ 로컬 pg(시험) 또는 `db-tunnel.sh`(공유 RDS)에 `python -m photoselect analyze --gallery N --db` 가 `DbStore`로 `photo_analysis`에 쓴다.
`ai_analysis_jobs` 상태도 이 CLI가 갱신한다(EC2 루프가 나중에 할 일과 동일 코드).
**GPU EC2는 E2E가 이어진 뒤 3단계에서.**

### 2-2. 임베딩은 `photo_analysis.embedding`에 **CLIP 그대로** 넣는다 (`photos.embedding` 이전 보류)
schema-proposal §9-1·2(DINOv2 통일, `photos.embedding` DROP)는 embedder·`Photo.kt`·인프라 변수까지
건드린다. E2E 목적엔 불필요. `photo_analysis.embedding vector(768)`을 별도로 두고 CLIP을 넣는다.
통일은 성능 검증 단계에서.

### 2-3. 이유 문장은 **템플릿** 기본, 노트북 워커에서만 `--llm`
노트북 워커는 AWS 자격이 있어 Bedrock 호출이 된다. EC2 사이드카는 인스턴스 역할에 `bedrock:InvokeModel`이
없어(infra 동결) 템플릿만. 워커 옵션으로 두고 기본은 템플릿.

---

## 3. 단계별 작업

각 단계 끝에 **확인 기준**이 있다. 순서는 의존성 순.

### 1단계 — 스키마 + DbStore (wes V29 · AI repo) — 2~3일

**wes** (`../organic-agent-server/wes`, 규칙 `.claude/rules/migration.md`)
- [ ] `V29__ai_selection.sql` — schema-proposal §1~§5 그대로 5테이블. `photo_analysis.embedding vector(768)` 포함
- [ ] `scripts/reset-test-data.sh` TRUNCATE 목록에 5테이블 추가 (FK 자식 우선: `ai_recommendations` → `pair_comparison_events` → `ai_selection_jobs` → `ai_analysis_jobs` → `photo_analysis`)
- [ ] DB 유저 `photoselect` 생성 + GRANT (수동, 임베더 유저와 같은 방식):
  `SELECT` on photos·galleries·photo_selections·photo_selection_items·photo_ratings /
  `SELECT,INSERT,UPDATE` on photo_analysis·ai_analysis_jobs·ai_recommendations·ai_selection_jobs /
  `SELECT` on pair_comparison_events. **`photo_ratings`는 정책상 읽지 않지만 GRANT는 준다**(evidence 코드가 있음, config로 끔)
- [ ] 마이그레이션은 wes CD(main push)로 적용됨 — `flyway_schema_history` 확인

**AI repo**
- [x] `store.DbStore` 구현 — `psycopg` + pgvector, 인터페이스는 `Store` 그대로. 매핑은 schema-proposal §7 표
  - `read_analysis`/`read_embeddings`: `photo_analysis JOIN photos(deleted_at IS NULL)`
  - `read_evidence(gallery, selection_id)`: `photo_selection_items`(담긴 것) + `pair_comparison_events` + `ai_recommendations`(이미 제시·거절)
  - `write_recommendations`: `ai_recommendations` INSERT (round, rank, score, reason, tags)
  - 잡 상태 갱신 헬퍼: `claim_job / finish_job`
- [x] `config.py` DB 모드 — 임베더처럼 **env 직접**(`DB_HOST/PORT/NAME/USER/PASSWORD`, `DB_SSLMODE=verify-full`, `DB_SSLROOTCERT`). Lambda 서브넷에 SSM 경로가 없다
- [x] `__main__.py`에 `--db` 스위치 (기본은 LocalStore)
- [x] `Settings.target_count` → `galleries.max_selectable_photo_count` 읽기

**로컬 세팅** (1·2·4단계 공통 — 여기서 다 돌린다)
- [ ] `docker compose -f docker-compose.local.yml up -d` → wes `bootRun --spring.profiles.active=local` (V29 자동 적용. 고칠 땐 `down -v` 후 재기동)
- [ ] 테스트 갤러리: 로컬 wes에 작가·부부 계정 OAuth 로그인 → `../dataset/dataset1/멘토님께` 200장 업로드 → `embeddings/run`은 로컬에 Lambda가 없으므로 `python -m embedder --gallery-id N`을 로컬 DB·S3로 직접 실행 (previews 생성 필요)
- [ ] AI CLI: `DB_HOST=localhost DB_PORT=5432 DB_NAME=wes DB_USER=wes DB_PASSWORD=wes python -m photoselect analyze --gallery N --db`
- [ ] 스키마 변경은 V29 파일을 **직접 고친다** (공유 RDS에 올리기 전까지는 자유)

**확인**: 로컬 `SELECT count(*) FROM photo_analysis` = 200. `draft --gallery N --db --selection-id <id>` → `ai_recommendations` 30행.
공유 RDS 적용은 **2·4단계까지 로컬에서 끝난 뒤** V29를 main에 머지하고, 터널로 갤러리 2에 같은 CLI를 돌린다.

### 2단계 — wes API 5개 (wes) — 3~4일

패키지 `com.soma.wes.recommendation` (한 단어 규칙). 권한은 `GalleryAccessPolicy` — 트리거·조회·반응은 **부부**(`requireCouple`), 분석 잡 상태는 `requireViewer`.

| 메서드 | 경로 | 하는 일 |
|---|---|---|
| `POST` | `/api/v1/galleries/{g}/photo-selection/recommendations` | body `{mode: "draft"|"refine"}`. `ai_selection_jobs` INSERT(PENDING, 활성 1개 유니크로 중복 방지). **호출은 없음 — 워커가 집어간다.** 202 + jobId |
| `GET` | `…/recommendations` | 최신 라운드(또는 `?round=`) `ai_recommendations` + photo 요약(preview URL) + 잡 상태. 프론트 폴링용 |
| `GET` | `…/recommendations/jobs/{jobId}` | 잡 상태 (PENDING/RUNNING/DONE/FAILED) |
| `POST` | `…/recommendations/{recId}/reject` | `rejected_at` 기록. (담기는 기존 `POST …/photo-selection/photos` — 서비스에서 `accepted_at` 기록 훅 추가) |
| `POST` | `/api/v1/galleries/{g}/ai-analysis` | 작가. `ai_analysis_jobs` PENDING INSERT (워커가 집어감). 임베더 `embeddings/run` 성공 후 프론트가 이어 호출 |

- [ ] 엔티티 3개: `PhotoAnalysis`(읽기 전용), `AiRecommendation`(반응 컬럼만 쓰기), `AiSelectionJob`, `AiAnalysisJob`
- [ ] (Invoker 없음) 잡 INSERT만. 20분 초과 RUNNING → PENDING 되돌리는 스케줄러(선택, 파일럿 땐 수동 UPDATE로 대체 가능)
- [ ] `PhotoSelectionService.selectPhotos` 에서 해당 photo의 미반응 추천 행 `accepted_at = now()`
- [ ] 가드: 셀렉 `SUBMITTED`면 트리거 거부. `photo_analysis` 없는 갤러리면 409 `ANALYSIS_NOT_READY`
- [ ] 통합 테스트 (`@IntegrationTest`, 서비스 직접 호출): 잡 중복 거부·권한·accepted 훅

**확인**: 로컬 wes(`local` 프로필 + 로컬 pg)에서 Swagger로 POST → PENDING 잡 → 로컬 `python -m photoselect worker --once` → GET이 결과를 돌려주는지.

### 3단계 — 워커 (AI repo · wes compose) — 1~2일

**AI repo**
- [ ] `python -m photoselect worker [--once] [--kinds analysis,selection]` — 두 잡 테이블 폴링(5s), `FOR UPDATE SKIP LOCKED`로 claim, 실행, DONE/FAILED + `result`/`error` 기록. 예외는 잡 하나에 격리
- [ ] `handler.py`는 유지하되 워커와 같은 `run_selection_job(job)`을 호출하도록 정리 (infra 열리면 그대로 Lambda)
- [ ] `Dockerfile.worker` — `python:3.12-slim` **arm64**, B 전용 경량 의존(psycopg·numpy·scipy), 이미지 <300MB. `CMD ["python","-m","photoselect","worker","--kinds","selection"]`
- [ ] S3 previews 읽기는 A만 — 노트북 AWS 자격(`aws sso`/프로파일)으로 prod 버킷 `s3:GetObject previews/*`

**wes repo** (사이드카가 필요해질 때 — 첫 E2E는 노트북 워커로 건너뜀)
- [ ] `docker-compose.prod.yml`에 `photoselect-worker` 서비스 — GHCR 이미지, env `DB_*`(CD 배포 스크립트가 Parameter Store에서 읽어 넘기는 방식과 동일), 메모리 한도 384m, `restart: always`
- [ ] AI repo에 GHCR 빌드 워크플로(arm64) — wes `cd.yml`과 같은 OIDC 없이 GHCR 토큰만

**확인**: 노트북에서 `worker --once` → 현 서버 갤러리 2 잡이 DONE, `ai_recommendations` 30행. 사이드카까지 갔다면 노트북 없이 프론트 버튼만으로 DONE.

### 4단계 — 프론트 "AI 추천" UI (test-web) — 2~3일

`gallery-detail-page.tsx` 셀렉 탭(`CLIENT_TABS`) 안에 패널 하나.

- [ ] `api/fetch-recommendations.ts` · `request-recommendations.ts` · `reject-recommendation.ts`
- [ ] `components/selection/ai-recommendation-panel.tsx`
  - 버튼 "AI 추천 받기" → POST → jobId → 3초 폴링 → 그리드
  - 카드: 썸네일 + 이유 문장 + [담기](기존 `select-photos`) [건너뛰기](reject)
  - 상단 진행: "담은 15 / 목표 30" · "더 추천 받기"(refine)
  - `ANALYSIS_NOT_READY`면 "작가가 분석을 아직 안 돌렸어요" 안내
- [ ] 작가 탭: `run-embeddings-button` 옆에 "AI 분석" 버튼(`POST …/ai-analysis`) + 잡 상태 뱃지
- [ ] 빌드 → 기존 방식(수동)으로 `easyselect.vercel.app` 배포

**확인 (= E2E 완료 기준)**: 테스트 계정(user 3)으로 갤러리 2 로그인 → AI 추천 받기 → 30장·이유 표시 → 5장 담기, 3장 건너뛰기 → 더 추천 → 2라운드에 그 8장이 안 나오고 담긴 15장 반영 → 제출은 기존 버튼으로(AI는 `status` 안 건드림) 확인.

### 5단계 — 파일럿 실사용 · 계측 — 1주 병행

- [ ] 갤러리 1(743장) 분석 A 돌리기 (노트북, ~75분)
- [ ] 파일럿 2~3명에게 테스트 계정으로 갤러리 2 셀렉시켜 보기 → `ai_recommendations`의 `accepted_at/rejected_at`로 수락률 산출 (STATUS 2.1 "다수 평가자 검증"의 첫 데이터)
- [ ] 잡 결과는 `ai_selection_jobs.result`(elapsed·k·accepted)에 남는다 — 대시보드는 그 테이블 SQL로 충분. 사이드카면 `./logs`에 JSON 로그를 써 alloy가 집게
- [ ] `pair_comparison_events` 온보딩 UI는 **이 단계에서도 보류** — 기존 별점·담기 신호만으로 취향 학습이 도는지 먼저 본다

### 이후 (E2E 완료 후, 별도 이슈)
- infra 열리면: Lambda B(워커 코드 그대로) · GPU EC2 + systemd 루프 + `Dockerfile.analyze` (STATUS 1.1 "1,000장 ≤10분")
- `--llm` 이유 문장: 노트북 워커는 지금도 가능(AWS 자격). EC2 사이드카는 인스턴스 역할에 `bedrock:InvokeModel` 필요(infra)
- `photos.embedding` → `photo_analysis` 이전, DINOv2 통일
- 온보딩 쌍 비교 UI + `pair_comparison_events` 쓰기 API

---

## 4. 일정·순서 요약

```
주차  1        2        3
wes   [1단계 V29]──[2단계 API ──────]
AI    [1단계 DbStore]──[3단계 워커]
web                        [4단계 UI]──[5단계 파일럿]
```

1·2단계는 wes와 AI repo에서 병렬. infra 작업 없음. 3단계는 API 없이도 잡 행을 손으로 INSERT해 먼저 검증 가능.
총 **약 2~3주**(1인 기준). 1·2·4단계는 로컬에서, 3·5단계만 현 서버.

---

## 5. 리스크·주의

| 리스크 | 대응 |
|---|---|
| 노트북 워커는 사람이 붙어 있을 때만 돈다 — 파일럿 중 잡이 PENDING에 머묾 | 파일럿 시간대에 워커 켜 두기. 무인 필요 시 EC2 사이드카(3단계 wes 항목) |
| EC2 t4g.micro 1GiB에 사이드카 추가 시 메모리 부족 | B는 유휴 시 <100MB, 한도 384m + 스왑. 앱 OOM 나면 사이드카 내리고 노트북으로 복귀 |
| `bootRun --profile local`이 **공유 RDS**에 붙으면 미확정 V29를 실행해 버림 | 로컬 시험 중엔 `SPRING_DATASOURCE_*`를 절대 RDS로 바꾸지 않는다. wes `run-remote` 규칙대로 `flyway_schema_history` 먼저 대조. V29는 반드시 main 머지 → CD로 적용 |
| 분석 A 노트북 실행 중 터널 끊김 | 멱등 재실행이라 재개 가능 (`photo_analysis` 있는 사진은 스킵) |
| 프론트 배포가 수동·git 없음 | 이번엔 그대로 감. 별도 이슈로 Vercel git 연동 |
| 실갤러리 1·2는 실제 부부 데이터 | 테스트 계정이 이미 그 갤러리 부부 멤버. 추가 사용자 초대 없이 계정 공유로 파일럿 |
| `galleries.max_selectable_photo_count` 미설정(0/null) 갤러리 | target 없으면 409, 프론트에서 작가에게 설정 유도 |

---

## 6. 지금 결정이 필요한 것

0. **`api.easyselect.kr`에 실제 고객이 있는가** — 있으면 별도 스택 이슈가 먼저

1. **DB 유저 생성·GRANT 권한** — RDS 마스터로 접속 가능한 사람이 해야 함 (임베더 때 누가 했는지 확인)
2. **API 경로** — `/photo-selection/recommendations` 아래 두는 것(§3 2단계)에 wes 쪽 동의
3. **잡 테이블 1개 vs 2개** (schema-proposal §9-5) — 이 계획은 2개 기준. 합치면 V29만 바뀜
4. **프론트 담당** — test-web이 git repo가 아니라 협업 방식 정해야 함
