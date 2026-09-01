# photoselect — AI 셀렉터 서버

설계: `docs/plan.md`(단일 소스) · 로드맵: `docs/roadmap.md` · 스택: `docs/tech-stack.md`.
이 README는 **코드가 지금 무엇을 할 수 있는가**만 적는다.

## 두 단계

| | 상태 | 무엇 |
|---|---|---|
| **1단계 사진학 추천** | ✅ 동작 | 경량 3종 + VLM 태그 + 클러스터 → 사진학 prior + 장면 커버리지 + MMR → 초안 K장. **실제 데이터셋으로 테스트한다.** |
| **2단계 취향 반영** | ✅ 코드 동작 · 데이터 대기 | 선택·별점·온보딩 쌍 → Bradley-Terry → λ·conf → 같은 초안 코드가 개인화된다. 데이터는 `scripts/review.py` 페이지로 만들거나(팀원 채점) 실제 고객 데이터(DB 연결 후). |

**분기가 없다.** `evidence`가 비어 있으면 1단계, 채워지면 2단계다. 같은 `draft/job.py`다.

## v1 / v2 — 두 파이프라인

`src/photoselect/v1`(VLM·얼굴·Bradley-Terry 원본)과 `src/photoselect/v2`(슬림 — `docs/plan-v2-slim.md`)가
나란히 있다. **두 버전은 클래스를 하나도 공유하지 않는다** — `store` `config`(손잡이) `gallery` `runners/`
`cluster` `llm/client` 까지 각자 사본을 갖는다(중복은 의도: v2 규격을 바꿀 때 v1 이 발목을 잡지 않게).
루트에 남은 것은 버전과 무관한 배관뿐이다: `config.Settings`(환경변수) · `db` · `jobs` · `storage` ·
`worker` · `handler` · `__main__`. 루트 배관은 각 버전의 파사드(`v1/__init__.py`, `v2/__init__.py`:
`settings_from` `store` `gallery` `analyze_module` `draft_module` `bedrock_client`)만 부른다.
선택은 `--pipeline v1|v2` 또는 환경변수 `PHOTOSELECT_PIPELINE`(기본 **v2**); 워커·Lambda 도 같은 설정을 본다.
로컬 출력도 갈린다: v1 `out/<gallery>/`, v2 `out/v2/<gallery>/`. `scripts/review.py --pipeline v1|v2`.

| | v1 | v2 |
|---|---|---|
| 배치 A | ARNIQA · LAION · MediaPipe 얼굴 · VLM(gemma3) 5축+캡션 · 연사 클러스터 | ARNIQA · LAION · 고전 지표(선명도·노출) · 연사 클러스터 · **컨셉 그룹**(임베딩 계층 클러스터) · CLIP zero-shot 피사체(검증 전, 저장만) |
| 배치 B | BT 취향 학습 · λ 곡선 · 장면 쿼터 · MMR · 근거 9종 · 피드백 번역 | 사진학 점수 · **컨셉 쿼터** · MMR · 근거 4종(balance·quality·sibling·concept) · 유형 균형(4단계, `subjects_trusted` 시) |
| model_version | `photoselect-a-0.3` | `photoselect-v2-a-0.1` |
| 실행 환경 | GPU(VLM), 사진당 ~7초 | CPU, 사진당 ~0.8초 |

v2 가 DB 에 쓸 때 VLM 컬럼(scene·framing·lighting·expression)은 `unknown`, `concept_id` 는
`sub_scores.concept_id` 로 들어간다 — wes 마이그레이션(CHECK 완화·컬럼 추가) 전까지의 임시 (`docs/plan-v2-slim.md` §4).

### v3 — 폴더화 (V45 스키마)

v3 는 **폴더화 파이프라인**이다 (wes `docs/plans/ai-folder-structure.md`, V45). v2 는 손대지 않고
그대로 두며(V29 스키마의 이전 기능), v3 도 같은 파사드 규칙의 독립 사본이다. 추천(draft)은 아직 없다 —
폴더별 추천(`docs/plan-v3-folder-compare.md`)을 구현할 때 이 패키지에 온다.

- **FULL** — 사진별 분석: ARNIQA·LAION·고전 지표·CLIP zero-shot 피사체 + 연사 클러스터 +
  **임베딩 그룹**(`embed_group_id`, concat(DINOv3⊕CLIP) 계층 클러스터, 거리 0.2). CLIP 벡터는
  `photo_analysis.clip_embedding` 에 저장한다. 끝에 naming 까지 이어 돈다.
- **naming** — 크기순 상위 K그룹 대표 1장씩만 Bedrock Sonnet 에 (청크 → 통합 1회). 부모는 촬영
  종류(`galleries.shoot_type`)별 닫힌 목록(스키마 enum 강제), 컨셉은 열린 이름. K 밖 소그룹은
  concat 최근접 상속(τ 초과면 '기타'+needs_review), CLIP zero-shot 은 부모 **검증 전용**.
  산출물은 `ai_concept_assignments`(잡에 매달림) — 폴더 구체화는 wes `POST /folder-groups/ai`.
- model_version `photoselect-v3-a-0.1`, 로컬 출력 `out/v3/<gallery>/`.

```bash
python -m photoselect --pipeline v3 worker --llm                          # 웹 버튼(FULL·NAMING 잡) 처리
python -m photoselect --pipeline v3 analyze --db --gallery 12 --llm       # FULL 수동 실행 (naming까지)
python -m photoselect --pipeline v3 naming  --db --gallery 12 --job-id J  # naming만 다시
```

v3 잡(FULL·NAMING)은 naming 산출물까지가 계약이라 워커를 `--llm` 없이 띄우면 잡을 시작하지 않고
바로 실패시킨다. v3 아닌 워커가 NAMING 잡을 집으면 명확한 메시지로 FAILED 처리한다.

## AI 추천 기능 지도 — 무엇이 기능이고 무엇이 아닌가

```
photoselect/
├── src/photoselect/   ← 서비스에서 실제로 도는 코드 전부 (Lambda·EC2·wes local-ai.sh 가 부르는 것)
├── scripts/           개발 도구 — 검수 페이지·스파이크·회귀기 비교·실행 venv
├── tests/             pytest
├── docs/              설계 문서
├── out/  weights/     로컬 산출물·가중치 캐시 (gitignore)
└── pyproject.toml  requirements.txt  README.md
```

**`src/photoselect/` 밖에 있는 것은 서비스에 배포되지 않는다.** AI 추천 기능은 아래 ①~④가 전부다.

### ① 진입점 — 어디서 부르나

| 파일 | 역할 |
|---|---|
| `__main__.py` | 로컬 CLI. `analyze` / `draft` / `worker` / `reset` 서브커맨드 (검수 페이지는 `scripts/review.py`). wes `scripts/local-ai.sh`가 이걸 부른다 |
| `handler.py` | Lambda 진입점. **B(draft)만** 받는다 — `{"jobId": M}` 또는 `{"selectionId": N, "mode": "draft"|"refine"}` |
| `worker.py` | **잡 폴링 워커.** wes가 버튼으로 만든 PENDING 잡을 `claim_next`로 집어 A·B를 돌린다. 로컬 딸깍 흐름과 운영 EC2 워커가 같은 코드 |
| `jobs.py` | `ai_analysis_jobs`·`ai_selection_jobs` 상태 전이(claim/finish/fail). 진입점만 쓰고 파이프라인은 모른다 |

### ② 파이프라인 — 기능 본체 (A → B → C 순서로 흐른다)

```
[A 전수 분석]  갤러리당 1회, GPU EC2/노트북           analyze/
   runners/faces.py   MediaPipe   → 얼굴 수·눈 뜸·미소·bbox
   runners/laion.py   LAION v2    → 미학 점수 (CLIP 임베딩은 점수에만 쓰고 저장 안 함)
   runners/arniqa.py  ARNIQA      → 기술 품질
   vlm.py             Ollama/vLLM → 고정 축 태그 + 캡션 (axes.py 어휘 강제)
   cluster.py         연사 묶기 (DINOv3 코사인 + 파일명 window)
   represent.py       클러스터 대표 선정 (규칙, 모델 없음)
   job.py             위를 순서대로 → store.write_analysis
                                        │
                                        ▼  photo_analysis
[B 초안 한 라운드]  셀렉당 요청마다, Lambda (torch 없음)   draft/
   rerank.explain_selection  사진마다 뽑힌 슬롯(quota·diversity·fill) — 이유 문장 재료
   job._facts / _template    셀 수 있는 사실만 → 주 사유 하나 → 셀렉터 말투 템플릿 (LLM 없이도 납득)
                             취향·유사(되비추기) > 형제("N장 중 이거, 나머지는 눈 감김…") > 희소("전체 800장 중 6장뿐")
                             > 품질 상위 15% > 앨범 자리("준비 장면은 3장이면 충분") > 다양성
                             score_breakdown 에 moment·alternatives(+why_not)·rarity·album_role 을 실어 프론트가
                             "카드 = 순간, 펼치면 형제" 로 그릴 수 있게 한다 (스키마 변경 없음)
   evidence.py        담기·별점·쌍비교·거절 → Bradley-Terry 쌍
   preference.py      BT 학습 + 축별 신뢰도 conf(axis)
   scoring.py         score = (1−λ)·prior + λ·pref
   rerank.py          장면 커버리지 쿼터 + MMR 중복 억제
   job.py             위를 순서대로 → store.write_recommendations
                                        │
                                        ▼  ai_recommendations
[C 텍스트 LLM]  B 안에서 호출, Bedrock, 텍스트만 전송    llm/
   client.py          Bedrock structured-output 래퍼
   reasons.py         초안 확정 시 사진별 이유 문장 일괄 — 사진(본인 + 형제 컷) + 코드북 재료를 Sonnet 에 주고 설득형 문장을 받는다
   feedback.py        자연어 피드백("가족 사진 더") → 축 가중치 delta
```

**1단계(사진학 추천)와 2단계(취향 반영)는 코드 분기가 없다.** evidence가 비면 λ=0으로 1단계, 채워지면
2단계다. 같은 `draft/job.py`가 라운드마다 돈다.

### ③ 계약·설정 — 파이프라인 전체가 공유하는 것

| 파일 | 역할 |
|---|---|
| `axes.py` | **고정 축 어휘 = 이 서버의 계약.** VLM 프롬프트·점수 피처·LLM 피드백·review 페이지가 전부 이걸 본다. `docs/feature-design.md` §고정 축 어휘와 동일해야 한다 |
| `config.py` | `Settings.from_env()` 하나로 환경변수·손잡이(`AnalyzeKnobs`·`ScoreKnobs`·`LlmSettings`) 수렴 |

### ④ 데이터 경계 — 파이프라인이 바깥과 만나는 유일한 통로

| 파일 | 역할 |
|---|---|
| `store.py` | `Store` 인터페이스 + `LocalStore`(out/ JSONL·npy) + `DbStore`(wes Postgres). 쓰기는 `write_analysis`·`write_recommendations` 둘뿐 — CLAUDE.md 접근 규칙을 여기서 강제 |
| `db.py` | Postgres 접속만. SQL 없음 |
| `gallery.py` | "이 갤러리에 어떤 사진이 있나" — 로컬 데이터셋 폴더 / DB `photos` |
| `storage.py` | S3 미리보기 파생본 내려받기. 원본은 열지 않는다 |

### 런타임이 **아닌** 것 — 기능을 만들고 검증하는 도구

| 경로 | 무엇 | 언제 쓰나 |
|---|---|---|
| `scripts/review.py` | 초안 검수 HTML(프론트 대용). 담기·별점·쌍비교 → `evidence.json` | 로컬에서 2단계 데이터를 손으로 만들 때. 서비스에서는 프론트+DB가 이 역할 |
| `scripts/spike/` | 모델 선정 스파이크 하네스 + **실행 venv**(`.venv`) + Ollama 스크립트 | 새 모델 비교할 때. `runners/`는 `analyze/runners/`로 이관됨 — 그쪽이 정본 |
| `scripts/arniqa_regressors.py` | ARNIQA 회귀기 8종 비교 | 회귀기 바꿀 때 한 번 |
| `tests/` | `test_cluster`·`test_draft`·`test_llm`(합성 데이터) · `test_db_store`(DB_HOST 있을 때만) | `pytest` |
| `docs/` | 아래 문서 색인 | 설계 확인 |
| `out/`, `weights/` | 로컬 실행 산출물, 모델 가중치 캐시 (gitignore) | — |
| `pyproject.toml` | src 레이아웃 선언 — `pip install -e photoselect` 로 `python -m photoselect` 가 어디서든 돈다 | 최초 1회 |

### `docs/` 색인 — 읽는 순서

| 문서 | 무엇 | 상태 |
|---|---|---|
| `plan.md` | 4단계 퍼널·점수식·DB 계약 — **단일 소스** | 확정 |
| `feature-design.md` | 사용자 제공 형태·단계별 산출물·고정 축 어휘 | 확정 |
| `tech-stack.md` | 모델·패키지·인프라(A는 EC2, B는 Lambda) | 확정 |
| `roadmap.md` | 단계 ↔ GitHub 이슈 1:1 | 진행 중 |
| `e2e-test-plan.md` | 프론트→wes→AI→DB 배관 확인 계획 | 진행 중 |
| `schema-proposal.md` | AI 테이블 최종 엔티티 제안 (wes Flyway V29로 반영됨) | 참고 |
| `reasons-evolution.md` | 이유 문장이 "캡션"→"사진학 분해"→"셀렉터의 말"로 바뀐 과정과 근거, 현재 재료·기준·테스트 | 확정(2026-08-28) |
| `paper-digest.md` | 채택·비교 논문별 무엇을·어떻게·어떤 테스트·**근거로 분해되는가** — 이유 문장 설계의 전제 | 참고 |
| `research.md` | 모델 후보 논문·벤치마크·라이선스 조사 (digest의 원본) | 참고(2026-08-19) |
| `linear-issues.md` | 이슈 9건 Linear 이관 규격 | 참고 |

## 실행

torch·mediapipe가 있는 venv가 필요하다. 스파이크 venv를 그대로 쓴다. **src 레이아웃이라 최초 1회
editable 설치가 필요하다** (안 하면 `python -m photoselect` 가 모듈을 못 찾는다).

```bash
cd organic-agent-ai
PY=photoselect/scripts/spike/.venv/bin/python
$PY -m pip install -e photoselect --no-deps                             # 최초 1회

$PY -m photoselect analyze --list                                      # 갤러리 목록
$PY -m photoselect analyze --gallery "dataset1/류지혜고객님 (2)"          # A (VLM은 Ollama 떠 있을 때)
$PY -m photoselect draft   --gallery "dataset1/류지혜고객님 (2)"          # B — 1단계 (evidence 없음)
$PY photoselect/scripts/review.py --gallery "dataset1/류지혜고객님 (2)"  # HTML 검수 페이지 (개발 도구)
# 페이지에서 담기/별점/쌍 비교 → '내보내기' → out/<갤러리>/evidence.json 으로 저장
$PY -m photoselect draft   --gallery "dataset1/류지혜고객님 (2)"          # B — 2단계 (evidence 반영, round 2)
(cd photoselect && $PY -m pytest -q)
```

### DB 모드 (`--db`) — wes 공유 Postgres에 직접 읽고 쓴다

`store.DbStore`. 스키마는 wes Flyway V29(`photo_analysis`·`ai_analysis_jobs`·`ai_selection_jobs`·
`ai_recommendations`·`pair_comparison_events`). 갤러리는 `photos.gallery_id` 숫자, 추천은 셀렉
(`photo_selections.id`) 단위다.

```bash
export DB_HOST=localhost DB_PORT=5432 DB_NAME=wes DB_USER=wes DB_PASSWORD=wes DB_SSLMODE=disable  # 로컬 pg
export S3_BUCKET=<미리보기 버킷>      # analyze 가 preview_key 를 내려받는다 (PHOTOSELECT_WORK, 기본 /tmp/photoselect)
$PY -m photoselect analyze --db --gallery 12 [--job-id J] [--no-vlm] [--force]
$PY -m photoselect draft   --db --selection-id 3 [--job-id J] [--llm]
(cd photoselect && DB_HOST=… $PY -m pytest tests/test_db_store.py -q)   # DB_HOST 없으면 skip
```

### 웹 버튼으로 끝까지 — 워커

wes는 "AI 분석"·"AI 추천" 버튼이 눌리면 잡 테이블에 PENDING 행만 넣는다. 이걸 집어가는 게 워커다:

```bash
../organic-agent-server/wes/scripts/local-worker.sh [--no-vlm] [--llm]   # DB·S3 버킷·Ollama·editable 설치까지 알아서
$PY -m photoselect worker [--no-vlm] [--llm] [--poll 2]                  # 환경변수를 직접 줄 때
```

로컬 흐름: `docker compose up postgres` → wes(local) → 웹 → **워커**. 이후 업로드 → 임베딩 실행 → AI 분석 → AI 추천이
전부 웹 버튼이다. 임베딩은 워커가 아니라 wes(local 프로필)가 `scripts/local-ai.sh --only-embed`를 서브프로세스로
띄운다 — 운영의 Lambda EVENT 호출과 같은 자리. 잡마다 새 DB 접속을 열고, 한 잡이 실패해도 FAILED만 적고 워커는 계속 산다.

워커 없이 한 방에 돌리는 옛 경로는 wes `scripts/local-ai.sh <galleryId> --selection-id N`.

- **임베딩은 임베더(DINOv3 ViT-B/16, 768d)를 읽기만 한다.** analyze의 CLIP 벡터는 미학 점수에만 쓰고 저장하지 않는다 —
  같은 768차원이라 DB가 못 막으므로 `DbStore.write_analysis`가 막는다. 클러스터·선호 유사도는 DINOv3 위에서
  돈다(`AnalyzeKnobs.cluster_threshold`·컨셉 임계는 CLIP/DINOv2로 잡은 값이라 DINOv3 재임베딩 후 재측정 대상).
  `DbStore.read_embeddings`는 한 갤러리에 `embedding_model`이 둘 이상 섞이면 실패한다 — 모델 교체 후 전량 재임베딩(force) 전에는 분석을 돌리지 말 것.
- `photo_ratings`는 읽지 않는다(정책). `Evidence.ratings`는 DB 모드에서 항상 비어 있다.
- 잡 상태 전이(`jobs.claim/finish/fail`)는 진입점(CLI·handler)이 한다. `--job-id`가 없으면 잡 없이 돈다.
- RDS는 `DB_SSLMODE=require`(기본)·`db-tunnel.sh` 경유. **미확정 스키마 시험은 로컬 pg에서만.**

VLM은 `photoselect/scripts/spike/setup_ollama.sh`로 띄운다. 없으면 `--no-vlm` — 태그가 기본값으로
채워져 커버리지·이유 문장이 빈약해지지만 나머지는 돈다.

## 출력 (`out/<갤러리>/`)

| 파일 | DB 대응 |
|---|---|
| `analysis.jsonl` | `photo_analysis` 행 |
| `embeddings.npy` + `embedding_ids.json` | `photo_analysis.embedding` (DB는 임베더 DINOv3, 로컬은 CLIP) |
| `evidence.json` | `photo_selection_items` + `pair_comparison_events` + `ai_recommendations.rejected_at` (읽기 전용) |
| `recommendations.jsonl` | `ai_recommendations` (round별 누적) |
| `review-rN.html` | 프론트 대신 |

## 손잡이 (`config.py`)

전부 `study/01-recsys/lab`에서 합성 데이터로 정한 값이고 **실제 갤러리에서 다시 정할 대상**이다.
`analyze`가 찍는 `similarityProfile`로 `cluster_threshold`를, `review` 페이지를 눈으로 보고
`lambda_mmr`·`scene_cap_ratio`를 정한다. `lambda_k`는 쌍 비교 데이터가 생기면 재추정한다.

## 접근 규칙 (CLAUDE.md) — 인터페이스로 강제

`Store`에 `photo_selections.status`를 만지는 메서드가 없고, `photo_selection_items`·
`photo_ratings`는 읽기 메서드만 있다. 쓰기는 `write_analysis`·`write_recommendations` 둘뿐.
이미지는 로컬 모델(경량 3종·Ollama/vLLM)만 본다.
