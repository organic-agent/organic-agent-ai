# photoselect — AI 클러스터링 폴더화 배치

`photoselect_v1`은 갤러리 하나를 받아 **사진별 점수와 임베딩 그룹을 만들고, 그룹에 (큰 분류, 컨셉)
이름을 붙여** `photo_analysis`·`ai_concept_assignments`에 적재한다. 실제 폴더(`concept_folders` ·
`detail_folders` · `photo_category_assignments`)는 wes가 그 배정을 읽어 만든다
(`POST /concept-folders/ai`). 설계 정본은 wes `docs/plans/ai-folder-structure.md`, 검토 기록은
`docs/review-v3-design.md`.

폴더별 추천과 비교샷은 **wes로 이관됐다**(#25, wes `docs/plans/ai-feature-migration-to-wes.md`).
경계는 "갤러리 전수에 torch 모델 추론이 필요한가"다 — 임베딩·점수·그룹·이름은 여기, 저장된 숫자를
셈하고 미리보기 몇 장을 Bedrock에 보여 주는 일은 wes.

## 두 잡 — 점수(SCORE)와 카테고리(CATEGORIZE)

| 잡 | 하는 일 | 단위 | torch |
|---|---|---|---|
| **SCORE** (`foldering/score.py`) | CLIP 벡터 → LAION 미학 · zero-shot 피사체 · zero-shot 부모 라벨(`clip_parent`), ARNIQA 기술, 고전 지표 → `photo_analysis` 원점수 + `clip_embedding` | 사진 — 같은 `MODEL_VERSION` + CLIP 저장된 사진은 건너뛴다 | 필요 (~0.8s/장 CPU) |
| **CATEGORIZE** (`foldering/categorize.py`) | 백분위 · 연사 클러스터 · 임베딩 그룹 → `*_pct`·`cluster_*`·`embed_group_id`, 이어서 naming(Bedrock 이름 · 소그룹 최근접 · 저장된 `clip_parent` 다수결 검증) → `ai_concept_assignments` | 갤러리 — 항상 전체 재계산(수 초) | **없음** (numpy · scipy · Bedrock) |

wes `ai_analysis_jobs.mode`는 그대로다. 워커가 **FULL = SCORE → CATEGORIZE**, **NAMING = CATEGORIZE**로
매핑한다(`worker.MODE_STEPS`). 잡(job_id 있음)은 naming 산출물까지가 계약이라 워커를 `--llm` 없이 띄우면
시작하지 않고 실패시킨다. 가른 이유는 #26 — 무게(torch 유무)·단위(사진 vs 갤러리)·재계산 이유(모델 vs
임계값·프롬프트)가 다르다. CATEGORIZE는 torch가 없어 가벼운 Lambda나 wes로 옮길 수 있다.

```
[SCORE — foldering/score.py]                                   사진마다
  preview.jpg ─┬─▶ LaionRunner.embed        → CLIP 768d (저장: photo_analysis.clip_embedding)
               │      ├─▶ score_from_embedding → aesthetic_score
               │      ├─▶ SubjectsTagger.tag   → subjects (bride|groom|couple|group|unknown)
               │      └─▶ ParentTagger.tag     → sub_scores.clip_parent (부모 고정 목록 argmax, 검증용)
               ├─▶ ArniqaRunner.score         → technical_score
               └─▶ classical.measure          → sharpness · highlight_clip · shadow_clip
  → store.write_scores  (subjects · sub_scores · clip_embedding · model_version 만)

[CATEGORIZE — foldering/categorize.py]                         갤러리 한 번
  technical/aesthetic_score → 갤러리 백분위 → technical_pct · aesthetic_pct · sharpness_pct
  E = DINOv3(임베더) ──▶ 카메라 파티션 ∧ 창 ≤ 8 ∧ cos ≥ 0.96 → cluster_id · cluster_rank
  X = concat(E ⊕ CLIP) ──▶ 평균연결 계층 클러스터, 거리 0.2 (적응) → embed_group_id
  → store.write_groups  (pct · cluster · group · sub_scores 만)
  → naming.run:
      그룹 = embed_group_id, 대표 = 중심 최근접 (spread 크면 + 최원점)
      커버리지 85%(상한 120그룹)까지 대표를 15장 청크로 Sonnet → {parent(enum), concept, confidence}
      청크 둘 이상이면 통합 텍스트 호출 1회 · 소그룹은 concat 최근접(거리 > 0.25면 기타 + needs_review)
      저장된 clip_parent 그룹 다수결 ≠ VLM 부모 또는 confidence < 0.8 → needs_review
      → store.write_assignments (ai_concept_assignments, job_id에 매달림)
```

## 구조

```
photoselect/
├── src/photoselect_v1/      서비스에서 도는 코드 전부
│   ├── __main__.py          CLI: score | categorize | analyze(=둘) | naming(=categorize) | worker
│   ├── worker.py            ai_analysis_jobs 폴링 + run_analysis_job — wes mode → 잡 순서 매핑
│   ├── jobs.py              잡 전이 claim / claim_next / finish / fail
│   ├── config.py            Settings(환경) · Knobs(손잡이) · LlmKnobs · PARENTS(부모 고정 목록)
│   ├── store.py             Store 프로토콜 + LocalStore(out/v3/) + DbStore — 쓰기는 write_scores·write_groups·write_assignments
│   ├── gallery.py           PhotoRef 목록 — 로컬 폴더 / DB(EMBEDDED만, 미리보기 다운로드)
│   ├── storage.py           S3 미리보기 다운로드
│   ├── llm.py               BedrockClient.complete_json (JSON 스키마 강제, 이미지 블록)
│   ├── subjects.py          CLIP zero-shot — SubjectsTagger(피사체) · ParentTagger(부모 라벨) · majority
│   └── foldering/
│       ├── score.py         SCORE (torch)
│       ├── categorize.py    CATEGORIZE (torch 없음) — 백분위·연사·그룹 → naming
│       ├── naming.py        Bedrock 이름 · 최근접 · 검증
│       ├── cluster.py       연사 union-find
│       ├── concept.py       임베딩 그룹 (적응 임계)
│       ├── classical.py     Laplacian 선명도 · 노출 클립
│       └── runners/         ArniqaRunner · LaionRunner (torch — score 만 쓴다)
├── scripts/                 개발 도구 (vlm_tag_gallery.py 등). 배포되지 않는다
├── tests/                   pytest 24건 — 합성 데이터, 가짜 LLM·러너. categorize 가 torch 를 import 하지 않음을 고정
├── docs/                    설계 문서
└── out/  weights/           로컬 산출물 · 가중치 캐시 (gitignore)
```

## 실행 모양

wes는 "AI 분석" 버튼이 눌리면 `ai_analysis_jobs`에 PENDING 행만 넣고 깨우지 않는다. 워커가
`claim_next`(SKIP LOCKED)로 집는다. 잡마다 새 DB 접속을 열고, 한 잡이 실패해도 FAILED만 적고
워커는 계속 산다.

```bash
PY=photoselect/scripts/spike/.venv/bin/python          # torch·open_clip이 있는 venv
$PY -m pip install -e photoselect --no-deps              # 최초 1회 (src 레이아웃)

$PY -m photoselect_v1 score --list                       # 로컬 데이터셋 갤러리 목록
$PY -m photoselect_v1 score      --gallery "dataset1/데이터셋1" [--limit 50] [--force]   # 사진별 점수
$PY -m photoselect_v1 categorize --gallery "dataset1/데이터셋1" [--llm]                  # 그룹 (+ naming)
$PY -m photoselect_v1 analyze    --gallery "dataset1/데이터셋1" [--llm]                  # = 둘 다
(cd photoselect && $PY -m pytest -q)

# DB 모드 — wes 공유 Postgres. 갤러리는 photos.gallery_id 숫자
export DB_HOST=localhost DB_PORT=5432 DB_NAME=wes DB_USER=wes DB_PASSWORD=wes DB_SSLMODE=disable
export S3_BUCKET=<미리보기 버킷>
$PY -m photoselect_v1 analyze    --db --gallery 12 --llm [--job-id J]   # wes FULL 과 같은 순서
$PY -m photoselect_v1 categorize --db --gallery 12 --job-id J --llm     # wes NAMING 과 같음
$PY -m photoselect_v1 score      --db --gallery 12 [--force]            # 점수만 (잡 계약 밖, 증분 확인용)
$PY -m photoselect_v1 worker --llm [--poll 2] [--once]
../organic-agent-server/wes/scripts/local-worker.sh --llm   # DB·버킷·editable 설치까지 알아서
```

- **임베더가 먼저다.** `gallery.load_db`는 `status='EMBEDDED'`이고 `preview_key`가 있는 사진만 고른다.
  DINOv3 벡터는 읽기만 하고 `embedding`·`embedding_model`은 건드리지 않는다. 한 갤러리에
  `embedding_model`이 둘 이상 섞이면 분석을 거부한다 — 모델 교체 후 전량 재임베딩(force) 전에는
  돌리지 말 것.
- **재개는 SCORE의 일이다.** 같은 `MODEL_VERSION`이고 CLIP 벡터가 저장된 사진은 건너뛴다(임베더 벡터
  유무는 보지 않는다 — 그건 CATEGORIZE의 입력 조건). `MODEL_VERSION`을 올리면 전 갤러리가 재점수
  대상이 된다. CATEGORIZE는 항상 갤러리 전체를 다시 계산한다.
- **로컬 모드는 임베더가 없어 CATEGORIZE가 CLIP을 DINOv3 자리에 쓴다**(경고 로그, 결과의
  `embeddingsSource: clip`). concat 공간이 CLIP 단독으로 퇴화하므로 데이터셋 실험 한정으로 감수한다.
- `photo_ratings`·`photo_selection_items`는 읽지 않는다(정책).

## 출력 (`out/v3/<갤러리>/`)

| 파일 | DB 대응 |
|---|---|
| `analysis.jsonl` | `photo_analysis` 행 |
| `embeddings.npy` + `embeddings_ids.json` | `photo_analysis.embedding` (로컬은 CLIP) |
| `clip_embeddings.npy` + `clip_embeddings_ids.json` | `photo_analysis.clip_embedding` |
| `assignments.jsonl` | `ai_concept_assignments` |

## wes가 읽는 계약 — 바꾸면 wes와 함께 바꾼다

- `photo_analysis`: `subjects` · `cluster_id` · `cluster_rank` · `embed_group_id` · `technical_pct` ·
  `aesthetic_pct` · `clip_embedding` · `sub_scores{technical_score, aesthetic_score, sharpness,
  sharpness_pct, highlight_clip, shadow_clip, rank_reason, clip_parent}` · `model_version`
  (SCORE가 `subjects`·`sub_scores`·`clip_embedding`·`model_version`, CATEGORIZE가 나머지를 쓴다)
- `ai_concept_assignments`: `job_id` · `embed_group_id` · `parent_name` · `concept_name` · `confidence` ·
  `assigned_by` · `proposed_parent` · `clip_parent` · `needs_review`
- 용어: 이 repo의 `parent_name`(큰 분류)이 wes `ConceptFolder`, `concept_name`(컨셉)이 wes `DetailFolder`다.

## 손잡이 (`config.py`)

연사 임계 0.96, 그룹 거리 0.2, 최근접 τ 0.25, 커버리지 0.85, review confidence 0.8. 연사·그룹 값은
CLIP/DINOv2 시절 실측이라 DINOv3 기준 재측정 대상이다 — `analyze` 결과의 `similarityProfile`이 근거.

## `docs/` 색인

| 문서 | 무엇 |
|---|---|
| `plan-v3-folder-compare.md` | 추천·비교샷 설계 — **구현은 wes** (§2·§3 머리 참조) |
| `review-v3-design.md` | 폴더화 설계 검토 (적응 임계·대표 2장·품질 하한 게이트 근거) |
| `architecture-v2.html` | v2 시절 파이프라인 그림 (역사) |
| `pipeline-history.md` · `reasons-evolution.md` | v1→v2→v3 변천 |
| `paper-digest.md` · `research.md` | 모델 후보 조사 |
| `e2e-test-plan.md` | 프론트→wes→AI→DB 배관 확인 |
