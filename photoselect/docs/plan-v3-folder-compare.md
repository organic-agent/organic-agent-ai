# Photoselect v3 계획 — 폴더별 추천 · 상세보기 이유 · 비교샷 AI 판정

작성 2026-08-30 (같은 날 AI 폴더화 계획·7000장 규모 반영). 전제: v2 파이프라인(`docs/plan-v2-slim.md`,
`docs/architecture-v2.html`), wes 현재 코드(폴더 V18, AI 셀렉 V29 작업 트리), 그리고 **AI 폴더화 계획**
(wes `docs/plans/ai-folder-structure.md` — 큰 분류(부모) → 컨셉(자식) 2단 폴더를 naming 잡의 배정으로 서버가 만든다).
제품 원칙(하드 필터 금지, AI는 초안·제안만, 최종 결정은 사람)은 그대로다.

**용어.** 이 문서의 "폴더"는 사진을 실제로 물고 있는 **자식 폴더**(`photo_folders`, = VLM 컨셉)다. 부모(`photo_folder_groups`,
= 큰 분류)는 표시 단위일 뿐 추천 단위가 아니다. `embed_group_id`(임베딩 그룹, 옛 `concept_id`)는 화면에 안 나오는 내부 단위.

## 0. 무엇이 바뀌나 — 한 줄씩

| | v2 (지금) | v3 (요청) |
|---|---|---|
| 추천 단위 | 갤러리 전체에서 k=30장 한 묶음 (임베딩 그룹 쿼터) | **자식 폴더마다** 점수 상위 n장. 폴더 화면에 "AI 추천" 표시 |
| 이유 문장 | 추천 카드에 바로 노출 | **상세보기를 열 때** 노출 |
| 비교샷 | 테이블만 있음 (온보딩 취향 질문용, 미구현) | **두 사진 중 AI가 하나를 고르고 이유를 말한다** — 부부가 고르는 걸 돕는 동기 기능 |

바뀌지 않는 것: 점수식(prior = 0.5·tech_pct + 0.5·aes_pct), 연사 클러스터, 근거
재료(수치 1층 + 사진 2층), Sonnet 4.6 + 사진 전송, 템플릿 폴백.

## 1. wes 쪽 사실 (탐색 2026-08-30)

- 폴더: `photo_folder_groups(id, gallery_id, name, origin, analysis_job_id)` → `photo_folders(id, group_id, gallery_id, name,
  category, needs_review)` → `photo_folder_items(group_id, folder_id, photo_id)`, `UNIQUE(group_id, photo_id)`. `photos.folder_id` 없음.
  `POST /folder-groups/ai` 한 번이 부모 N개(큰 분류)를 만들고, 그 N개는 `analysis_job_id`로 **한 세트**다. 재실행하면 세트가
  하나 더 생긴다(옛 것은 사용자가 지운다). API `/api/v1/galleries/{g}/folder-groups[/{group}/folders/{folder}/photos]`.
- 추천: `ai_recommendations(selection_id, photo_id, round, rank, score_breakdown jsonb, reason TEXT, presented_at,
  accepted_at, rejected_at, unselected_at, accept_mode)`. `reason` TEXT — 길이 제한 없음. 프론트
  `ai-recommendation-panel.tsx`가 그리드 카드에 reason을 바로 그린다. 상세는 공용 `PhotoDetailDialog`.
- 비교샷: `pair_comparison_events(selection_id, photo_a, photo_b, chosen_photo_id, axis, answered_by)` —
  **사람의 답**을 담는 테이블. Kotlin 엔티티·API·프론트 없음. AI 선택·이유 컬럼 없음.
- wes AI 셀렉 코드 전체가 아직 미커밋 작업 트리. 프론트 repo는 커밋 0.

## 2. 기능 1 — 폴더별 점수 기반 추천

### 2.1 동작

```
입력   AI 폴더 세트 S (요청에 analysisJobId, 없으면 갤러리의 최신 AI 세트) · photo_analysis · 담은 사진 · target
       S = photo_folder_groups WHERE analysis_job_id = J 의 모든 자식 폴더 (현재 photo_folder_items 기준 — 사용자가 옮긴 뒤 상태)
폴더마다 f ∈ S:
   후보  = f.photos − 담은 사진 − 거절                         (재노출 금지는 폴더 단위에선 뺀다 — §2.4)
   n_f   = max(1, round(target · |f| / Σ|f'|))                  target 비례, 폴더당 최소 1
           단, n_f ≤ ceil(|f| · 0.5)                            폴더의 절반 넘게 추천하지 않는다
   순위  = score 내림차순, 연사 클러스터당 1장 (클러스터 대표 = 기존 tech ▸ sharp ▸ aes)
   선택  = 상위 n_f  → (선택) MMR 0.7·score − 0.3·maxCos 로 같은 폴더 안 중복 구도 억제
출력   ai_recommendations 행: folder_id · rank(폴더 안 순위) · score_breakdown · reason=NULL
후속   LLM 이유 문장 채우기 (§2.3) → reason UPDATE
```

- **점수는 갤러리 내 백분위 그대로** 쓴다(폴더 안 재정규화 안 함). 폴더가 작으면 백분위가 무의미해지고,
  "전체에서 상위 3%"라는 근거 문장도 갤러리 기준이어야 말이 된다.
- 세트에 안 들어간 사진(폴더 생성 뒤 올라온 사진, wes의 `unfiledCount`)은 "미분류" 가상 폴더로 같은 규칙을 적용한다.
  안 그러면 영영 추천이 안 된다.
- AI 폴더 세트가 없는 갤러리 → **409** (`FOLDERS_NOT_READY`). 폴더화가 정식 단계(업로드 → 임베딩 → 분석 → naming → 폴더 →
  추천)가 됐으므로 컨셉 폴백을 두지 않는다 — 폴백이 있으면 "폴더별 추천"이 두 가지 모양이 된다.
  구현의 뼈대는 `rerank.explain_selection`의 `groups` 인자에 임베딩 그룹 대신 **폴더 id**를 넣는 것이고, 폴더 안 연사 묶기는
  `cluster_id` 그대로다.
- 유형 균형(4단계, `subjects_trusted`)은 유지 — 폴더 안 순위에 `balance_z`가 그대로 가산된다.

### 2.2 목표 장수 n_f 의 해석

`target = galleries.max_selectable_photo_count`. 폴더 비례 배분은 "앨범에 세트마다 자리를 준다"는 뜻이라
v2 컨셉 쿼터(min 1 · 비례 · cap 40%)와 같은 논리다. 부부가 담은 사진이 늘면 `remaining = target − 담은 수`
기준으로 재배분한다(refine). **폴더별 표시가 목적이므로 remaining ≤ 0 이어도 `done` 으로 끝내지 않고**
n_f = 1 로 폴더마다 대표 1장은 남긴다 — 폴더 화면에 AI 마크가 사라지는 것이 더 이상하다.

### 2.3 이유 문장 — 언제 만들고 언제 보여 주나

| 안 | 장점 | 단점 |
|---|---|---|
| A. 배치에서 미리 생성(지금) | 상세보기 즉시 표시. 동기 경로 불필요 | 추천 수가 늘면 비용·시간 (폴더 69개 × 1~3장 ≈ 100장 → Sonnet 호출 10회, ~$1, ~2분) |
| B. 상세보기 열 때 생성 + 캐시 | 본 사진만 과금 | 첫 열람 5~10초 대기(이미지 3장 + Sonnet). 동기 경로 신설 |
| **C. 2단계 배치 (채택)** | 추천 표시는 즉시(reason NULL로 INSERT), 이유는 같은 잡 안에서 이어서 UPDATE. 프론트는 NULL이면 "이유 준비 중" | 열람이 이유 생성보다 빠르면 잠깐 비어 보임 |

C 로 간다. 비교샷(§3)이 어차피 동기 경로를 만드니, 그 뒤에 B 를 "NULL 인 이유만 요청 시 생성"으로 덧붙일 수
있다(옵션). 이유 생성 순서는 폴더 크기 큰 순 → 사용자가 먼저 볼 폴더부터 채워진다.

프롬프트(`v2/reasons.py` `SYSTEM`)는 그대로. 재료에 폴더 사실 한 줄이 추가된다:
`폴더: "{parent.name} › {folder.name}" {|f|}장 중 {rank}위 (추천 {n_f}장)`. 이름은 naming 잡(VLM)이나 사용자가 붙인 것이므로
문장에 써도 된다 — 임베딩 그룹과 달리 이름이 있다. `needs_review` 폴더면 "이 폴더는 분류를 한 번 확인해 주세요"를 재료에 넣지
않는다(추천 이유와 무관한 말이 섞인다) — 배지로만.

**규모(7000장).** 컨셉 20~30개 × n_f → 추천 100~200장 → 이유 호출 10~20회 × ~20s = 3~7분. 배지가 먼저 뜨고 이유가 뒤따르는
2단계가 여기서도 맞다. 이유 생성 순서는 폴더 크기 큰 순.

### 2.4 라운드·재노출

v2의 "한 번 보여준 사진은 다시 안 나온다"는 30장씩 넘겨 보는 UI의 규칙이었다. 폴더 표시는 **상태**(이 폴더의
AI 추천은 이 n장)이므로 라운드마다 바뀌면 안 된다. 따라서:
- draft: 폴더마다 n_f 장 표시.
- refine(담은 뒤 다시 요청): 담은 사진·거절 제외하고 **다시 계산해 덮어쓴다**. 이전 라운드 노출은 제외 조건에서 뺀다.
- `round` 컬럼은 히스토리로만 남기고 조회는 최신 라운드.

### 2.5 데이터 계약 변경 (wes Flyway 요청)

| 대상 | 변경 | 이유 |
|---|---|---|
| `ai_recommendations` | `folder_id BIGINT NULL REFERENCES photo_folders` 추가. `UNIQUE(selection_id, photo_id)` → `UNIQUE(selection_id, round, photo_id)` | 재현용(폴더 화면은 현재 `photo_folder_items` 조인으로 그린다 — §7). 미분류는 NULL. 덮어쓰기 라운드에 같은 사진이 다시 나올 수 있다 |
| `ai_selection_jobs` | `folder_set_job_id BIGINT NULL REFERENCES ai_analysis_jobs` 추가, `result`에 `perFolder` | 어느 AI 세트 기준이었는지 재현 |
| `POST …/recommendations` | body `{analysisJobId?}` | 없으면 최신 AI 세트. 세트 없으면 409 |
| `GET …/recommendations` | `?folderId=` 필터, 응답에 `folderId`·`reasonReady` | 폴더 화면용 |
| 프론트 | 폴더 그리드 카드에 AI 배지(rank). 카드에서 reason 제거, `PhotoDetailDialog`에 "AI 추천 이유" 섹션(NULL → "준비 중", 폴링) | 요청 사항 |

이 repo의 `DbStore.read_folder_set(gallery, analysis_job_id) -> {folder_id: (parent_name, name, [photo_id])}`(읽기 전용 —
`photo_folder_*`는 wes 소유)가 추가된다. 두 마이그레이션(폴더화 V30, 이 문서 V31·V32)은 wes 한 브랜치에서 번호를 맞춘다.

## 3. 기능 2 — 비교샷: 두 장 중 AI 판정 + 이유

### 3.1 동작

```
입력   selection_id · photo_a · photo_b (같은 갤러리, 분석 완료)
DB     두 사진의 photo_analysis · 미리보기 · 같은 연사 클러스터인지 · 폴더 · 담은 사진 여부
사실   초점(sharpness 비), 기술/미학 백분위 차, 노출, 연사 관계, 폴더, 유형(subjects)
LLM    Sonnet 4.6 — 이미지 2장 + 사실 → {chosen: "a"|"b", confidence: "clear"|"slight", reason}
출력   저장 후 응답. 같은 (selection, a, b) 재요청은 캐시 반환
```

- **항상 하나를 고른다**(요청 사항). 대신 `confidence: slight` 로 "거의 같아요, 굳이 고르면"을 표현한다.
- 판정 규칙은 프롬프트에 둔다: 초점·눈·시선·표정처럼 **사진에서 확인되는 차이**를 우선, 백분위 차는 5pt 이상일
  때만 근거로, 흑백/컬러·구도는 취향이므로 "두 분이 정하실 몫"이라고 말하되 선택은 한다.
- 사람의 답(`pair_comparison_events.chosen_photo_id`)과 AI 판정은 **분리 저장**한다. 사람 답 테이블에 AI 컬럼을
  섞으면 "누가 골랐나"가 흐려진다. 사람 답은 나중에 AI 판정과 일치율을 재는 평가 데이터가 된다.

### 3.2 실행 모양 — 동기, 사용자 대기

CLAUDE.md의 두 번째 워크로드("대화형 · 저지연 동기 API")가 처음 생긴다.

| 환경 | 경로 |
|---|---|
| 운영 | wes → Lambda `RequestResponse` `{"mode":"compare","selectionId":N,"photoA":A,"photoB":B}` → `handler.py` → 응답 JSON. 콜드스타트 회피: compare 는 torch 를 import 하지 않는다(DB + PIL + Bedrock만). 같은 이미지에 두 진입점이면 A/B 배치와 분리된 **경량 Lambda** 로 |
| 로컬 | wes local 프로필이 embedder 와 같은 방식으로 `python -m photoselect compare --db --selection-id N --a A --b B` 서브프로세스 실행, stdout JSON |

지연 예산: 미리보기 2장 S3 read(0.3s) + JPEG 축소(0.2s) + Sonnet 2장 입력 ~2k 토큰·출력 ~200 토큰(3~5s) ≈ **5s**.
프론트는 "AI가 보는 중…" 표시. 8s 넘으면 템플릿 판정(사실만으로 — 초점 → 화질 → 미학 순, 없으면 a)으로 응답하고
`confidence: slight`, `source: template`.

### 3.3 데이터 계약 (wes Flyway 요청)

```sql
CREATE TABLE ai_pair_verdicts (
  id BIGSERIAL PRIMARY KEY,
  selection_id BIGINT NOT NULL REFERENCES photo_selections,
  photo_a BIGINT NOT NULL REFERENCES photos,
  photo_b BIGINT NOT NULL REFERENCES photos,
  chosen_photo_id BIGINT NOT NULL CHECK (chosen_photo_id IN (photo_a, photo_b)),
  confidence VARCHAR(10) NOT NULL,          -- clear | slight
  reason TEXT NOT NULL,
  facts JSONB NOT NULL,                     -- 재현용 사실 목록
  model_version VARCHAR(50) NOT NULL,       -- 모델 id + 프롬프트 버전
  source VARCHAR(10) NOT NULL,              -- llm | template
  created_at, updated_at, version,
  UNIQUE (selection_id, LEAST(photo_a, photo_b), GREATEST(photo_a, photo_b))   -- 순서 무관 캐시
);
```

API: `POST /api/v1/galleries/{g}/photo-selection/compare {photoA, photoB}` → 200 `{chosenPhotoId, confidence,
reason, source}`. 사람 답은 기존 `pair_comparison_events` 로 별도 `POST …/compare/answer`(선택).
프론트: 두 사진을 고르는 UI(폴더 안에서 두 장 선택 → "AI에게 물어보기")가 새로 필요하다 — 현재 없음.

## 4. 이 repo 변경 목록

| 파일 | 변경 |
|---|---|
| `v2/store.py` | `read_folder_set(gallery, analysis_job_id)` (DbStore; LocalStore는 naming 잡 산출 `folders.json`). `write_recommendations`에 `folder_id`. `update_reasons(gallery, {photo_id: reason})`. `read_pair_verdict / write_pair_verdict` |
| `v2/draft.py` | 그룹 = 폴더 세트(없으면 409에 해당하는 예외). 폴더당 `n_f` 계산 → `rerank.explain_selection(groups=folder)`. 2단계: INSERT(reason NULL) → `reasons.generate`(큰 폴더부터) → UPDATE. `done` 게이트 완화(§2.2) |
| `v2/rerank.py` | 쿼터 함수에 `min_per_group` 외 `cap_ratio_per_group`(폴더 절반) 추가. 그 외 재사용 |
| `v2/reasons.py` | 재료에 `folder` 사실 추가. 그대로 |
| `v2/compare.py` (신규) | 사실 수집 → 템플릿 판정 → LLM 판정(`COMPARE_SYSTEM`, 스키마 `{chosen, confidence, reason}`) → 저장. torch 미사용 |
| `v2/config.py` | `V2Knobs.folder_cap_ratio=0.5`, `LlmKnobs.compare_max_tokens=1024`, `compare_timeout_s=8` |
| `handler.py` | `mode: "compare"` 분기(RequestResponse 응답 반환). draft/refine 은 기존 |
| `__main__.py` | `compare --db --selection-id --a --b` 서브커맨드. `draft --folder-group N` |
| `worker.py` | 변화 없음(compare 는 워커를 거치지 않는다) |
| `scripts/review.py` | 폴더별 섹션으로 그리기 + 두 장 골라 compare 로컬 실행 버튼(선택) |
| tests | 폴더 쿼터 배분, 미분류 가상 폴더, 세트 없음 → 409, 2단계 reason UPDATE, compare 템플릿 판정·LLM 파싱·순서 무관 캐시 |

## 5. 이슈 분할 (`.claude/rules/git-workflow.md` 형식)

| # | 제목 | 의존 | 산출물 |
|---|---|---|---|
| A | `[photoselect] feat: 폴더별 점수 기반 추천(배치 B v3)` | 폴더화 A1·A2(AI repo) + wes V30·V31 | `draft.py` 폴더 세트 모드, `read_folder_set`, 미분류 가상 폴더, 테스트, `architecture-v2.html` §③ 갱신 |
| B | `[photoselect] feat: 추천 이유 2단계 생성과 상세보기 노출` | A | reason NULL INSERT → UPDATE, `reasonReady`, 폴더 사실 재료. 프론트 변경은 test-web 이슈로 |
| C | `[photoselect] feat: 비교샷 AI 판정 동기 API` | wes `ai_pair_verdicts` + compare 엔드포인트 | `compare.py`, `handler` compare 모드, CLI, 템플릿 폴백, 지연 실측 |
| D | `[photoselect] cicd: compare 경량 Lambda 분리` | C | torch 없는 이미지, infra 변경은 organic-agent-infra |
| wes | 마이그레이션 2건(V30 폴더 컬럼, V31 pair verdicts) + API 3개 + local 프로필 compare 인보커 | — | wes repo 이슈 |
| web | 폴더 카드 AI 배지 · 상세 다이얼로그 이유 섹션 · 두 장 비교 UI | wes | test-web 이슈 |

순서: **폴더화(A1 full 잡 규모 대응 → A2 naming 잡 → wes 폴더 API)** → A → B → (병렬) C → D. A는 폴더 세트가 전제이므로
폴더화가 끝나기 전에는 머지해도 409만 낸다 — C(비교샷)는 폴더와 무관하니 폴더화와 병렬로 먼저 갈 수 있다.

## 6. 열린 질문 (확인 필요)

1. **기준 세트** — 최신 AI 세트(`analysis_job_id` 최대)를 기본으로 하고, 프론트가 `analysisJobId`로 고를 수 있게 했다.
   사용자가 만든 MANUAL 폴더 그룹을 기준으로 추천하고 싶다면? (지금은 AI 세트만. MANUAL은 `analysis_job_id` NULL이라 세트 개념이 없다)
2. ~~폴더 없는 갤러리 폴백~~ → 409로 확정. 프론트는 "먼저 AI 폴더를 만드세요"를 띄운다.
3. **비교샷의 두 장은 누가 고르나** — 부부가 임의 두 장? 연사 형제만? 임의 두 장을 전제했다(형제면 사실이 더 풍부해질 뿐).
4. **사람 답 수집 여부** — `pair_comparison_events` 에 사람 선택을 남기면 AI 판정 정확도를 잴 수 있다. UI 한 줄("두 분은 어느 쪽?")이 필요.
5. 이유 문장이 상세보기에서만 보이면 **길이 400자**는 적당한가, 더 길어도 되는가(지금 상한 600).

## 7. 리스크

- **동기 지연**: Sonnet 4.6 + 이미지 2장 실측 필요. 8s 초과 시 템플릿 판정 폴백이 있으나 UX 는 "AI가 못 골랐어요"가 아니라 "기준으로만 골랐어요"로 보여야 한다.
- **비용·시간(7000장)**: 컨셉이 VLM 이름이라 폴더 수는 20~30개로 묶이지만 폴더 하나가 1000장을 넘는다. `n_f` 합의 상한은
  `max(target, 폴더 수)`. 추천 잡 자체는 MMR이 k·N·d 라 7000장에서도 수초. 이유 생성만 3~7분 — 2단계로 흡수.
- **폴더 편집과 불일치**: 부부/작가가 사진을 폴더 간 옮기면 `ai_recommendations.folder_id` 가 옛 값. 폴더 화면은 `folder_id` 가 아니라 **현재 `photo_folder_items` 조인**으로 그리고, `folder_id` 는 재현용으로만 쓴다. 추천 잡도 세트를 읽을 때 현재 items 기준이다.
- **세트가 바뀌면**(폴더화 재실행) 옛 세트 기준 추천은 새 세트 화면과 안 맞는다 — 프론트는 `ai_selection_jobs.folder_set_job_id`가 현재 보는 세트와 다르면 "추천을 다시 받으세요"를 띄운다.
- **이미지 국외 라우팅**: v2 갭 그대로. 비교샷은 사용자가 명시적으로 두 장을 보내는 행위라 동의 문구를 UI 에 두기 쉽다.
