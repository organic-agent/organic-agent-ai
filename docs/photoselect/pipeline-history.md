# Photoselect 파이프라인 역사 — v1 → v2 → v3 → photoselect_v1 → photoselect

작성 2026-09-02. **옛 버전 코드(`src/photoselect/` — v1·v2·v3와 그 배관, 관련 스파이크 스크립트)를
삭제하면서** 각 버전이 무엇이었고, 무엇을 실측으로 배웠고, 왜 다음 버전으로 넘어갔는지를 남긴다.
삭제된 코드 전체는 git에 있다 — **복구 지점: `bbaf204`** (`[photoselect] feat: v3 비교샷 AI 판정…(#16)`,
삭제 직전 마지막 main 커밋). `git show bbaf204:photoselect/src/photoselect/v1/...` 로 언제든 읽을 수 있다.

살아남은 코드는 `src/photoselect_v1/` 하나였다 — v3의 확정 기능 3개(폴더화·폴더별 추천·비교샷)만
기능별 구조로 재구성한 것. 그 뒤 추천·비교샷은 wes로 이관(#25), 폴더화는 SCORE·CATEGORIZE 두 잡으로
분리(#26), 경로는 `photoselect/photoselect/` 평탄 구조로 정리됐다(#31). 코드 해설은 별도 아티팩트(photoselect_v1 코드 해설) 참조.

## 0. 연대기 한 눈에

| 버전 | 기간 | 한 줄 정의 | model_version | 스키마 세대 |
|---|---|---|---|---|
| **v1** | ~2026-08-28 | VLM 5축 태그 · MediaPipe 얼굴 · Bradley-Terry 취향 학습 · 근거 9종의 "다 넣은" 원본 | `photoselect-a-0.3` | V29 (photo_analysis 구판) |
| **v2** | 08-29 ~ 08-30 | 실데이터 감사로 **덜어낸** 파이프라인 — ARNIQA·LAION·고전 지표·임베딩 클러스터·근거 4종 | `photoselect-v2-a-0.1` | V29 |
| **v3** | 08-30 ~ 09-01 | **폴더화**(경계는 임베딩, 이름은 VLM) + 폴더별 추천 + 이유 2단계 + 비교샷 동기 판정 | `photoselect-v3-a-0.1` | V45~V47 |
| **photoselect_v1** | 09-02~ | v3 확정분만 기능별 패키지로 재구성 (foldering/recommend/compare). model_version·계약은 v3 그대로 | `photoselect-v3-a-0.1` | V45~V48 |

버전 간 import 관계는 없었다(중복은 의도 — 한 버전의 규격 변경이 다른 버전을 못 건드리게).
루트 배관(`worker.py`·`handler.py`·`__main__.py`)이 `PHOTOSELECT_PIPELINE` 환경변수로 버전을 골랐고,
photoselect_v1에서 이 분기 자체를 없앴다(파이프라인이 하나뿐이므로).

---

## 1. v1 — 원본 파이프라인 (VLM · 얼굴 · Bradley-Terry)

### 구성

배치 A(전수 분석) → 배치 B(초안 추천) → 동기 C(LLM 근거·피드백). 사진마다:

- **VLM 태그** — 로컬 Ollama `gemma3:12b`(4bit), structured outputs로 enum 강제.
  고정 축 5종(scene·framing·lighting·expression·subjects) + 한 줄 캡션.
  프로덕션 계획은 vLLM guided decoding(같은 프롬프트·스키마, 호출부만 교체). 이미지는 외부로 안 나감.
- **얼굴 신호** — MediaPipe Face Landmarker(blendshapes): face_count·eyes_open(전원 눈 뜸 최소)·
  smile(평균)·max_face_ratio. 연사 안 상대 비교 전용(절대 판정 아님).
- **점수** — ARNIQA(spaq) 기술 + LAION Aesthetic v2 미학 → 갤러리 내 백분위.
- **연사 클러스터** — 임베딩(로컬 CLIP / DB는 임베더 DINOv2→v3) 코사인 ≥ 0.96 ∧ 순서 창 ≤ 8,
  union-find. 대표 선정(cluster_rank·rank_reason_code). *이 부품은 v3까지 그대로 살아남았다.*
- **취향 학습 (BT)** — 축 어휘(`axes.py`, 33차원 one-hot)가 계약. 세 신호(온보딩 쌍비교 /
  별점 파생 쌍 / 선택 파생 쌍)를 `evidence.to_pairs`가 (chosen, rejected) 쌍으로 통일 →
  `preference.fit_bt`(차이 벡터 L2 로지스틱 MAP) → 축 가중치 w_hat + **leave-one-out
  conf(axis)** → 증거량 λ(n)로 prior와 혼합.
- **재랭킹** — scene 쿼터(coverage_quota: 최소+비례+상한) → 클러스터당 1장 → MMR.
  `explain_selection`이 slot(quota/diversity/fill)을 함께 반환 — 근거 재료.
- **LLM** — 근거 문장 9종 다듬기 + **자연어 피드백 번역**("가족 사진 더" → [{axis, tag, delta}],
  delta ±0.5 클램프, 어휘 밖 버림 — LLM은 사진을 고르지도 빼지도 못한다).

### 실측으로 배운 것 (v1의 유산)

- **VLM 프롬프트는 표본 50장에서 튜닝 불가** — 1차안 재실행은 250칸 전부 동일(결정적)인데,
  축 규칙을 추가한 v2안은 scene 74→66%·framing 94→82%, 캡션 문장만 바꾼 v3안도 scene 74→62%.
  **관련 없는 부분을 건드려도 축 정확도가 흔들린다.** scene 74%는 프롬프트가 아니라 모델
  크기·데이터(본식 갤러리 부재)의 문제로 결론 (`scripts/spike/vlm_compare.py`, 당시 STATUS.md 1.1).
- **ARNIQA 회귀기는 spaq** — hub 기본 kadid10k(합성 왜곡)는 실사 회귀기들과 순위 상관
  0.16~0.34로 "다른 것을 잰다". koniq10k·spaq·clive는 서로 0.88~0.90 일치. spaq이 흐림
  반응/반전 잡음비 23.7배로 최고, LAION 미학과 상관 0.10으로 가장 독립, IQR 0.122
  (`scripts/arniqa_regressors.py`, 류지혜(2) 60장).
- **in-sample 승률은 낙관 편향** — BT conf를 학습 쌍에서 재면 0.86, LOO로 재면 0.55.
  증거 적은 축일수록 심하다 (study/02 step3 [B]). → preference probe 계획의 평가 설계에 계승.
- **점수는 한 장의 성질, 중복·누락은 목록의 성질** (study/01 step4) — 쿼터·MMR·대표 1장의
  3층 구조. 쏠림 실측(snap 68%) 후 쿼터에 상한(cap) 추가.
- **이유 문장의 3단 진화** (`docs/reasons-evolution.md`) — "캡션+숫자"(haiku, 상위 68% 같은
  약점을 그대로 읽어줌) → 원인은 모델이 아니라 **재료** → 사실 분해(slot·연사 비교·백분위
  방향 지침) → "셀렉터의 말"(코드북 프롬프트). 선정 알고리즘은 이 과정에서 한 번도 안 바뀜.

### 왜 v2로 갔나

2026-08-29 갤러리 1(822장) 실데이터 감사(`docs/analysis-audit-gallery1.md`)가 결정타:

- 추천 결과를 실제로 바꾸는 부품은 **prior·임베딩 클러스터·MMR·재노출 금지·목표 장수 게이트
  다섯 개뿐**. 나머지는 비용만 냈다.
- VLM이 런타임의 96%(5,468/5,709초), GPU 필수. 작용하는 축은 scene 하나인데 라벨 오류가
  크고(detail 82장 전부 인물, 스튜디오 사진에 walk 174장), lighting 87% 단일값, 캡션은 근거
  재료로만 쓰임.
- MediaPipe 얼굴은 **81% 미검출**(short-range 검출기 한계), smile·max_face_ratio는 읽는 코드 없음.
- BT는 증거 0으로 **한 번도 돌지 않았고**, 33차원 회귀+LOO+λ 곡선은 결과를 사람이 설명하기 어려움.
  4단계 요구("이 유형이 부족해요")는 집계 규칙으로 그대로 표현됨.

---

## 2. v2 — 슬림 파이프라인 (`docs/plan-v2-slim.md`)

### 뺀 것 / 새로 넣은 것

**뺐다**: VLM 전체, MediaPipe 얼굴(눈감김 회피 상실 — 갤러리 1 실제 눈감김 4장, 필요 시
2단계 검출 18→68% 실측으로 복귀 가능), BT·conf·λ·evidence.to_pairs, 피드백 번역(생산자 없음),
온보딩 쌍비교 소비자, 근거 9종 → 4종, axes.py 33차원.

**새로 넣었다**:

- **고전 기술 지표**(`classical.py`) — Laplacian 분산 선명도, 하이라이트/섀도 클리핑.
  ARNIQA 스칼라가 못 하는 "왜 높은가"의 사진별 분해. 사진당 <50ms.
- **컨셉 그룹**(`concept.py`) — 임베딩 평균연결 계층 클러스터, 코사인 거리 0.2.
  갤러리 1(DINOv2) 실측: 70그룹이 케이크 세트 50·해변 41·소파 39·정원 37·덩굴 아치 36장으로
  실제 촬영 세트와 일치, **0.3부터 다른 세트가 섞인다**. VLM scene 쿼터의 대체물 —
  "이름은 없다, 경계만 있다".
- **subjects zero-shot**(`subjects.py`) — 이미 있는 CLIP 임베딩 + 텍스트 프롬프트 4종
  (신부/신랑/커플/단체), 추가 비용 ~0. 검증: 확신 라벨 36/36 정답, margin<0.01 구간은
  84장 중 9장 오인(커플→단독) → unknown 기권. **v1 VLM subjects의 대체 성공.**
- **4단계 집계 규칙** — BT 대신 유형별 deficit(부족분)·select_lift. 결과가 곧 이유 문장.

배치 A가 CPU ~1초/장이 되어 Lambda 사정권에 들어옴. 추천은 갤러리 전체에서 30장 한 묶음
(컨셉 쿼터 min 1·비례·cap 40%), 라운드마다 재노출 금지.

### v2 후반의 진화 (#5)

- 이유 문장에 **사진을 직접 전달** — 본인 컷 + 연사 형제 컷(왜 밀렸는지 라벨과 함께)을
  이미지로 보내고, "눈에 보이는 것은 사진에서 본 것만 / 숫자는 재료에 있는 것만"의 코드북
  프롬프트. 모델을 Sonnet 4.6으로 전환(`global.` 크로스 리전 — 서울 온디맨드에 Sonnet 없음,
  Mantle 엔드포인트도 ap-northeast-2에 없어 legacy InvokeModel + output_config 사용 확인).

### 왜 v3로 갔나

제품 요구가 바뀌었다: "30장 한 묶음"이 아니라 **폴더 화면**이 추천의 무대가 됐다.
컨셉 그룹은 경계만 있고 이름이 없어 폴더가 될 수 없었고, 사용자에게 보이는 구조(부모/자식
폴더)와 추천 단위를 일치시켜야 했다.

---

## 3. v3 — 폴더화 · 폴더별 추천 · 비교샷 (`docs/plan-v3-folder-compare.md`, wes `docs/plans/ai-folder-structure.md`)

원칙 한 줄: **경계는 임베딩, 이름은 VLM, 부모 목록은 고정, CLIP은 검증 전용. 추천은 자식 폴더
단위, LLM은 고르지 않고 설명만.** 이 버전이 그대로 photoselect_v1이 됐으므로 여기 수치가
현행 계약의 근거다.

### 기능 1 — AI 폴더화 (FULL·NAMING 잡, V45)

- **concat 공간** — 임베딩 그룹을 DINOv3 단독이 아니라 concat(DINOv3⊕CLIP)에서 자른다.
  실측: concat 0.2에서 58그룹, DINOv3 단독과 **ARI 0.95**(경계 유지)이면서 VLM 저신뢰 그룹
  9→5로 감소(이름 안정화). CLIP 벡터를 `photo_analysis.clip_embedding`에 저장하는 이유 =
  NAMING 단독 잡이 재분석 없이 돈다.
- **naming 3층** — ② 이름: 상수 K 대신 **커버리지 목표 85%**(상한 120그룹)까지 크기순으로
  골라 대표 1~2장(spread>0.12면 중심 최근접+최원점 2장)을 청크(이미지 15장) 호출 →
  통합 텍스트 호출 1회로 표기 통일. 부모는 촬영종류별 닫힌 목록을 JSON 스키마 enum으로 강제.
  ③ 배정: K 밖 소그룹은 최근접(거리>τ=0.25면 기타+review). CLIP 텍스트는 안 쓴다 —
  컨셉 층은 텍스트로 못 가른다. ④ 검증: CLIP zero-shot 부모 다수결, **일치 79%**(822장) —
  판정을 뒤집기엔 부족해서 needs_review 배지 근거로만.
- Bedrock 스키마 제약 발견: InvokeModel output_config는 number의 minimum/maximum 미지원(400)
  → confidence 범위는 프롬프트+읽기측 클램프.
- 실제 폴더 생성은 wes(`POST /folder-groups/ai`, 1회 = 세트 1개 = analysis_job_id).
- 연사 클러스터에 **멀티 카메라 파티션** 추가 — 카메라 바디별 파티션 + EXIF 시각 정렬
  (다른 카메라면 연사일 수 없다), 시각 없는 사진이 섞인 파티션은 정렬 포기(부분 정렬이 더 위험).

### 기능 2 — 폴더별 추천 + 이유 2단계 (V46)

- 폴더마다 독립: n_f = max(1, round(remaining·|f|/Σ|f'|)), 상한 ceil(|f|·0.5), 폴더당 최소 1
  (refine에서 목표를 채워도 대표 1장은 남긴다 — AI 마크가 사라지는 게 더 이상함).
  v2의 그룹 간 쿼터 경쟁(coverage_quota)이 사라져 배분식이 단순해짐.
- 점수는 **갤러리 내 백분위 그대로**(폴더 안 재정규화 없음 — "전체 상위 3%"가 말이 되려면).
- 미분류 가상 폴더(세트 밖 사진), 세트 없으면 409(FOLDERS_NOT_READY, 폴백 없음 — 폴백이
  있으면 "폴더별 추천"이 두 모양이 된다).
- **이유 2단계**(안 A/B/C 비교 후 C 채택): reason NULL INSERT(배지 즉시) → 큰 폴더부터
  LLM 생성 → UPDATE. 실측: 9폴더 30장 배분 0.02초, 사진 포함 이유 30건 136초 — 2단계가 답.
- 품질 하한 게이트(quality_floor_*): 백분위가 절대 품질을 소거하는 문제의 방어 —
  **문장에서 품질 표현만 빼고, 후보에서 사진을 거르지 않는다**(하드 필터 금지 유지).

### 기능 3 — 비교샷 동기 판정 (V47)

- 첫 대화형 워크로드. torch 미import(DB+PIL+Bedrock), 타임아웃 8s·재시도 0, 실패는 전부
  템플릿 판정 폴백(초점▸화질▸미학▸a, source=template) — UX는 "못 골랐어요"가 아니라
  "기준으로만 골랐어요".
- 항상 하나를 고른다(기권 없음, 비슷하면 slight). 사진에서 보이는 차이 우선, 백분위 차는
  5pt 이상만, 취향 문제는 "두 분 몫"이라 말하되 선택은 한다.
- 캐시 = 순서 무관 쌍 + model_version(모델 id+프롬프트 세대) — 프롬프트를 올리면 자동
  재판정·덮어쓰기. **compare-p1 실측에서 "b컷" 누출** → p2에서 a/b 내부 라벨 금지 규칙 추가.
- 실측: LLM 판정 5.4~6.0초(이미지 2장, 입력 ~2.1k 토큰), 서브프로세스 총 5.9초(예산 8s 안),
  캐시 히트 0초.
- 사람의 답(`pair_comparison_events`)과 분리 저장 — 나중에 AI 판정 일치율 평가 데이터.

---

## 4. photoselect_v1 — v3의 재구성 (2026-09-02)

코드 변화 없이 **구조만** 바꿨다: 기능별 패키지(foldering/recommend/compare + 공용 루트),
버전 분기(`--pipeline`, `pipeline_of`) 제거, `V3Knobs`→`Knobs`. 유지한 계약:
`model_version = "photoselect-v3-a-0.1"`(재개 판정이 이 문자열), 결과 payload `pipeline: "v3"`.

재구성 후 정리한 것(서비스 범위 = 결혼식 전 스튜디오 컨셉 촬영):

- **본식·피로연 제거** — `PARENTS` CEREMONY/OTHER 분기와 `shoot_type` 배선 전부 삭제.
  부모 목록은 리허설 7종 평면 리스트 하나.
- **face_boxes 제거** — v3에 얼굴 검출이 없어 항상 '{}'였다. wes V48이 컬럼 드롭.
  P1(앨범 크롭 안전)이 바운딩 박스를 쓰게 되면 그때 계약으로 새로 만든다.
- 미사용 헬퍼(measure_image, LaionRunner.score, 러너 name/version 메타, sel_share,
  mmr candidates 기본값) 제거.
- wes 로컬 스크립트 3종(local-worker/-ai/-compare.sh)을 photoselect_v1으로 전환,
  Ollama(VLM) 블록 삭제.

## 5. 연구 기록 인덱스 — 어디에 뭐가 남아 있나

| 문서 | 내용 |
|---|---|
| `docs/plan-v2-slim.md` | v1→v2 판단 근거 전체 (부품별 삭제 이유·실측 수치·목표 파이프라인) |
| `docs/analysis-audit-gallery1.md` | 갤러리 1(822장) 실데이터 감사 — v1 해체의 근거 |
| `docs/reasons-evolution.md` | 이유 문장 3단 진화 (캡션 → 사실 분해 → 셀렉터의 말) |
| `docs/paper-digest.md` | 모델 논문 요약 — 어떤 점수가 분해 가능한가 |
| `docs/plan-v3-folder-compare.md` | v3 설계 정본 + 구현 노트(계획과 달라진 점) |
| `docs/review-v3-design.md` | v3 설계 리뷰 (커버리지 목표, spread 2장 대표, 품질 하한, 대칭 적응 등) |
| wes `docs/plans/ai-folder-structure.md` | 폴더화 계획·concat/ARI 실측 |
| `docs/plan-preference-probe.md` | 다음 계획 — v1 `fit_bt` 수식을 임베딩 위 probe로 재활용 |
| `docs/tech-stack.md` · `docs/e2e-test-plan.md` | 모델 채택 실측치 · 로컬 E2E 절차 |
| (삭제됨, git bbaf204) `scripts/spike/` `scripts/arniqa_regressors.py` `scripts/review.py` | 스파이크 하네스 원본 — 결과 수치는 위 문서들과 이 문서에 채록됨 |

## 6. 다시 필요해지면

| 잃은 능력 | 어디서 되찾나 |
|---|---|
| 눈감김·표정 신호 | v1 `runners/faces.py` (bbaf204). 복귀 조건: 2단계 검출(실측 18→68%)로 재설계 |
| VLM 축 태그·캡션 | v1 `analyze/vlm.py` + 프롬프트 실측 교훈(§1) — 사실상 재설계 권장 |
| 부부 단위 취향 학습(BT) | v1 `draft/preference.py`의 `fit_bt` — preference probe 계획이 수식을 계승 (입력만 one-hot→임베딩) |
| 자연어 피드백 → 가중치 | v1 `llm/feedback.py` — 가드레일 설계(어휘 밖 버림·클램프·실패=빈 목록)가 참고할 자산 |
| 갤러리 전체 k장 묶음 추천 | v2 `draft.py`+`rerank.coverage_quota` (그룹 간 쿼터 경쟁 버전) |
| 본식(CEREMONY) 부모 목록 | photoselect_v1 정리 커밋 직전 이력 — 단, 본식 데이터로 재설계가 맞다 |
