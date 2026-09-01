# Photoselect v2 — 덜어낸 추천 파이프라인 계획

작성 2026-08-29. 전제: `docs/analysis-audit-gallery1.md`(갤러리 1 실데이터 감사)와
현재 코드(`photoselect/feat/5-llm-reasons`). 제품 원칙(하드 필터 금지, AI는 초안만, 최종
결정은 사람)은 유지한다.

## 0. 제안된 4단계에 대한 판단

| 단계 | 제안 | 판단 | 단서 |
|---|---|---|---|
| 1 | 미학·기술 점수로 줄세우기 + 이유 | **동의.** 현재 prior(0.5·tech_pct + 0.5·aes_pct)가 이미 이것. 갤러리 1에서 두 점수는 독립(상관 −0.05)이고 추천 평균 백분위 75/80으로 작동 중 | "왜 높은가"는 두 층으로 나눈다. ① 모델이 *무엇을* 재는지는 항상 말할 수 있다(기준 설명). ② 사진별 *분해*는 ARNIQA/LAION 스칼라로는 불가 → 기술 점수만 고전 지표(선명도·노출)를 곁들여 사진별 이유를 만든다. 미학은 기준 설명 + 순위로 그친다 — 정직한 한계 |
| 2 | 유사 사진(연사·같은 컨셉)은 점수가 높아도 다양성 우선 | **동의, 그리고 VLM 없이 된다.** 연사는 이미 임베더 벡터 0.96으로 묶인다(301 클러스터, DINOv2 실측). 컨셉은 같은 벡터를 느슨하게(코사인 ≥ 0.8) 묶으면 70개 그룹이 나오고 케이크 세트·소파·덩굴 아치·해변·정원 등 실제 촬영 세트와 일치했다(§3.2 실측) | "컨셉"의 *이름*은 못 붙인다. 다양성 확보와 "비슷한 배경의 n장 중"이라는 근거에는 이름이 필요 없다. 이름이 필요한 건 4단계뿐 |
| 3 | 거짓 아닌 한 과장해도 좋으니 추천 이유 제공 | **동의. 현재 구조(사실 → 템플릿 → LLM 다듬기)가 정확히 이것.** 문제는 사실 재료의 절반이 신뢰 불가(scene 라벨, 얼굴 신호)라는 점 | 재료를 "검증된 것"만으로 줄인다: 점수 순위·기술 세부지표·연사 내 비교·컨셉 그룹 크기·(4단계 후) 유형 균형. 사진에 없는 것을 말할 가능성이 있는 재료(caption, VLM expression, scene 이름)는 뺀다 |
| 4 | 선택·별점이 쌓이면 유형별 선호·부족분 반영 | **동의하되 현재 BT(Bradley-Terry) 기계는 과하다.** 제안한 예시("별점이 이 유형에 높다", "남편만 n장 vs 아내만 n−2장")는 **집계 규칙**으로 그대로 표현되고 이유도 그 자체로 나온다. BT는 33차원 회귀 + LOO 신뢰도 + λ 곡선인데, 지금까지 증거 0으로 한 번도 돌지 않았고 결과를 사람이 설명하기도 어렵다 | "유형"은 신뢰 가능한 축 하나로 시작: `subjects`(신부/신랑/커플/단체). VLM은 빼되 CLIP zero-shot으로 대체 가능한지 스파이크로 확인. 별점 사용은 **정책 변경**(현재 CLAUDE.md는 `photo_ratings` 접근 금지) |

한 줄 결론: **덜어내는 데 동의한다.** 지금 파이프라인에서 추천 결과를 실제로 바꾸는 부품은
prior·임베딩 클러스터·MMR·재노출 금지·목표 장수 게이트 다섯 개뿐이고, 나머지(VLM 5축·캡션,
얼굴 4신호, BT·λ·conf, 피드백 번역, 온보딩 쌍비교, 9종 근거)는 비용만 내고 있다.

## 1. 뺄 것 / 남길 것 / 새로 넣을 것

### 뺀다

| 부품 | 이유 | 영향 |
|---|---|---|
| VLM(gemma3:12b) 전체 — scene·framing·lighting·expression·subjects·caption | 런타임 96%(5,468/5,709초), GPU 필수. 결과에 작용하는 축은 scene 하나인데 라벨 오류가 크고(detail 82장 전부 인물, 스튜디오에 walk 174장), lighting은 87% 단일값, caption은 근거 재료로만 쓰이며 칭찬어 위반 | 배치 A가 CPU ~1초/장 → Lambda 사정권. scene 쿼터 → 컨셉 그룹 쿼터로 대체 |
| MediaPipe Face Landmarker — face_count·eyes_open·smile·max_face_ratio | 81% 얼굴 미검출(short-range 검출기 한계), smile·max_face_ratio는 읽는 코드 없음 | 눈감김 회피 기능 상실 — 갤러리 1에서 실제 눈감김 4장. 필요해지면 2단계 검출(실측 18→68%)로 복귀 |
| Bradley-Terry·axis_confidence·λ(n)·evidence.to_pairs·top_preferences | 증거 0으로 미작동. 4단계는 집계 규칙으로 대체 | `draft/preference.py`, `draft/evidence.py`, `scoring.combine`의 λ 혼합 삭제 |
| 자연어 피드백 번역(`llm/feedback.py`) | 생산자(wes API·프론트) 없음 | 후속 과제로 보류 |
| 온보딩 쌍비교(`pair_comparison_events`, review 페이지 생성기) | 생산자 없음, BT 제거로 소비자도 없음 | 테이블은 wes 소유라 그대로 둠 |
| 근거 종류 preference·similar·rarity·coverage(장면명)·burst·moment | 신뢰 불가 재료 의존 | 근거 종류 4개로 축소(§3.4) |
| `axes.py` 33차원 one-hot | 소비자 소멸 | `subjects` 단일 축만 남김(4단계용) |

### 남긴다 (검증됨)

- ARNIQA(spaq) `technical_score` → 갤러리 내 백분위
- LAION Aesthetic v2 `aesthetic_score` → 백분위 (절대값은 쓰지 않는다 — IQR 0.31)
- 임베더 임베딩(DINOv3 ViT-B/16 — 2026-08-29 wes embedder에서 DINOv2→v3 교체, 768d 동일) + 연사 클러스터(순서 ≤ 8, 코사인 ≥ 0.96) + 대표 선정
- MMR(0.7·score − 0.3·maxCos), 클러스터당 1장, 재노출 금지, 목표 장수 게이트
- 사실 → 결정적 템플릿 → LLM 다듬기(텍스트만, 폴백 템플릿) 구조
- 하드 필터 금지, `photo_selections.status` 불가침, MANUAL 행 읽기 전용

### 새로 넣는다

| 부품 | 무엇 | 왜 |
|---|---|---|
| 고전 기술 지표 | Laplacian 분산(선명도), 하이라이트/섀도 클리핑 비율(노출), 얼굴 무관. OpenCV/NumPy, 사진당 < 50ms | 1단계의 "기술 점수가 왜 높은가"를 사진별로 말할 유일한 방법. ARNIQA 백분위와 함께 제시("전체 상위 12%, 특히 선명도가 그룹 최고") |
| 컨셉 그룹 `concept_id` | 임베더 벡터 평균연결 계층 클러스터, 코사인 거리 ≤ 0.2 (DINOv2 실측값 — DINOv3로 재측정) | 2단계의 "같은 컨셉" 다양성. VLM scene 대체. 갤러리당 O(N²) — 822장 0.5초 |
| `subjects` 태거 (CLIP zero-shot) | 이미 계산하는 CLIP ViT-L/14 임베딩 + 텍스트 프롬프트 4개(신부 단독/신랑 단독/둘/단체) | 4단계 "유형"의 최소 단위. 추가 비용 0. **정밀도 스파이크 통과가 조건**(VLM 기준 0.98 대비) |
| 유형 선호·균형 집계 (4단계) | 별점 lift, 선택 비율 lift, 유형별 선택 부족분 | BT 대체. 결과가 곧 이유 문장 |

## 2. 목표 파이프라인

```
배치 A (갤러리당 1회, CPU)
  photos(EMBEDDED) → preview 다운로드
  → ARNIQA → technical_score        → technical_pct
  → CLIP 임베딩 → LAION MLP → aesthetic_score → aesthetic_pct
                → zero-shot subjects (스파이크 통과 시)
  → 고전 지표 → sharpness, exposure_clip  (sub_scores)
  → 임베더 E(DINOv3) → 연사 cluster_id / cluster_rank (기존)
              → concept_id (신규, 느슨한 임계)
  → photo_analysis UPSERT

배치 B (라운드당 1회, Lambda)
  score = 0.5·z(tech_pct) + 0.5·z(aes_pct)                ← 1단계
  (4단계 이후) score += w_pref · affinity(subjects) + w_bal · deficit(subjects)
  exclude = 담김 ∪ 이전 라운드 노출 ∪ 거절
  후보 = 연사 클러스터당 대표 1장                            ← 2단계 (연사)
  concept 쿼터: 그룹당 최소 1, 비례, cap 40%·k               ← 2단계 (컨셉)
  그룹 안에서 MMR → k장
  근거: 사실 → 템플릿 → LLM 다듬기                           ← 3단계
  ai_recommendations INSERT
```

## 3. 단계별 상세

### 3.1 1단계 — 줄세우기와 "기준 설명"

- 점수는 지금과 동일: `0.5·technical_pct + 0.5·aesthetic_pct`(z 표준화). 가중치는 다이얼로
  남기되 기본 반반.
- 근거 문장의 고정 재료(사진과 무관하게 참인 문장, 프롬프트 코드북에 둔다):
  - 미학: "구도·색감·피사체 배치를 수십만 장의 사람 평가로 학습한 모델의 점수(LAION Aesthetic)"
  - 기술: "초점·흔들림·노이즈·노출 이상을 재는 모델의 점수(ARNIQA)"
- 사진별 재료: 갤러리 내 백분위(상위 n%), 고전 지표의 상대 위치(연사/컨셉 그룹 내 선명도
  1위, 노출 클리핑 최소). 백분위는 **상위 15% 이상일 때만** 문장에 쓴다(기존 규칙 유지).
- 미학 점수의 사진별 이유는 만들지 않는다. "왜"를 묻는 사용자에게는 기준 설명으로 답한다.
  과장은 허용하되 검증 불가한 속성(표정·시선·감정)은 재료에 넣지 않는다.

### 3.2 2단계 — 연사와 컨셉, 두 층의 다양성

연사(기존): 순서 ≤ 8 ∧ 코사인 ≥ 0.96 → 301 클러스터. 유지.

컨셉(신규): 갤러리 1 **DINOv2** 실측(임베더가 DINOv3로 바뀌었으므로 재임베딩 후 같은 스크립트로 재측정 — 임계 0.2는 잠정값)(평균연결 계층 클러스터, 코사인 거리 임계별):

| 임계 | 그룹 수 | 상위 그룹 | 해석 |
|---|---|---|---|
| 0.10 | 144 | 33·25·23·22 | 연사 몇 개가 합쳐진 수준 |
| **0.20** | **70** | 50(케이크·샴페인 세트) · 41(해변 배경) · 39(소파) · 37(정원) · 36(덩굴 아치) | **촬영 세트 단위와 일치** — 채택 |
| 0.30 | 43 | 62(소파+정원 합침) | 다른 세트가 섞이기 시작 |

- 임계 0.2를 기본값으로 두고 갤러리 2~3개에서 재확인. `similarity_profile`처럼 그룹 수·
  최대 그룹 크기를 잡 결과에 기록해 튜닝 근거로 남긴다.
- 쿼터는 기존 `coverage_quota`를 scene → concept_id로 바꿔 재사용(최소 1·비례·cap 40%).
- 근거 재료: "비슷한 배경·구도의 사진 n장 중 점수 1위" — 그룹 이름 없이 성립.
- "야외 사진만 추천되는" 문제: 컨셉 그룹 cap 40%가 곧 그 방어. 실외/실내 이름이 필요하면
  4단계 zero-shot 축에 `outdoor/indoor`를 추가로 스파이크한다.

### 3.3 3단계 — 근거 문장

근거 종류를 9 → 4로 줄인다. 우선순위 순:

| 종류 | 조건 | 템플릿 예 |
|---|---|---|
| `balance` (4단계 후) | 담은 사진에서 이 유형이 부족 | "지금 담긴 사진에 신부 단독 컷이 신랑보다 2장 적어요 — 이 컷으로 균형을 맞출 수 있어요" |
| `sibling` | 같은 연사 클러스터에 후보 형제가 있음 | "비슷한 4장 중 가장 선명한 컷이에요 (선명도 1위, 기술 점수 상위 8%)" |
| `quality` | tech 또는 aes 백분위 상위 15% | "구도·색감 점수가 전체 상위 5%예요. 초점·노출도 이상 없이 깨끗해요" |
| `concept` | 컨셉 그룹 대표(그룹 크기 ≥ 3) | "이 배경으로 찍은 37장 중 점수가 가장 높은 컷이에요" |

- LLM 다듬기는 유지(배치 40장, ≤ 60자, photo_id 검증, 실패 시 템플릿). 코드북에 위 기준
  설명을 넣어 "과장은 되 거짓은 안 되게" 한다. 사진에 없는 것(사람·표정·장소명)은 재료에
  없으므로 말할 수 없다.
- Lambda 경로에도 LLM 클라이언트를 연결한다(현재 `handler.py`는 템플릿 전용 — 결함).

### 3.4 4단계 — 선호와 균형 (증거가 생기면)

유형 = `subjects` ∈ {bride, groom, couple, group}. 세 가지 집계, 전부 셀렉션 단위:

```
sel_share(t)   = 담은 사진 중 유형 t 비율
pool_share(t)  = 갤러리 중 유형 t 비율
deficit(t)     = max(0, pool_share(t) − sel_share(t))          # "부족한 유형"
rating_lift(t) = mean(별점 | 유형 t) − mean(별점 전체)           # "이 유형에 별점이 높다"
select_lift(t) = P(담김 | 유형 t) − P(담김)                       # "이 유형을 잘 담는다"
affinity(t)    = w_r · rating_lift(t) + w_s · select_lift(t)

score(p) = z(prior) + w_pref · affinity(subjects(p)) + w_bal · deficit(subjects(p))
```

- 발동 조건: 담은 사진 ≥ 5장 또는 별점 ≥ 5건. 그 전엔 항 자체가 0 — λ 곡선 불필요.
- 이유 문장은 집계값을 그대로 읽는다: "별점을 준 사진 중 신부 단독 컷의 평균이 4.5로 가장
  높아요", "담은 사진에 신랑 단독 n장 vs 신부 단독 n−2장".
- **정책 결정 필요**: `photo_ratings`는 현재 "개인 취향 신호라 AI 입력 제외"(CLAUDE.md).
  사용자 요구는 별점 반영이므로 CLAUDE.md·`store.py:12`의 금지를 해제하는 결정을 명시적으로
  내려야 한다. 결정 전까지는 `select_lift`·`deficit`만으로 4단계를 켠다.
- "담은 사진과 비슷한 컷"(코사인 ≥ 0.6) 근거는 유지 가치가 있으나 3단계 4종에 안 넣었다 —
  균형(deficit)과 방향이 반대(비슷한 걸 더 담게 유도)라서, 4단계 실험 후 결정.

## 4. 현재 코드에서 바꿀 것

| 파일 | 변경 |
|---|---|
| `analyze/job.py` | VLM·FacesRunner 호출 제거. 고전 지표 러너 추가. `concept_id` 계산 추가. `MODEL_VERSION` → `photoselect-a-1.0` |
| `analyze/runners/faces.py`, `analyze/vlm.py` | 삭제(또는 `scripts/spike/`로 이동). 스파이크 하네스는 보존 |
| `analyze/runners/classical.py` (신규) | Laplacian 분산, 클리핑 비율 |
| `analyze/runners/laion.py` | `zero_shot_subjects(emb)` 추가(스파이크 통과 시) |
| `analyze/cluster.py` | `concept_groups(E, dist)` 추가(scipy 평균연결) |
| `axes.py` | `subjects`만 남김. `AXIS_PRECISION`은 스파이크 결과로 갱신 |
| `draft/job.py` | evidence→BT→λ 블록 삭제. scene → concept 쿼터. 근거 4종. 4단계 집계 함수 |
| `draft/preference.py`, `draft/evidence.py` | 삭제. 코드는 `study/`에 참고로 남김 |
| `draft/scoring.py` | `combine` → prior + pref/bal 가산 |
| `draft/rerank.py` | 인터페이스 유지, 입력만 concept_id |
| `llm/reasons.py` | 코드북에 모델 기준 설명 추가, 근거 4종 예시로 교체 |
| `llm/feedback.py` | 보류(삭제 대신 미연결) |
| `handler.py` | LLM 클라이언트 연결 |
| `store.py` | `read_evidence`: pairs 제거, ratings 읽기(정책 결정 후), selected는 유지. `PhotoAnalysis`에 `concept_id`, sub_scores 키 추가 |
| `config.py` | `AnalyzeKnobs`: vlm_* 제거, `concept_dist=0.2` 추가. `ScoreKnobs`: λ·bt·conf·w_pair·w_rating 제거, `w_pref`·`w_bal`·발동 임계 추가 |
| `scripts/review.py` | 온보딩 쌍비교 섹션 제거, 별점·담기만 유지 |
| `docs/feature-design.md`, `README.md`, `CLAUDE.md` | 축 어휘·모델 표·정책 갱신 |

### wes 스키마 요청 (이 repo는 마이그레이션을 만들지 않는다)

- `photo_analysis.concept_id INTEGER` 추가.
- `ck_photo_analysis_analyzed` CHECK가 scene·framing·lighting·expression을 NOT NULL로 강제 —
  VLM 제거 시 해제 필요. 마이그레이션 전까지는 `unknown` 등 기본값을 써서 통과시킨다.
- 컬럼 삭제(scene·framing·lighting·expression·caption·face_boxes)는 서두르지 않는다 —
  프론트 `analysis` 응답이 읽고 있으므로 wes/프론트와 함께 정리.
- 별점 반영 결정 시 `photo_ratings` 읽기 권한 정책 변경(코드가 아니라 문서·규칙).

## 5. 순서와 검증

| 단계 | 산출물 | 완료 기준 |
|---|---|---|
| S0 스파이크 (1~2일) | `scripts/spike/concept_threshold.py`, `zero_shot_subjects.py`, `classical_metrics.py` | 갤러리 1·2·3에서 concept 그룹이 세트 단위와 맞는지 눈으로 확인. subjects zero-shot 정밀도 ≥ 0.9(50장 채점셋). 고전 선명도와 ARNIQA 백분위 상관 > 0.3(같은 방향인지만 확인) |
| S1 배치 A 슬림 | VLM·얼굴 제거, 고전 지표·concept 추가, MODEL_VERSION 1.0 | 갤러리 1 전수 재분석 < 15분(CPU). `perStageSeconds` 기록 |
| S2 배치 B 슬림 | BT 제거, concept 쿼터, 근거 4종, Lambda에 LLM | 갤러리 1 라운드 1·2 결과를 현재 결과와 나란히 review 페이지로 비교. 근거 문장 60개 전수 검독: 사진에 없는 것을 말한 문장 0건 |
| S3 4단계 | 집계 함수 + 근거 `balance` | 담은 사진 11장인 셀렉션 17에서 deficit이 계산되고 이유가 맞는지 확인 |
| S4 정리 | 문서·스키마 요청·dead code 삭제 | CLAUDE.md, feature-design, README 갱신 |

S0에서 subjects zero-shot이 0.9에 못 미치면: 4단계 유형 축을 당장은 포기하고 컨셉 그룹
기반 균형("이 배경의 사진이 아직 안 담겼어요")으로 대체한다. 다른 단계는 영향 없다.

## 6. 남는 리스크

- **미학 점수 분포가 좁다**(IQR 0.31/10). 백분위 5pt 차이가 원점수 0.03 — 형제 간 "인상"
  비교는 노이즈일 수 있다. sibling 근거는 기술 지표(선명도·ARNIQA)를 우선하고 미학은
  전체 상위 15%일 때만 언급한다.
- **눈감김 컷 회피 상실.** 갤러리 1에서 4장(0.5%). 클레임이 생기면 2단계 얼굴 검출을
  `classical.py` 옆에 되살린다(실측 코드 `scripts/spike/face_probe.py`).
- **컨셉 임계가 갤러리마다 다를 수 있다.** 야외 스냅·본식 갤러리에서 재측정. 그룹 수가
  3 미만이거나 N/2 초과면 임계를 자동 조정하는 폴백을 둔다.
- **CLIP zero-shot 편향**(신부 드레스 = 흰색 등 색에 의존). 채점셋으로 확인 전엔 근거
  문장에 인용하지 않는다.
- 배치 A가 CPU로 갤러리 800장에 ~14분 → Lambda 15분 한계 근접. 잡을 200장 단위로
  이어 돌리는 재개 로직은 이미 있다(`model_version` 스킵).

## 7. 진행 방식 — v1/v2 병존 (2026-08-29 구현)

결정: 데이터(S3·로컬 DB)를 전부 비우고 업로드부터 다시 검증하기로 했으므로 "v2 는 로컬에만 쓴다" 제약은
불필요해졌다. 대신 코드를 `photoselect/v1`·`photoselect/v2` 로 나눠 두고 `--pipeline` 으로 고른다(기본 v2).

- v1↔v2 import 없음. `store`·`config`·`gallery`·`runners`·`cluster`·`llm/client` 까지 버전별 사본 — 루트에는
  환경 `Settings`·`db`·`jobs`·`storage`·배관(`worker` `handler` `__main__`)만 남고, 배관은 각 버전의 파사드
  (`v1/__init__.py`, `v2/__init__.py`)만 부른다. 로컬 출력은 `out/<g>`(v1) / `out/v2/<g>`(v2).
- 같은 갤러리에 v1·v2 를 번갈아 돌리면 `photo_analysis` 가 덮어써진다(`model_version` 접두사가 달라 서로
  재개 대상으로 보지 않음). 비교가 필요하면 갤러리를 둘 만들거나 v1 은 `--pipeline v1` 로 로컬 모드에서.
- 비교 기준은 그대로: ① 블라인드 세트 선호 ② 근거 문장 검독(사진에 없는 것을 말한 문장 수) ③ 세트 내
  컨셉 점유율·연사 중복 ④ 런타임.
- v2 확정 시 `v1/` 삭제, wes 에 `concept_id`·CHECK 완화 요청.
