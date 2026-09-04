# 01 추천 시스템 기초 — 학습 자료

> URL은 2026-08-23 접근 검증(미검증 섹션 제외). 학습 완료한 자료는 체크한다.
> **자료는 혼자 읽지 않는다.** 각 항목의 `→ lab/`는 함께 돌릴 실습이다 (`lab/README.md`).

## 0. 이 영역의 문제 정의 — 무엇을 배우고 무엇을 버리는가

AI 셀렉터는 통상적 의미의 추천 시스템(협업 필터링·행렬 분해)이 아니다.

| | 고전 추천 (CF/MF) | AI 셀렉터 |
|---|---|---|
| 신호 | 다수 사용자 × 다수 아이템 행렬 | 부부 1팀의 쌍 비교 8~12개 + 별점 |
| 핵심 가정 | "나와 비슷한 사용자가 좋아한 것" | 협업 신호가 존재하지 않음 |
| 아이템 풀 | 전체 카탈로그 공유 | 갤러리마다 사진 집합이 완전히 다름 |
| 콜드 스타트 | 예외 상황 | **모든 사용자의 기본 상태** |

배울 것은 세 덩어리다: **① pairwise 선호 학습(Bradley-Terry) ② prior와 취향의 결합
(shrinkage/λ) ③ 다양성 재랭킹(MMR)과 평가·실험 설계.** 행렬 분해는 쓸 자리가 없다.

## 학습 순서 (이 순서대로 하면 된다)

| # | 하는 일 | 시간 |
|---|---|---|
| 1 | TDS BT 튜토리얼 읽기 | 40분~1시간 |
| 2 | `lab/step1_bt.py` 실행 + 자가 점검 | 30분 |
| 3 | RLHF Book Ch.5 읽기 | 1.5~2시간 |
| 4 | `lab/step2_features.py` · `step3_lambda.py` | 1시간 |
| 5 | Elastic MMR 글 → `lab/step4_mmr.py` | 1시간 |
| 6 | `lab/step5_eval.py` (+ 02-ml-stats 검정 부분) | 1시간 |
| 7 | `lab/step6_pipeline.py` 로 한 바퀴 | 30분 |
| 8 | Google 코스 개요 섹션 · 기술블로그 | 1~2시간 |
| 9 | MMR 원논문 · Jongya 지표 글 (W3 이후) | 1.5시간 |

합계 약 9~11시간. **읽기 5 : 돌려보기 5** 비율을 지킨다.

## 필수

- [ ] **1. TDS — Learning From Pairwise Preferences: Intro to Bradley-Terry** · 40분~1시간 · 영어
  https://towardsdatascience.com/learning-from-pairwise-preferences-an-introduction-to-the-bradley-terry-model/
  `P(i≻j)=π_i/(π_i+π_j)`, 로그우도 최대화, Chatbot Arena 응용. **수식을 처음 볼 때 이것부터.**
  RLHF Book보다 진입장벽이 훨씬 낮다. → `lab/step1_bt.py`
- [ ] **2. `lab/step1_bt.py` 실행** · 30분
  BT MLE를 직접 구현한 것과 sklearn 로지스틱 회귀의 계수가 소수점 6자리까지 같다는 것을
  눈으로 확인한다. 이 등가성을 손에 넣고 나면 다음 자료의 체감 난이도가 내려간다.
- [ ] **3. RLHF Book (Nathan Lambert) — Ch.5 Reward Modeling** · 1.5~2시간 · 영어
  https://rlhfbook.com/c/05-reward-models
  "보상 모델의 표준 구현은 Bradley-Terry에서 유도된다" — chosen/rejected contrastive loss,
  MLE, Python 구현. **홀드아웃 쌍 비교 평가의 수학이 정확히 이 챕터.**
  1·2번을 먼저 하고 오면 "아는 걸 다시 보는" 독서가 된다.
- [ ] **4. `lab/step2_features.py` + `lab/step3_lambda.py`** · 1시간
  p≫n·정규화·식별 문제 / λ=f(증거량)·conf(axis). 설계안 §3-B 점수식의 모든 항이 여기 있다.
- [ ] **5. Elastic Search Labs — MMR로 검색 결과 다양화** · 30분 · 영어
  https://www.elastic.co/search-labs/blog/maximum-marginal-relevance-diversify-results
  λ 값별 가이드(0.3~0.9) + NumPy `maximal_marginal_relevance()` 구현. → `lab/step4_mmr.py`
- [ ] **6. `lab/step5_eval.py`** · 1시간
  유의성·검정력·필요 표본 수·군집 데이터. **W1~2 실험 설계를 여기서 확정한다.**
  통계 배경이 약하면 `../../02-ml-stats/materials.md`의 가설검정 항목을 병행.
- [ ] **7. `lab/step6_pipeline.py`** · 30분 — 온보딩→초안→피드백→재추천 한 바퀴.

## 그다음 (전체 그림·구현 참고)

- [ ] **Google — Recommendation Systems 코스** · **개요 섹션만** 1~2시간 · 영어(`?hl=ko` 번역 있음)
  https://developers.google.com/machine-learning/recommendation
  Retrieval → Scoring → Re-ranking 2단계 아키텍처가 우리 파이프라인(전수 분석 → 점수 → MMR)과
  1:1 대응한다. **행렬 분해(MF)·softmax DNN 실습 섹션은 건너뛴다** — 우리 문제에 안 쓴다.
  전체 4~8시간을 다 듣는 것은 과투자.
- [ ] **Carbonell & Goldstein (1998) MMR 원논문** · 4쪽, 1시간 · 영어 — **구현 후에** 읽는다
  https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf
  step4를 돌린 뒤 읽으면 "내가 짠 게 이거였구나"로 30분에 끝난다.
- [ ] **우아한형제들 — 실시간 반응형 추천 개발 일지 1부** · 30~40분 · 한국어
  https://techblog.woowahan.com/17383/ — Kafka + 임베딩 + **pgvector 코사인 검색**.
  스택이 겹치는 가장 가까운 실전 사례.
- [ ] **카카오 — 토픽 모델링과 MAB 개인화 추천** · 20~30분 · 한국어
  https://tech.kakao.com/2021/06/25/kakao-ai-recommendation-01/ — 탐색/활용 관점의 콜드 스타트.
  온보딩 쌍을 "어떤 쌍을 물어볼 것인가"(정보 획득)로 보는 시각을 준다.

## W3 이후 (랭킹 리스트 평가를 시작할 때)

- [ ] **Jongya's blog — 추천 품질 평가지표 4가지 (HitRate·MRR·MAP·NDCG)** · 40분~1시간 · 한국어
  https://whdrns2013.github.io/recommend_system/20240819_001_recommend_quan_eval/
  주지표가 쌍 정확도인 동안은 급하지 않다. 수락률·nDCG 리포트를 짤 때(로드맵 #9) 읽는다.
  `lab/metrics.py`에 nDCG 구현이 이미 있으니 대조하며 읽을 것.
- [ ] **한국어 위키 — 인간 피드백을 통한 강화 학습** · 15분
  https://ko.wikipedia.org/wiki/인간_피드백을_통한_강화_학습 — BT↔RLHF 연결을 한국어로 재확인.

## 제외 — 여기에 시간을 쓰지 않는다

| 자료 | 제외 이유 |
|---|---|
| 인프런 — Python 개인화 추천시스템 (6h14m) | CF/MF 중심. "콜드 스타트"를 다루지만 그건 *상호작용 이력이 없는 신규 유저*의 문제이고, 우리 문제는 *이력이라는 개념 자체가 없는* 구조다. 대응 방법이 다르다 |
| 인프런 — 추천 시스템 입문편 (7h34m) | 위와 동일. CF 심화는 얕고 우리에게 전이되는 부분이 거의 없다 |
| T아카데미 — 추천시스템 분석 입문하기 (4~6시간) | 위와 동일 (CF 중심 무료 강의) |
| Google 코스의 MF·softmax DNN 실습 | 행렬 분해를 쓰지 않는다 |

CF가 필요해지는 시점은 "여러 커플의 취향을 가로질러 학습"할 때인데, 그건 현재 설계 범위 밖이고
데이터도 없다. 필요해지면 그때 위 강의로 돌아온다.

## 커버되지 않은 개념 — 스스로 근거를 만들어야 하는 부분

- **선호 도출(preference elicitation)**: 검증된 단독 자료 없음. "온보딩 8~12쌍을 어떻게 고르나"는
  카카오 MAB 글(탐색/활용) + BT 자료 + `lab/step2_features.py` 실험 C(쌍 규칙 비교)로 스스로 논증한다.
- **실험 설계·검정력**: 추천 시스템 자료에는 거의 안 나온다. `lab/step5_eval.py`와
  `../../02-ml-stats/`에서 채운다. **W1~2에 실제로 발목을 잡는 건 알고리즘이 아니라 이쪽이다.**

## 미검증 (직접 확인 필요)

- 당근 — 딥러닝 개인화 추천 https://medium.com/daangn/딥러닝-개인화-추천-1eda682c2e8c (Medium 봇 차단)
- 위키독스 — MMR 검색(한국어) https://wikidocs.net/231585 (봇 차단)
- Dr. Tag Kim — Bradley-Terry Model (한국어, 2015, R 중심)
  http://drtagkim.blogspot.com/2015/11/bradley-terry-model-paired-comparison.html
