# 02 ML/통계 기초 — 학습 자료

> 모든 URL은 2026-08-23 실제 접근 검증됨 (미검증 섹션 제외). 학습 완료한 자료는 체크한다.
> 우선 학습 대상: **가설검정·표본 크기** (W1~2 쌍 비교 실험에 직결). 나머지는 W5 이후.

## 필수

- [ ] **StatQuest (Josh Starmer, YouTube)** · 개념당 5~20분, 선별 3~5시간 · 영어(쉬움)
  전체 색인: https://statquest.org/video_index.html
  볼 것: Bias and Variance / Cross Validation / Confusion Matrix / ROC and AUC /
  Confidence Intervals / p-values / How to Calculate p-values / Sample size.
  최소 시간으로 전 개념 직관 확보 — 이번 주 안에 가능.
- [ ] **Coursera — Machine Learning Specialization (Andrew Ng)** · 청강 무료 · 영어(한글 자막)
  https://www.coursera.org/specializations/machine-learning-introduction
  Course 1(지도학습·과적합·정규화, 33h) → Course 2(평가·bias/variance 진단, 34h)만 우선.
  Course 2의 "Advice for applying ML"이 "홀드아웃으로 뭘 판단할 것인가"에 직결.
- [ ] **Understanding empirical Bayes estimation (baseball statistics) — David Robinson**
  · 블로그 30분~1시간 · 영어
  https://www.r-bloggers.com/2015/10/understanding-empirical-bayes-estimation-using-baseball-statistics/
  **우리 λ 설계가 정확히 이 글의 shrinkage다** — "10타수 4안타 vs 1000타수 300안타는
  증거량이 다르므로 다르게 신뢰"를 고객 취향 신호로 바꿔 읽으면 설계 근거 문서가 됨.

## 가설검정·표본 크기 (홀드아웃 실측 직전 필수)

- [ ] **데이터 사이언스 스쿨 — 9.5 사이파이를 사용한 검정** · 교재 1~2시간 · **한국어**
  https://datascienceschool.net/02%20mathematics/09.05%20%EC%82%AC%EC%9D%B4%ED%8C%8C%EC%9D%B4%EB%A5%BC%20%EC%82%AC%EC%9A%A9%ED%95%9C%20%EA%B2%80%EC%A0%95.html
  이항검정 = "홀드아웃 N쌍에서 맞춘 횟수가 찍기(50%)보다 나은가"와 동일 구조.
  `scipy.stats.binomtest(k, n, 0.5)` 한 줄. **핵심 감각: n=30에서 18/30(60%)은 단측
  p≈0.18로 유의하지 않음 — 30쌍이면 ≥21맞춤(70%)부터 p<0.05.**
- [ ] **Epitools — 비율 신뢰구간 계산기** · 도구 10분 · https://epitools.ausvet.com.au/ciproportion
  Wilson/Clopper-Pearson 등 5종. "18/30"을 넣어 구간이 0.5를 포함하는지 즉시 확인 —
  스파이크 리포트 검산 도구.
- [ ] **공돌이의 수학정리노트 — 신뢰구간 / p-value** · 각 20~30분 · **한국어**
  https://angeloyeo.github.io/2021/01/05/confidence_interval.html
  https://angeloyeo.github.io/2020/03/29/p_value.html
  "유의하다 ≠ 크게 낫다" — 소표본 결과를 팀에 보고할 때의 언어.

## 선택

- [ ] **공돌이의 수학정리노트 — 베이즈 정리의 의미** · 20분 · 한국어
  https://angeloyeo.github.io/2020/01/09/Bayes_rule.html — empirical Bayes 글 읽기 전 준비물.
- [ ] **dontech — 편향-분산 트레이드오프** · 30분 · 한국어 · https://donmain.dev/ml/bias-variance/
  진단표(훈련 성능 나쁨=고편향, train-val 격차=고분산)와 처방 — λ에서 취향 항 비중을
  늘릴지 줄일지 판단하는 사고방식과 동형.
- [ ] **김성범 교수 — 핵심 머신러닝 (고려대, YouTube)** · 선별 5~10시간 · 한국어
  https://www.youtube.com/playlist?list=PLpIPLT0Pf7IoTxTCi2MEQ94MZnHaxrP0j
  StatQuest의 한국어 대응물 + 더 깊은 수식. 직관 다음 단계.
- [ ] **인프런 — 파이썬 머신러닝 완벽 가이드 (권철민)** · 유료(세일가 수만 원대) · 192강 37.5h
  https://www.inflearn.com/course/파이썬-머신러닝-완벽가이드
  평가 챕터(교차검증·AUC)가 독립 섹션. 1~3장만 우선. 결제 전 커리큘럼 확인.
- [ ] **인프런 — 파이썬 기초부터 쌓아가는 머신러닝 (거친코딩)** · 무료(확인 시점) · 25강 9h
  https://www.inflearn.com/course/파이썬-머신러닝 — 유료 강의 전 무료로 발 담그기.
- [ ] **혁펜하임 — "꽂히는" 딥러닝** · 한국어 · https://www.youtube.com/playlist?list=PL_iJu012NOxdDZEygsVG4jS8srnSdIgdn
  이번 영역 필수는 아니고 03·04 단계 대비용.

## 미검증 (직접 확인 필요)

- varianceexplained.org 원본 시리즈 http://varianceexplained.org/r/empirical_bayes_baseball/
  (인증서 오류로 R-bloggers 재게시본을 검증본으로 사용)
- David Robinson 전자책 『Introduction to Empirical Bayes』 — 블로그 시리즈로 대체 가능
- 김성범 교수 "핵심 확률통계" 재생목록 — 채널 재생목록 탭에서 찾을 것
