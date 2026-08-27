# AI 셀렉터 — 기능 목차와 진행 상태

살아 있는 체크리스트. 항목 하나 끝낼 때마다 여기를 갱신한다. 상세 설계는 `photoselect/docs/plan.md`,
코드는 `photoselect/`, 실측 기록은 `photoselect/docs/spike-report.md`.

범례 · `[x]` 구현·실측 완료 · `[~]` 코드는 있으나 조건부/검증 부족 · `[ ]` 미구현

최종 갱신: 2026-08-25 (1·2·3·5번 완료)

---

## 1. 사용자 데이터 없을 때 사진 추천 (사진학 기반)

### 1.1 사진들에서 데이터 뽑아내기 — `photoselect/analyze/`

- [x] 얼굴·눈 뜸·미소 (MediaPipe) — 0.1s/장
- [x] 기술 품질 (ARNIQA) — 회귀기 **`spaq`** 확정 (08-25 비교: kadid10k는 실사 회귀기와 상관 0.16~0.34로 다른 것을 잼. spaq는 흐림 반응 23.7배·미학과 상관 0.10). `scripts/arniqa_regressors.py`
- [x] 미학 (LAION Aesthetic v2)
- [~] 임베딩 — 지금 CLIP 768d. **최종은 DINOv2(`photos.embedding`)로 통일, 전환 시 클러스터 임계값 재측정**
- [~] VLM 고정 축 태그 5종 — gemma3:12b 4bit, 6s/장. 1차 프롬프트 **확정**(scene 74 / lighting 84 / expression 90 / framing 94 / subjects 98). **프롬프트 튜닝 종료**: 1차 재실행 250칸 동일(결정적), v2·v3는 전 축 하락 → 남은 오차는 모델 크기(31B 비교 미실행)·데이터(예식 장면 5개 값 미검증) 문제. `scripts/spike/vlm_compare.py`
- [x] VLM 캡션
- [x] 갤러리 내 백분위 (technical_pct · aesthetic_pct)
- [x] 근접 중복 클러스터 — 임계값 0.96 실측 (CLIP 기준)
- [x] 클러스터 대표 선정 — 눈뜸 → 기술 → 미학
- [x] 멱등 재실행 · 실패 격리 · 단계별 시간 측정
- [ ] DB 적재 `photo_analysis` — `LocalStore`만. 스키마는 `photoselect/docs/schema-proposal.md`
- [ ] GPU EC2 · vLLM 배치 — 1,000장 ≤ 10분 실측
- [ ] 갤러리(2) 재분석 v0.3 — 프롬프트 복원 후 캡션 갱신용 (23분, 급하지 않음)
- [ ] A 러너 단위 테스트

### 1.2 해당 값 기반으로 사진 뽑기 — `photoselect/draft/`

- [x] 사진학 prior (기술 + 미학 백분위)
- [~] 장면 커버리지 + 상한 40% — 장면 2~3종 갤러리에서 무력. **본식 갤러리 확보 후 재검토**
- [x] 클러스터당 1장
- [~] MMR 다양성 — `λ_mmr` 0.7 **가정값**. 사람 눈으로 정할 것
- [x] 목표 장수 · 완료 조건 — `min(K, 목표 − 담긴 수)`, 도달 시 done
- [x] 실측 — 200장 → 30장 (R1, 사진학만)

### 1.3 VLM으로 추천 이유 달기

- [x] 템플릿 이유 — 캡션 + 대표 선정 근거 + 품질
- [x] 정밀도 낮은 태그 값은 이유에 안 쓰기 — `AXIS_PRECISION`(08-25 채점) ≥ 0.85인 축만 취향 근거로. 지금은 lighting·scene 제외, scene은 캡션으로 대체
- [x] LLM 일괄 문장화 (Bedrock Haiku 4.5, `llm/reasons.py`) — `draft --llm`. 실호출 검증 08-25: 30장 1회 ≈ 1.3k in / 0.4k out 토큰, 40~53자, 취향 근거 앞세움. 실패 시 템플릿 폴백
- [ ] 캡션 프롬프트 어미 통일

---

## 2. 사용자 데이터로 사진 추천 (취향 반영)

### 2.0 데이터 수집 — 온보딩 방식 미정

- [x] 별점 → 쌍 변환 (같은 장면 · 차 ≥ 2 · 가중 0.5)
- [x] 선택사진 → 쌍 변환 (같은 장면 · 품질 근접 · 가중 0.3)
- [x] 온보딩 쌍 비교 생성 — 같은 장면 · 한 축만 다름 · 품질 근접 · 다른 클러스터. 첫 라운드 12쌍
- [~] 수집 UI — 로컬 `review` HTML → `evidence.json`. **프론트 연동 미정**
- [ ] DB 읽기 — `photo_ratings` · `photo_selection_items` · `pair_comparison_events`
- [ ] 온보딩 방식 결정 — 실제 고객 데이터 vs 예시 데이터

### 2.1 취향 + 중복 지양 + 사진학으로 필터링 — `photoselect/draft/`

- [x] Bradley-Terry 취향 학습 (33차원, L2)
- [x] 축별 신뢰도 conf — leave-one-out
- [~] λ = f(증거 가중합) — `k=30` **가정값**. 실제 쌍 데이터로 재추정
- [x] prior · pref 결합 (z-표준화)
- [x] 후보 집합 — 담은 사진 + **이미 제시한 사진** 제외
- [x] 중복 지양 · 커버리지 · MMR (1.2와 동일 코드)
- [x] 실측 — 4라운드, 65장 보고 30장 완료, 수락률 30→43→83% (평가자 1명)
- [ ] 다수 평가자 검증 — 10명 × 120쌍 (로드맵 #1). 도구는 준비됨, 파일럿 2~3명부터

### 2.2 VLM으로 이유 달기

- [x] 취향 근거 문장 ("선호하신 전신·두 분 스타일") — conf ≥ 0.4인 축만
- [x] 자연어 피드백 → 축 가중치 번역 (`llm/feedback.py`) — `evidence.json`의 `feedback` 배열. 실호출 검증: "클로즈업 너무 많아요" → framing closeup −0.5 / full +0.5, 무관한 문장 → []
- [ ] 👍/👎 반응 반영 — 스키마 컬럼만 있음

---

## 3. 공통 — 인프라 · 계약

- [x] 모듈 구조 (embedder 패턴) — PR #11
- [x] 저장소 인터페이스 `Store` — Local
- [ ] `DbStore` — wes V22 머지 후
- [x] 스키마 제안 문서 — `photoselect/docs/schema-proposal.md`
- [ ] wes 스키마 적용 (Flyway) — AI 셀렉 완성 시 일괄
- [ ] Lambda handler 동작 · EC2 systemd 루프 · Terraform
- [x] 테스트 — B · 클러스터 8개
- [ ] `plan.md` §5를 schema-proposal로 대체, "MANUAL 승격" 문구 삭제
- [ ] `tech-stack.md` §5: `AnthropicBedrockMantle` → `AnthropicBedrock` (ap-northeast-2에 Mantle 엔드포인트 없음, 08-25 확인)

---

## 진행 순서 (권장)

DB·인프라 없이 할 수 있는 것부터. 각 항목은 반나절 단위.

1. ~~ARNIQA 회귀기 비교~~ ✅ spaq
2. ~~VLM 프롬프트 2차 채점~~ ✅ 결론: 프롬프트로 개선 불가, 1차 확정. 다음 손잡이는 12B vs 31B
3. ~~정밀도 기준 이유 문장~~ ✅
4. **온보딩 파일럿 2~3명** (2.1) — review 페이지로 한 바퀴, λ_k 첫 재추정
5. ~~LLM 이유 문장~~ ✅ 실호출 검증
6. **DbStore** (3) — 스키마 적용과 함께
7. **EC2·Lambda·Terraform** (3)
