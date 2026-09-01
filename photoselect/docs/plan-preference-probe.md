# Preference Probe 계획 — 서비스 전체 선택 데이터로 배우는 선호 레이어

작성 2026-08-31. 전제: v2 파이프라인(`docs/plan-v2-slim.md`), v3 폴더별 추천(`docs/plan-v3-folder-compare.md`),
스키마 제안(`docs/schema-proposal.md`), wes 현재 계약(`ai_recommendations`, `photo_selection_items`,
`pair_comparison_events`, `photo_ratings`).

**한 줄 요약.** 서비스 전체에 쌓이는 **최종 선택·추천 반응(·정책 해결 시 별점)** 을 학습 데이터로,
이미 pgvector에 있는 **임베딩(DINOv3 768d) 위에 linear probe(pairwise ranking head)** 를 얹어
"우리 서비스에서 실제로 선택받는 웨딩 컷"의 선호 점수를 만든다. 추론 비용 0(내적 하나),
학습은 CPU 몇 초. 점수는 배치 B rerank의 **가산 항**으로만 들어간다.

## 0. 원칙 — 바뀌지 않는 것

- **하드 필터 금지.** probe 점수는 순위를 밀어올릴 뿐, 어떤 사진도 제외하지 않는다.
- **AI는 초안만.** `photo_selections.status` 불가침, MANUAL 행 읽기 전용 — 기존 계약 그대로.
- **원가 원칙.** 전수 처리는 경량(임베딩은 이미 있음), 학습·추론 모두 CPU. LLM 무관.
- **설명은 규칙 층이 한다.** 768d head는 이유 문장을 못 만든다. 근거는 지금처럼
  수치·집계 재료(백분위·연사·폴더·유형 균형)에서 나오고, probe는 랭킹만 민다.
- **개인화가 아니라 서비스 학습.** 부부 1쌍 단위 학습(BT)은 v2에서 데이터 부족으로 뺐고
  그 판단은 유지한다. 이 계획의 단위는 **글로벌 → 스튜디오**다.

## 1. 학습 신호 — 무엇이 어디에 이미 있나

| 신호 | 테이블 (현황) | 학습 가치 | 상태 |
|---|---|---|---|
| AI 추천 반응 | `ai_recommendations.presented_at / accepted_at / rejected_at / unselected_at` | **최상.** 노출이 기록된 유일한 신호 — 노출 편향 보정이 공짜. accepted vs (presented ∧ ¬accepted)가 그대로 학습 쌍 | **이미 있음** |
| 최종 선택 | `photo_selection_items` (`source` MANUAL/AI) + `photo_selections` 제출 상태 | **상.** 갤러리 안 "선택 vs 미선택". 특히 같은 연사/폴더 안의 형제 컷이 자연스러운 hard negative. MANUAL 행 = AI가 못 맞춘 컷 = 교정 신호 | 있음. 제출 시점 스냅샷 보장 필요 (§2) |
| 쌍 비교 | `pair_comparison_events` (사람의 답) | **상(밀도).** 한 행이 곧 학습 쌍 1개. 단 UI 미구현 — 생기면 그대로 흡수 | 테이블만 있음 |
| 별점 | `photo_ratings` | 중. 선택과 달리 강도(1~5)가 있음 | **정책 블로커** — CLAUDE.md 접근 금지. §7 |
| 보정 요청 | retouch 도메인 (P2) | 셀렉 probe에는 안 씀. 스튜디오별 보정 prior는 **별도 후속 계획** | 범위 외 |

**학습 쌍 만들기 (갤러리 g 안에서만 비교한다 — 갤러리 간 비교는 무의미):**

```
D_g = { (선택 p⁺, 미선택 p⁻) : p⁺, p⁻ 모두 노출됨(presented) 또는 최종 제출 스냅샷 소속 }
가중치: 같은 연사 클러스터 쌍 ×2 (형제 중 고른 것 = 제일 진한 신호)
        MANUAL로 추가된 p⁺ (AI가 못 맞춘 컷) ×2
제외:   노출 기록이 없는 사진과의 쌍 (피드백 루프 방어 — §7)
```

## 2. Phase 0 — 로깅·계약 확인 (지금, 학습 전에)

이 계획의 성패는 모델이 아니라 **데이터가 재구성 가능한 형태로 남느냐**다. wes와 협의할 항목
(스키마는 전부 wes Flyway 소유 — 이 repo는 요청만 한다):

| # | 요구 | 이유 | 예상 변경 |
|---|---|---|---|
| 1 | **제출 시점 선택 스냅샷** — `photo_selection_items`가 제출 후 수정되면 학습 라벨이 오염 | "최종 선택"의 진실 확보 | 제출 시 항목 불변이면 변경 0. 아니면 스냅샷 테이블 |
| 2 | **`ai_recommendations` 행 보존** — v3의 "라운드 덮어쓰기"가 DELETE면 노출 이력 소실 | 노출 로그가 학습 분모 | v3 §2.4대로 round 히스토리 INSERT 유지 확인만 |
| 3 | **임베딩 model_version과 학습 데이터의 결속** — `photo_analysis`는 모델 교체 시 통째 재적재라 과거 임베딩이 사라진다 | probe는 특정 임베더 버전 위의 함수. 라벨(선택)은 영원하지만 입력(임베딩)은 휘발 | 학습 시점에 (photo_id, model_version, embedding, 라벨)을 **S3 parquet로 스냅샷** — DB 변경 0 |
| 4 | 갤러리 → 스튜디오/작가 매핑 조회 가능 | 스튜디오 잔차 학습 (§3) | 기존 도메인에 있으면 변경 0 |

산출물: wes에 보낼 계약 확인 문서 1장 + 학습 스냅샷 잡(`scripts/snapshot_training_data.py`,
갤러리 제출 시 또는 주기 배치로 S3 적재).

## 3. 모델 설계

```
입력   x_p = photo_analysis.embedding (DINOv3 768d, L2 정규화)
       (옵션) CLIP ViT-L/14 concat — 오프라인 평가에서 이득 있을 때만
목적   pairwise logistic (임베딩 위의 Bradley-Terry):
       L = Σ_g Σ_(p⁺,p⁻)∈D_g  ω · log σ( wᵀ(x_p⁺ − x_p⁻) )  −  ½λ‖w‖²
구조   w = w_global + Δw_studio(s)
       · w_global: 전 갤러리 pooled 학습
       · Δw_studio: 스튜디오 s의 쌍 ≥ N_min(초기 500)일 때만, 강한 shrinkage(λ_s ≫ λ)
         — "LoRA처럼"의 이 스택 버전: 공유 head + 스튜디오별 저용량 잔차
출력   probe(p) = wᵀx_p → 갤러리 내 백분위 probe_pct (원점수는 안 쓴다 — tech/aes와 동일 규칙)
학습   scikit-learn / v1 preference.py의 fit_bt 재활용 가능(같은 수식, 입력만 one-hot→임베딩).
       CPU 수 초~수 분. 주 1회 배치면 충분.
```

설계 근거:

- **pairwise(랭킹)이지 분류가 아니다.** "선택률"의 절대값은 갤러리 크기·목표 장수에 좌우된다.
  갤러리 안 상대 비교만이 이식 가능한 신호다.
- **v1 BT 코드가 그대로 자산이다.** `v1/draft/preference.py`의 `fit_bt`(차이 벡터 로지스틱
  MAP)는 입력만 바꾸면 이 수식이다. one-hot 33차원 + 부부 단위(데이터 0)가 문제였지,
  수식이 문제가 아니었다.
- **스튜디오 잔차는 2단계 산출물.** 글로벌 head가 먼저 검증돼야 한다. cold start 스튜디오는
  Δw = 0, 즉 글로벌로 동작.

## 4. Phase 1 — 오프라인 평가 (승격 관문)

- **데이터 분할**: leave-one-gallery-out (갤러리 간 누수 방지). 스튜디오 잔차는
  leave-one-gallery-out을 스튜디오 안에서.
- **지표**:
  - pairwise accuracy — held-out 쌍에서 wᵀ(x⁺−x⁻) > 0 비율 (v1의 LOO conf 교훈 적용:
    in-sample 승률은 낙관 편향)
  - recall@k — "probe 상위 k" ∩ "최종 선택" / min(k, 선택 수), k = 목표 장수
- **베이스라인 2개** (둘 다 이겨야 승격):
  1. 현재 prior `0.5·tech_pct + 0.5·aes_pct`
  2. prior + probe 혼합(§5의 실제 서빙 형태) vs prior 단독 — **혼합 이득**이 진짜 관문
- **최소 데이터 관문**: 제출 완료 갤러리 ≥ 20개, 학습 쌍 ≥ 5,000. 미달이면 학습을 미루고
  스냅샷만 쌓는다.
- 산출물: `scripts/eval_probe.py` + 리포트(`docs/probe-eval-r1.md`). 실패 시 여기서 멈춘다 —
  못 이기는 모델은 서빙에 안 올린다.

## 5. Phase 2 — 서빙 통합

```
배치 B 점수식 (v2/v3 그대로 + 한 항):
  score = 0.5·z(tech_pct) + 0.5·z(aes_pct) + w_bal·balance_z  (기존)
        + w_probe · z(probe_pct)                               (신규, 초기 w_probe = 0.2)
```

- **probe_pct 계산 위치**: 배치 A 끝에 추가(갤러리당 1회, 내적 N번 — 비용 ~0).
  `photo_analysis.sub_scores`에 `probe_score` 키로 저장(스키마 변경 0, jsonb).
- **가중치 아티팩트**: S3 `models/preference-probe/{version}/w.npz`
  (w_global, Δw_studio, embedder model_version, 학습 데이터 범위 메타). Parameter Store에
  현재 버전 포인터. **임베더 버전이 다르면 로드 거부** — 그 갤러리는 w_probe = 0으로 동작.
- **점진 배포**: 갤러리 단위 켜기/끄기 플래그 → 온라인 지표는 `ai_recommendations`의
  accept rate (accepted / presented). probe on/off 갤러리 비교가 곧 A/B.
- **근거 문장**: probe는 이유 재료에 넣지 않는다. 단 코드북의 기준 설명에 한 줄 추가 가능:
  "실제 부부들의 최종 선택 데이터로 학습한 선호 점수를 반영합니다" — 사진별 분해는 안 한다
  (미학 점수와 같은 정직한 한계).

## 6. Phase 3 — 나중 옵션 (지금 안 함)

| 옵션 | 조건 | 메모 |
|---|---|---|
| 별점 편입 | `photo_ratings` 정책 변경 + 동의 확보 | 쌍 가중치(별점 차 = ω)로 흡수 — 구조 변경 없음 |
| 비선형 head (2층 MLP) | probe가 관문을 통과했고 쌍 ≥ 30k에서 linear가 정체 | 여전히 CPU |
| LoRA급 임베더 fine-tune | 쌍 ≥ 수만 + GPU 학습 인프라 정당화 + head 정체 | 산출물은 "우리 임베딩" — 전 파이프라인 재적재 비용 동반. 마지막 카드 |
| 보정 prior | P2 가동 후 | 스튜디오별 요청 분포 → P2 구조화의 사전확률. **별도 계획 문서** |

## 7. 리스크

| 리스크 | 대응 |
|---|---|
| **별점 정책** — CLAUDE.md·plan.md가 `photo_ratings` AI 입력 금지 명시 | 별점은 Phase 3 게이트 뒤. 정책 변경은 wes와 별도 논의(약관·동의 포함). 이 계획은 별점 없이 성립한다 |
| **피드백 루프** — AI가 노출한 것 중에서 선택됨 → 자기 편향 증폭 | 학습 쌍을 노출 기준으로만 구성(§1). 다양성 장치(MMR·폴더 쿼터·cap)가 탐색을 보장 — 이 장치들을 probe가 약화시키지 않게 w_probe 상한 유지 |
| **임베더 버전 결합** — DINOv2→v3처럼 교체되면 probe 무효 | 스냅샷에 model_version 결속(§2-3), 버전 불일치 시 로드 거부 + w_probe=0 폴백, 교체 시 스냅샷 재임베딩 후 재학습 |
| **부부 취향 ≠ 작가 스타일** — 라벨은 부부가 만들고, 고객은 작가 | 글로벌+스튜디오 단위라 "이 스튜디오 고객들이 고르는 컷" = 작가에게도 유효한 신호. 부부 개인화는 범위 외(집계 규칙이 담당) |
| **데이터 희소 / cold start** | 최소 데이터 관문(§4) 전엔 학습 안 함. 스튜디오 잔차는 N_min 미달 시 0 |
| **분포 이동** — 시즌·스튜디오 구성 변화 | 주기 재학습 + 학습 데이터 범위 메타로 롤백 가능. 평가 리포트 누적 |

## 8. 이슈 매핑 (git-workflow 규칙)

| 순서 | 이슈 | 산출물 |
|---|---|---|
| 1 | `[photoselect] docs: preference probe 데이터 계약 확인 요청` | wes 협의 문서, §2 체크리스트 결론 |
| 2 | `[photoselect] feat: 학습 스냅샷 잡 — 선택·노출·임베딩 S3 적재` | `scripts/snapshot_training_data.py`, parquet 스키마 |
| 3 | `[photoselect] feat: preference probe 학습·오프라인 평가` | `probe/train.py`, `scripts/eval_probe.py`, `docs/probe-eval-r1.md` (관문) |
| 4 | `[photoselect] feat: 배치 A probe 점수 + 배치 B 가산 항 편입` | `sub_scores.probe_score`, `w_probe` 다이얼, on/off 플래그 |
| 5 | `[photoselect] feat: 스튜디오 잔차 + 온라인 지표 대시보드` | Δw_studio, accept rate 리포트 |

1·2는 지금 바로 가능하고(데이터는 지금부터 쌓인다), 3은 §4 최소 데이터 관문이 차면 시작한다.
