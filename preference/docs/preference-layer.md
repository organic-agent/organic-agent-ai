# 선호 가중치 층 — 무엇을 만들고, 어떻게 배우고, 서비스에서 어떻게 쓰이나

2026-09-07 정리. 요구사항: 부부가 수천 장 중 수십 장을 고른 결정(`photo_selection_items`)이 서비스에 쌓인다.
이 데이터로 "k-웨딩 사진 선호" 가중치 층을 만들고, 갤러리가 마감될 때마다 다시 학습해, 학습에 쓰지 않은 갤러리에서
prior 대비 유의하게 나아진 뒤에만 추천에 붙인다. 코드는 이 디렉토리(`preference/`), 1단계는 #66.

## 1. 최종 산출물 — 가중치 벡터 한 행

ML 모델 파일도, 숫자 하나도 아니다. **숫자 1,548개짜리 행 하나**다.

| 조각 | 길이 | 뜻 |
|---|---|---|
| `w_scalar` | 11 | 기술·미학·선명도 백분위, subjects 원핫 4, 컨셉 그룹 비율, 타임라인 위치, 연사 길이 2 에 곱하는 가중치 |
| `w_emb` | 1,536 | DINOv3(768) ⊕ CLIP(768) 임베딩에 곱하는 가중치 |
| `bias` | 1 | 절편 |
| `lambda` | 1 | 감쇠 λ(n) = n/(n+n₀). n = 학습 갤러리 수 |

실체는 로지스틱 회귀의 가중치다. 은닉층·비선형이 없어 추론에 모델 런타임이나 torch 가 필요 없다.
이 행이 RDS `preference_models` 테이블에 저장되고, wes 추천 잡이 읽어 사진마다 내적 한 번으로 쓴다.

## 2. 학습 — 마감마다 전부 다시 푼다

갤러리 n 이 CLOSED 가 되면 wes 가 `preference` Lambda 를 EVENT 로 부른다. Lambda 안에서 `job.run_train` 이 아래 순서로 돈다.

```
1. 시험     갤러리 1..n-1 로 만든 벡터 V 로 갤러리 n 을 재정렬 → n 의 실제 선택이 후보 범위에 든 비율 (홀드아웃 한 줄)
2. 학습     갤러리 1..n 의 사진·선택을 한 표로 합쳐 벡터를 새로 푼다. 이전 V 는 버린다
3. 판정     쌓인 홀드아웃 줄들로 게이트 판정 → 통과하면 active
4. 저장     preference_models 에 행 INSERT (통과한 실행만 active=true, 이전 active 는 내린다)
```

헷갈리기 쉬운 점 둘:

- **갤러리별 벡터는 없다.** "갤러리 1 벡터, 2 벡터를 합쳐 하나로"가 아니다. 데이터를 합친 뒤 벡터를 한 번 푼다.
  갤러리별로 풀면 각 벡터가 그 부부의 30장을 외우고, 합쳐도 외운 것의 평균이다. 한 표로 풀면 부부마다 다른 취향은
  서로 상쇄되어 벡터에 들어가지 못하고 여러 부부에게 공통인 방향만 남는다. 그래서 결과는 "부부 평균 취향"이다.
- **홀드아웃은 학습하지 않는다. 재기만 한다.** 홀드아웃에서 나오는 것은 벡터가 아니라 점수 한 줄이다. 시험은 언제나
  아직 학습에 넣지 않은 새 갤러리에서, 넣기 직전에 본다. 학습에 들어간 갤러리에서 재면 자기가 배운 것을 다시 푸는
  것이라 점수가 부풀려진다(v1 실측: in-sample 0.86 vs 하나 빼고 0.55).

라벨 규칙 (`train.make_sample`):

| | 정의 | 왜 |
|---|---|---|
| 양성 | 부부의 최종 선택. 같은 연사 클러스터에 둘 이상이면 1/count | 클러스터당 양성 1 |
| 음성 | 양성이 없는 연사 클러스터의 대표 1장 | 클러스터 대표만 — 불균형도 함께 완화 |
| 라벨 없음 | 양성의 연사 형제 | 거의 같은 벡터에 반대 라벨을 주면 없는 경계를 그리게 된다 |

모델은 두 강도 L2 로지스틱(scipy L-BFGS). 갤러리 8 실측에서 학습 0.06초 — 시간은 DB 읽기가 대부분이다.

## 3. 서비스에서 쓰이는 방식 — 사진 한 장과 벡터가 만나는 지점

추천이다 아니다를 가르는 문턱값은 없다. 벡터는 **순위**만 바꾸고, 몇 장을 어디서 고르느냐는 지금 있는 폴더 쿼터와
MMR 이 정한다.

**사진 → 특징 x (1,547개).** 전부 DB 에 이미 있는 것이다. 순서가 계약(`domain/features.py`, `pref-v1`)이라 학습 쪽과 wes 가 똑같이 만든다.

| 자리 | 내용 | 출처 |
|---|---|---|
| 1~3 | 기술·미학·선명도 백분위 ÷ 100 | `photo_analysis.technical_pct` · `aesthetic_pct` · `sub_scores.sharpness_pct` |
| 4~7 | subjects 원핫 (신부·신랑·커플·단체) | `photo_analysis.subjects` |
| 8~11 | 컨셉 그룹 크기 로그 비율 · 타임라인 위치 · 연사 길이 비율 · 연사 여부 | `embed_group_id` · `display_order` · `cluster_id` |
| 12~1547 | L2 정규화한 DINOv3 ⊕ CLIP 에서 **갤러리 평균을 뺀 값** | `embedding` · `clip_embedding` |

갤러리 평균을 빼는 이유: 촬영 장소·작가 스타일은 갤러리마다 다르므로 절대 위치가 아니라 "그 갤러리 안에서 어느 쪽"을 배운다.

**내적 → pref 점수.** "이 사진이 다른 부부들이 고른 방향과 얼마나 닮았나"를 뜻하는 실수.

    pref(p) = w_scalar · x_scalar(p) + w_emb · x_emb(p) + bias

**prior 와 합치기.** 단위가 다르므로 각각 갤러리 안에서 z 정규화한 뒤 λ 로 비중을 정한다.

    score(p) = z(prior(p)) + λ · z(pref(p))        prior = 0.5·technical_pct + 0.5·aesthetic_pct

`active` 행이 없으면 λ 항이 없어 지금 점수와 비트 단위로 같다. 있어도 n 이 작으면 λ 가 작아 prior 에서 조금만 움직인다
(n=1 → 0.17, n=5 → 0.5). 이것이 요구사항 (3)의 소규모 데이터 안전장치다.

**그 뒤는 지금 코드 그대로.** `FolderQuota` 가 폴더별 장수를 정하고, `MmrSelector` 가 연사 클러스터당 최고점 1장만 남기고
MMR 로 고른다. 고른 사진이 `ai_recommendations` 행이 되어 AI 마크로 보인다.

**예.** 기술 40 · 미학 70 · 커플 컷이면 prior 55. 임베딩이 "베일을 든 신부 클로즈업" 방향이고 w_emb 가 그 방향을 양수로 배웠다면
pref 가 큰 양수. z 를 취해 prior +0.3, pref +1.8, λ 0.5 이면 score 1.2 — prior 만으로는 중간이던 사진이 폴더 안 상위로 올라간다.
반대 방향이면 조금 내려간다. 어느 쪽이든 잘라내지 않으므로 후보에서 사라지지는 않는다.

### prior 는 어디서 만들고 어디서 쓰나

| 단계 | 하는 일 | 놓이는 곳 |
|---|---|---|
| score Lambda | ARNIQA(spaq) 기술 점수, CLIP+LAION 미학 점수 — 사진 한 장의 절대 점수 | `photo_analysis.sub_scores` |
| categorize Lambda | 갤러리 안에서 두 점수를 백분위(0~100)로 | `technical_pct` · `aesthetic_pct` |
| wes `AiSelectionJobRunner` | 추천 잡이 돌 때 메모리에서 `0.5·tech + 0.5·aes` → z → deficit·lift 항 → score → 쿼터·MMR. `prior_z` 는 근거 문장 재료로도 | DB 에 없음 |

재료는 배치가 DB 에 적고, prior 자체는 추천 잡이 매번 메모리에서 만든다. 선호 층도 같은 모양이다 — 가중치 행은 DB 에,
pref 점수는 추천 잡이 그 자리에서 내적으로.

### wes 에 붙는 코드 (2단계)

```kotlin
// 1. active 행 — embedding_model · model_version 이 이 갤러리의 photo_analysis 와 같은 것. 없으면 아래 전부 건너뜀
val model = preferenceModelRepository.findActive(embeddingModel, modelVersion) ?: return combinedAsIs

// 2. 특징 조립 + 내적 (갤러리 평균을 먼저 구한다)
val mean = meanOf(rows) { concat(l2(it.embedding), l2(it.clipEmbedding)) }
val pref = DoubleArray(n) { i ->
    val xs = scalars(rows[i], groupSize, clusterSize, order)          // 11, pref-v1 순서
    val xe = concat(l2(rows[i].embedding), l2(rows[i].clipEmbedding)) - mean
    dot(model.wScalar, xs) + dot(model.wEmb, xe) + model.bias
}

// 3. RecommendationScoring.combine 에 항 하나
val score = z(priorRaw) + model.lambda * z(pref) /* + 기존 deficit · lift */
```

## 4. 저장 — RDS `preference_models`

Lambda 가 끝나면 행 하나가 INSERT 된다(`store.DbStore.write_model`). 실행마다 쌓이고 지우지 않으므로 테이블 자체가
"갤러리 수를 늘려가며 홀드아웃 지표를 누적 기록"하는 표다. 스키마는 wes Flyway 소유(제안: 루트 docs 의 plan §6).

| 컬럼 | 내용 |
|---|---|
| `w_scalar` `w_emb`(vector 1536) `bias` `lambda` | 벡터 |
| `embedding_model` `model_version` `feature_spec` | 유효 조건 — embedder·score 모델이 바뀌면 wes 가 안 읽는다 |
| `n_galleries` `n_positives` `train_gallery_ids` | 무엇으로 배웠나 |
| `holdout`(jsonb) | 갤러리별 홀드아웃 점수 · 부호 검정 p · 악화 비율 · 게이트 판정 이유 |
| `active` | 게이트 통과. 부분 유니크로 true 는 한 행 |

테이블이 없으면 `write_model` 이 None 을 돌려주고 CLI 는 `out/preference/models/` 에 JSON 으로 남긴다. S3·모델 파일은 없다.

## 5. 평가와 게이트

recall@K, K = 3 × 양성 수. 순위는 **연사 클러스터당 최고점 1장으로 dedup 한 뒤** 매긴다 — wes `MmrSelector.bestPerCluster`
가 그렇게 하고, 갤러리 8 실측에서 양성 30장의 클러스터가 합쳐 1,109장(중앙값 35)이라 dedup 없이는 형제가 상위를 채운다.

| 지표 | 정의 |
|---|---|
| recall@K strict | 양성 **그 사진**이 클러스터 대표로 상위 K 에 올라온 비율 |
| recall@K cluster | 양성의 **클러스터**가 상위 K 에 든 비율 — 실제 노출과 맞는 쪽. 게이트·sanity 는 이것 |
| AUC | (양성, 음성 대표) 쌍 중 양성 점수가 큰 비율 |

게이트(active 조건) 셋을 모두 만족: 융합 − prior 쌍 차이의 부호 검정 p < 0.05 · 악화 갤러리 비율 ≤ 30% · 최근 3회 추가에서
평균 차이가 0.05 넘게 줄지 않음. 부호 검정은 갤러리 5개부터 가능하고 6개면 5/6 우연(11%)에도 열리므로 갤러리 수를 고정하지
않고 늘려가며 본다.

## 6. 실측 — 갤러리 8 (dataset2 7,189장) + golden 30 (A컷 20 · B컷 10)

| 점수식 | recall@90 strict | recall@90 cluster | AUC |
|---|---|---|---|
| prior 단독 | 0.000 | **0.500** | 0.721 |
| pref 단독 (λ=1) | 0.333 | **1.000** | 0.999 |
| 융합 λ(n=1)=0.167 | 0.033 | 0.567 | 0.795 |

sanity(요구사항 (1)) 통과 — 30장의 클러스터가 전부 상위 90 안(최저 57위). 1,547차원이 30장을 외우므로 당연하고, 성능이 아니다.
prior 의 0.5 가 기준선 — 지금 추천은 부부가 고른 장면의 절반을 후보 90 안에 못 넣는다. strict 가 낮은 것은 형제 컷을 이
특징으로 못 가르기 때문이며 정상. 홀드아웃은 라벨 갤러리가 하나라 "측정 불가". RDS 의 실제 `photo_selection_items` 는 0건.

## 7. 실행 모양과 상태

```
preference/preference/                       (층 구조는 embedder 와 같다 — controller → service → repository, 모두 domain 을 본다)
  controller/handler.py   Lambda EVENT {"galleryId": N} → service.job.run_train   (galleryId 는 로그용, 학습은 CLOSED 전부)
  __main__.py             export / sanity / train  CLI — 같은 job.run_*
  repository/db_store.py  읽기 photos ⋈ photo_analysis · photo_selection_items(읽기 전용) / 쓰기 preference_models
  repository/local_store.py   npz 캐시 (export 로 받아둔 것)
  service/features.py → service/train.py → domain/model.py   특징 → 로지스틱(infrastructure/solver.py) → PreferenceModel(raw · fuse · to_row)
  service/evaluate.py     recall · AUC · leave-one-gallery-out · gate
```

| | 상태 |
|---|---|
| 모듈 · 테스트 11 · 갤러리 8 sanity | 1단계 (#66) |
| ECR `wes-preference` · Lambda · VPC/RDS 접근 (infra) · `deploy-lambda.yml` 모듈 추가 | 2단계 |
| wes: `preference_models` Flyway · CLOSED 전이에서 EVENT · `RecommendationScoring` pref 항 | 2단계 |
| 실제 마감 갤러리가 쌓이며 홀드아웃 표 누적 → n₀ · L2 확정 → 게이트 첫 통과 | 3단계 (데이터) |

트리거 대안: 마감 즉시일 필요는 없으므로 EventBridge 하루 1회 스케줄도 된다. wes 에 트리거 코드가 없어도 되고, 새 마감이 없으면
같은 결과를 다시 쓸 뿐이다.

## 8. 알려진 제약

- **홀드아웃은 지금 못 잰다.** 라벨 갤러리 하나. 갤러리 안에서 나누면 낙관 편향.
- **연사 형제는 라벨 없음.** 양성과 거의 같은 벡터에 음성을 주면 경계가 없는 곳에 경계를 그린다.
- **부부 평균만 배운다.** 부부별 편차는 기존 in-session 항(deficit · lift)이 맡는다. 개선 폭은 완만할 것 → 게이트는 절대값이
  아니라 쌍 차이의 부호로.
- **임베딩·score 버전 결합.** 바뀌면 행이 무효 — 키에 두 버전을 넣어 wes 가 안 읽게.
- **score 의 연사 클러스터가 크다**(중앙값 35, 최대 79). 추천이 그 장면에서 1장만 보여준다는 뜻 — 별도로 볼 문제.
