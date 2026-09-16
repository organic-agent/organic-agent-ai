# preference — 선택 사진으로 계속 학습하는 선호 가중치 층

부부가 수천 장 중 고른 최종 선택(`photo_selection_items`)으로 "k-웨딩 사진 선호" 가중치를 학습한다. 갤러리가 CLOSED
될 때마다 마감된 갤러리 전부로 다시 학습하고, 홀드아웃(leave-one-gallery-out) 지표가 prior 대비 유의하게 나아진
실행만 `active` 로 표시해 wes 추천에 붙는다. 그 전까지는 학습만 하고 추천은 그대로다.

**산출물은 모델 파일이 아니라 가중치 벡터 한 행**(스칼라 11 + 임베딩 1536 + bias, `preference_models`)이고,
추론은 wes 가 사진마다 내적 한 번이다:

    score = z(prior) + λ(n) · z(w·x + b)        prior = 0.5·tech_pct + 0.5·aes_pct,  λ(n) = n/(n+n₀)

## 무엇을 배우나

| | |
|---|---|
| 입력 x | `technical_pct` · `aesthetic_pct` · `sharpness_pct` · subjects one-hot 4 · 컨셉 그룹 비율 · 타임라인 위치 · 연사 길이 2 (11) + L2 정규화 DINOv3 ⊕ CLIP 에서 갤러리 평균을 뺀 것 (1536) |
| 양성 | 부부의 최종 선택. 같은 연사 클러스터에 둘 이상이면 1/count |
| 음성 | 양성이 없는 연사 클러스터의 대표 1장 |
| 라벨 없음 | 양성의 연사 형제 — 거의 같은 벡터라 음성으로 넣으면 학습이 망가진다 |
| 모델 | 로지스틱 회귀, 블록 분산으로 보정한 L2 두 강도 (scipy L-BFGS). torch 없음 |

## 평가와 게이트

recall@K (K = 3 × 양성 수) 를 **연사 클러스터당 1장으로 dedup 한 순위**에서 잰다 — wes `MmrSelector` 와 같다.
`prior 단독` · `pref 단독` · `융합` 세 점수식을 갤러리 홀드아웃으로 비교하고, 부호 검정 p<0.05 · 악화 비율 ≤ 30% ·
최근 3회 안정성을 모두 만족할 때만 active. 라벨 갤러리가 2개 미만이면 "측정 불가"를 기록한다.

## 실행

```bash
# venv 는 score/.venv 를 같이 쓴다 (numpy · scipy · psycopg · pgvector)
cd preference
../score/.venv/bin/python -m pytest -q                                                  # 13

# DB → 로컬 캐시 (터널: ../organic-agent-server/wes/scripts/db-tunnel.sh, 환경변수 DB_HOST=localhost DB_PORT=15432 …)
../score/.venv/bin/python -m preference export --gallery-id 8 --golden golden/파일명_정리.xlsx
#   --golden 은 xlsx(구분·번호·파일명) 또는 json(파일명 목록) — 데이터셋1 은 golden/dataset1-golden.json

# 요구사항 (1) sanity — in-sample 학습 → 재정렬 → 양성이 후보 범위에 드는가
../score/.venv/bin/python -m preference sanity --gallery-id 8 --local

# 홀드아웃 + 게이트 + 저장 (테이블 없으면 out/preference/models/ 에)
../score/.venv/bin/python -m preference train [--gallery-ids 8,9] [--local]
```

Lambda 는 `controller/handler.handler`, 페이로드 `{"galleryId": N}` (wes 가 CLOSED 전이에서 부른다 — 2단계).
`preference_models` 테이블은 wes Flyway 소유다. 이 repo 는 마이그레이션을 만들지 않는다.

## 구조

패키지는 embedder 와 같은 층으로 나뉘어 있다(#121 과 같은 규칙). 의존은 한 방향이다 — controller → service → repository,
그리고 모두가 domain 을 본다. scipy.optimize 는 infrastructure 에서만, psycopg 는 repository 에서만 쓴다.

```
preference/
├── __main__.py        로컬 CLI export / sanity / train (python -m 규약상 루트)
├── controller/        handler.py(Lambda 진입 — {"galleryId"} → service.job.run_train)
├── service/           job.py(run_train · run_sanity) · features.py(build) · train.py(make_sample · train) · evaluate.py(recall · AUC · LOGO · 게이트)
├── domain/            gallery.py(GalleryData · LabeledGallery) · features.py(FEATURE_SPEC · SCALAR_NAMES · Features) · model.py(PreferenceModel · λ · z · prior) · golden.py(GoldenItem)
├── repository/        connection.py(접속) · store.py(Store Protocol) · db_store.py(Postgres) · local_store.py(npz 캐시) · golden.py(xlsx · json 로더)
├── infrastructure/    solver.py(L-BFGS 로지스틱 회귀 — scipy.optimize 는 여기서만)
└── config/            settings.py(환경변수 → Settings · Knobs)
```

산출물·학습 순서·서비스 연결 설명은 `docs/preference-layer.md`(이 디렉토리). 상세 설계와 실측은 루트 `docs/photoselect/plan-preference-layer.md` · `preference-sanity-2026-09-07.md`(로컬).
