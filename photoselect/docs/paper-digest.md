# photoselect 기반 논문 요약 — 무엇을 · 어떻게 · 어떤 테스트 · 우리가 쓰는 것

`research.md`(2026-08-19 조사, 출처·라이선스 전체)를 **논문 단위로 압축**한 문서다. 각 항목은
"이 모델이 실제로 무엇을 재는가"와 **"거기서 추천 근거(사용자에게 설명할 말)를 뽑을 수 있는가"**를
분리해서 적는다 — 이유 문장(`llm/reasons.py`)이 캡션처럼 나오는 문제의 원인을 논문 수준에서
확인하기 위해서다. 채택 모델은 ★, 비교·참고 논문은 ☆.

핵심 결론을 먼저:

| 단계 | 채택 모델 | 출력 | 근거로 분해되나 |
|---|---|---|---|
| ① 기술 품질 | ★ ARNIQA | 스칼라 1개 | ❌ "초점/흔들림/노출/노이즈" 중 무엇인지 안 나옴 |
| ② 미학 | ★ LAION Aesthetic v2 | 스칼라 1개 | ❌ "구도/색/빛" 중 무엇인지 안 나옴 |
| ③ 미세 비교 | ★ MediaPipe Face Landmarker | 눈 뜸·미소·얼굴 크기 (각각) | ✅ 항목별로 말할 수 있음 |
| ④ 개인화·Top-K | ★ Bradley-Terry + 커버리지 그리디 + MMR | 축별 가중치·선정 슬롯 | ✅ 전부 설명 가능 (코드가 버리고 있을 뿐) |
| 태그·캡션 | ★ 오픈 VLM (gemma3:12b, Ollama) | 고정 축 enum + 한 줄 | ✅ 사진을 실제로 본 유일한 모델 |

IQA/IAA 논문들은 **MOS(평균 평점) 하나를 맞히는 회귀**가 목표이지 원인 분석이 아니다. 벤치마크도
전부 "사람 평점과의 순위 상관(SRCC)"이다. 따라서 ①②에서 "왜 좋은가"를 꺼내려면 모델 바깥에서
분해된 신호(고전 지표·bbox·VLM 속성)를 따로 만들어야 한다.

---

## ① Portrait IQA — 기술 품질 (초점·흔들림·노출·노이즈)

### ★ ARNIQA — "Learning Distortion Manifold for Image Quality Assessment"
Agnolucci et al., Univ. of Florence, **WACV 2024 (oral)**. Apache-2.0, torch.hub.

- **무엇을**: 참조 이미지 없이(NR-IQA) 사진의 기술 품질을 점수 하나로 낸다.
- **어떻게**: ResNet-50 인코더를 **자기지도(SimCLR 계열)** 로 학습 — 원본이 다른 두 사진에
  *같은 종류·강도의 합성 왜곡*(블러·노이즈·압축·색·밝기 등 수십 종의 조합)을 입히면 두 임베딩이
  가까워지도록. 즉 "내용"이 아니라 **"왜곡의 종류와 정도"** 를 임베딩하는 공간(distortion manifold)을
  만든다. 그 위에 **선형 회귀기(Ridge)** 하나를 얹어 데이터셋별 MOS로 맞춘다.
  회귀기는 KADID-10k(합성 왜곡, hub 기본)·KonIQ·SPAQ(실사 스마트폰)·LIVE·CSIQ·FLIVE 등
  8종 제공 — 우리는 실사 회귀기 **spaq** 를 쓴다 (`scripts/arniqa_regressors.py` 비교 결과).
- **테스트**: 표준 IQA 벤치마크에서 사람 MOS와의 SRCC/PLCC. LIVE 0.966, CSIQ 0.962,
  KADID 0.908, **SPAQ 0.905/0.910**. 소량 라벨(few-shot)·데이터셋 간 전이에서 강함이 논문의 주장.
- **우리가 쓰는 것**: `sub_scores.technical_score` → 갤러리 내 백분위 `technical_pct` → prior 점수.
- **근거 추출**: **불가.** 인코더는 왜곡 종류를 구분하는 공간을 배우지만, 배포되는 출력은 회귀기를
  통과한 스칼라 하나다. "흐림인지 노이즈인지"는 회귀 전 임베딩을 직접 해석하지 않는 한 안 나오고,
  그 해석은 논문에 없다. → 초점·노출·노이즈는 고전 지표(라플라시안 분산·히스토그램 클리핑·
  평탄영역 표준편차)로 **따로** 재야 말할 수 있다.

### ☆ 비교 후보 (채택 안 함)

| 논문 | 학회 | 방식 | 벤치마크 (KonIQ SRCC) | 왜 제외 |
|---|---|---|---|---|
| MUSIQ (Ke et al.) | ICCV 2021 | 원본 해상도를 멀티스케일로 넣는 ViT | 0.916 | TensorFlow 스택, 해상도 비례 비용 |
| HyperIQA (Su et al.) | CVPR 2020 | ResNet-50 + 사진마다 회귀 가중치를 생성하는 hypernet | 0.906 | ARNIQA 대비 이점 없음 (MIT, 백업 후보) |
| LIQE (Zhang et al.) | CVPR 2023 | CLIP으로 "품질 텍스트 프롬프트"와의 대응 학습 (장면·왜곡·품질 동시) | 0.919 | 멀티크롭이라 무거움 |
| TOPIQ (Chen et al.) | IEEE TIP 2024 | 의미→왜곡 top-down 어텐션 | 0.926 | pyiqa 배포 = 비상업 |
| Q-Align (Wu et al.) | ICML 2024 | 8B LMM에 "excellent~bad" 등급 텍스트로 학습 | 0.940 | GPU·비상업. 골든셋 오라클 용도만 |
| PIQ23 (DXOMARK) · NTIRE 2024 인물 IQA | CVPR 2023 / 2024 | 인물 특화 데이터셋, 전문가 쌍별 비교 (얼굴 디테일·얼굴 노출·종합) | 챌린지 1위 SRCC **0.554** | 인물 IQA 일반화 미해결 + 비상업. "얼굴 디테일"은 MediaPipe 신호로 대체 |

시사점: **"얼굴 디테일"을 재는 검증된 상업용 모델은 없다.** NTIRE 1위가 0.55면 범용 IQA + 얼굴 랜드마크
조합이 현실적이라는 게 research.md 결론 2.

---

## ② Aesthetic Assessment — 미학 (구도·조명·색감·배치·균형·배경)

### ★ LAION Aesthetic Predictor v2 (논문 아님, Schuhmann 2022)
Apache-2.0. frozen **CLIP ViT-L/14** 임베딩 + 작은 **MLP** 회귀.

- **무엇을**: 사진 하나에 1~10 미학 점수.
- **어떻게**: CLIP 이미지 임베딩(768차원)을 고정하고, MLP 하나를 **AVA 25만 장 + SAC 17.6만 장
  (AI 아트 평점) + LAION-Logos 1.5만 장**의 사람 평점으로 학습. CLIP이 이미 "사진을 언어로
  설명하는" 공간을 갖고 있어서 작은 MLP로도 평점을 어느 정도 맞힌다.
- **테스트**: 공식 벤치마크 없음. Q-Align 논문이 측정한 AVA SRCC **0.721**.
- **우리가 쓰는 것**: `sub_scores.aesthetic_score` → `aesthetic_pct`. CLIP 임베딩은 점수만 뽑고 버린다
  (DB 임베딩은 임베더의 DINOv2).
- **근거 추출**: **불가.** MLP 출력 스칼라 하나. 학습 데이터(AVA)가 DPChallenge **사진 공모전**
  평점이라 "관람자가 wow 하는 정도"이지 구도·색·빛 분해가 아니다. 게다가 SAC(AI 아트) 편향이 있어
  연출된 단일 이미지를 선호한다 — 웨딩 캔디드 수백 장의 미세 순위에는 약할 것으로 예상 (도메인 갭).

### ☆ 비교·참고 논문

| 논문 | 학회 | 방식 | AVA SRCC | 메모 |
|---|---|---|---|---|
| NIMA (Talebi & Milanfar) | IEEE TIP 2018 | 점수가 아니라 **10빈 평점 분포**를 EMD loss로 예측 | 0.612 | 미학 딥러닝의 출발점. 분포를 내므로 "호불호 갈림"은 말할 수 있으나 원인은 못 말함 |
| TANet (He et al.) | IJCAI 2022 | 테마(47종) 인지 브랜치 + 미학 브랜치 | 0.758 | 동반 데이터셋 TAD66K에서 최고 SRCC ~0.51 → **미학 기준이 테마 의존적**이라는 증거 |
| VILA (Ke et al.) | CVPR 2023 | AVA-Captions(사용자 코멘트)로 비전-언어 사전학습 → 제로샷 미학 | 0.774 | 코멘트를 학습해서 **텍스트 근거와 가장 가까운** 접근이지만 TF 스택 |
| Charm (Behrad et al.) | CVPR 2025 | 구도·종횡비·멀티스케일을 보존하는 ViT 토크나이즈, DINOv2-small 22M | (PARA 최고 주장) | **구도 인지 + 소형 + Apache** — ② 교체 1순위 후보 |
| PARA (Yang et al.) | CVPR 2022 | 데이터셋: 31,220장 × 438명, **객관 속성 9종(구도·빛·색·DoF…) + 주관 속성 4종** 라벨 | — | **속성 단위 라벨이 있는 유일한 데이터셋.** "구도가 좋다/빛이 좋다"를 학습하려면 여기서 시작 |
| HumanAesExpert (Kuaishou) | arXiv 2025 | 인물 미학 12차원, InternVL2 1B/8B | — | 인물 미학 파운데이션 모델 첫 시도. 라이선스 불확실 |

시사점: 미학을 **속성으로 분해해서 말하는** 연구는 PARA(속성 라벨)·VILA(코멘트)가 전부고 둘 다 우리가
바로 쓸 형태가 아니다. 현실적 경로는 **VLM에 고정 어휘 `strengths` 축을 묻는 것** — 갤러리당 1회라
원가 원칙에 맞고, 모델이 사진을 실제로 본다.

---

## ③ Fine-grained — 근접 중복 시리즈 내 비교 (표정·시선·눈·자세·순간)

### ☆ "Automatic Triage for a Photo Series" — Chang et al., Princeton + Adobe, SIGGRAPH 2016
③단계가 **왜 별도 단계여야 하는지**의 학술 근거.

- **무엇을**: 비슷한 사진 시리즈(2~8장) 안에서 사람이 고를 사진을 예측.
- **어떻게**: 크라우드소싱으로 **쌍별 선호**(A vs B)를 모아 15,545장/5,953 시리즈 데이터셋 구축.
  VGG **siamese** 네트워크가 두 사진을 받아 선호를 출력.
- **테스트**: 인간 합의 ≥70%인 쌍에서 **73%** 정확도 (기존 절대 미학 점수 베이스라인 57%).
- **핵심 논증**: "절대 미학 점수는 유사 이미지들에 비슷한 점수를 줘서 시리즈 내 순위에 무력하다."
  → 우리가 연사 클러스터 안에서 ①②점수가 아니라 **눈 뜸 → 기술 → 미학** 규칙(`represent.py`)으로
  대표를 고르는 이유.
- 데이터셋 사이트 사망(2026-08). 학습형 siamese 랭커는 골든셋이 쌓인 뒤 과제.

### ☆ Google Top Shot (Pixel 3, 2018 블로그) · Microsoft Best-of-Burst (arXiv 1803.07212, 특허 US 10,671,895)
- Top Shot: 커스텀 MobileNet으로 **블러·눈 뜸·표정** + 모션 saliency + 3A 신호를 **해석 가능한 GAM**으로
  합성해 연사 베스트 선택. → 우리의 신호 목록이 프로덕션에서 검증된 설계임을 확인.
- Microsoft: 0.47MB 경량 판별기, 사용자 선택 대비 top-1 64%·top-3 86%. **특허 있음** — 유사 학습형
  구현 시 검토 필요.

### ★ MediaPipe Face Landmarker (Google, Apache-2.0 코드+모델)
- **무엇을**: 얼굴 478 랜드마크 + **52 blendshape**(`eyeBlinkLeft/Right`, `mouthSmile*` …).
- **어떻게**: 경량 CNN, CPU 실시간. 논문이 아니라 프로덕션 툴킷. 눈 감김 판정의 원조 아이디어는
  EAR(Eye Aspect Ratio, Soukupová & Čech, CVWW 2016) — 우리는 랜드마크 대신 blendshape 값을 쓴다.
- **우리가 쓰는 것**: `eyes_open`(전원 최소값 — 한 명이라도 감으면 낮음), `smile`(평균),
  `max_face_ratio`, `face_count`. `represent.py`가 눈 뜸을 1순위로 대표 선정 → `rank_reason_code`.
- **근거 추출**: **가능.** "같은 순간 연사 중 눈을 뜬 컷", "미소가 가장 큰 컷", "얼굴이 크게 잡힌
  클로즈업"은 전부 측정값에서 직접 나온다. 시선·자세·순간 포착은 안 잰다 —
  6DRepNet(얼굴 방향, MIT)·L2CS-Net(시선, MIT)이 후보, 채택 전.

---

## ④ Personalized Curation — 개인화·다양성·Top-K

### ☆ PIAA 계열 — 개인화 미학은 신호가 약하다
PAM/FPMF (Ren et al., ICCV 2017) → PA-IAA (TIP 2020) → BLG-PIAA (T-Cyb 2020) → TAPP-PIAA (MM 2022).

- **무엇을**: 사용자별 미학 취향을 소량 예시로 학습해 점수를 개인화.
- **어떻게**: 범용 미학 모델 위에 사용자 임베딩/메타러닝/전이 전파로 개인 오프셋 학습.
- **테스트**: FLICKR-AES(40K장/210명), **REAL-CUR**(실제 개인 앨범 14개, 소유자 평가 — 우리와 가장
  유사). 개인 사진 **10장 학습 시 REAL-CUR SRCC 0.44~0.58**.
- **시사점**: 세션 내 10장 안팎의 선택으로 얻는 개인화는 보조 신호다. → plan.md "λ는 증거량에 따라",
  "하드 필터 금지". 우리는 딥 PIAA 대신 **Bradley-Terry(축 33차원 로지스틱 회귀)** 로 축별 가중치를
  배운다 — 해석이 그대로 근거가 된다 ("전신 컷을 선호하셔서").

### ☆ Yeh & Barsky, "Personalized Photograph Ranking and Selection System Considering Positive and Negative User Feedback", ACM TOMM 2014
- 사용자가 고른/뺀 **예시 사진**으로 나머지를 재정렬. 예시 기반이 특징 기반보다 2:1로 선호됨,
  이진 정확도 ~93%. → "담으신 #12와 비슷한 컷"이라는 **임베딩 유사도 근거**의 직접 선행 연구.

### ☆ DPP — Kulesza & Taskar (FnT ML 2012) · Wilhelm et al. "Practical Diversified Recommendations on YouTube with DPPs" (CIKM 2018)
- **무엇을**: 품질은 높고 서로는 다른 부분집합 선택.
- **어떻게**: 품질 벡터 × 유사도 커널의 행렬식이 큰 집합을 고른다. YouTube가 프로덕션에서 A/B로 검증.
- **우리가 쓰는 것**: DPP 대신 같은 목적의 **MMR**(`rerank.mmr_select`, λ_mmr)과
  **장면 커버리지 쿼터**(`coverage_quota` — 서브모듈러 커버리지 그리디의 단순형, Tschiatschek et al.
  NIPS 2014 / Nemhauser 1978 (1−1/e) 보장).
- **근거 추출**: **가능하나 지금은 버린다.** 어떤 사진이 "장면 쿼터로", "점수로", "다양성으로"
  뽑혔는지 `select_with_coverage`가 인덱스만 돌려준다. 이걸 돌려주면 ④의 모든 선정 이유가 설명된다.

### ☆ CUFED / Event-Specific Image Importance — Wang et al., CVPR 2016
- 1,883 앨범·23 이벤트(웨딩 포함)·94,797장에 "이 이벤트에서 중요한 사진" 라벨. 앨범 큐레이션
  벤치마크의 원형이지만 배포 페이지 사망. 골든셋 자체 구축 근거.

---

## 태그·캡션 — 오픈 VLM (논문 아님)

- **무엇을**: 고정 축 5종(`axes.py`: scene·framing·lighting·expression·subjects) enum 태그 + 한 줄 캡션.
- **어떻게**: Ollama structured outputs로 enum 강제 (프로덕션은 vLLM guided decoding).
  `scripts/spike/vlm_compare.py`로 모델·프롬프트 3판 비교 후 확정.
- **근거 추출**: **가장 유리한 자리.** 사진을 실제로 보는 유일한 모델이고 갤러리당 1회다.
  ②의 "구도·색감·배경·시선·순간"을 말하려면 여기에 `strengths` 고정 어휘 축을 추가하는 것이
  ML을 새로 들이지 않는 유일한 경로다.

---

## 상용 서비스가 실제로 쓰는 기준 (퍼널 타당성 대조)

| 서비스 | 공개 기준 | 대응 단계 |
|---|---|---|
| Aftershoot (웨딩 컬링) | 눈 감김·블러(객관) + 조명·배경·구도(주관), 유사 그룹 내 베스트, Duplicates/Blurry/Closed-Eyes 필터 | ①②③ |
| Narrative Select | 초점·눈을 신호등으로, 장면 그룹 내 선명도 랭킹. **"AI는 정보만, 결정은 사진가"** | ①③ |
| Google Top Shot | 눈·표정·블러·모션 GAM | ③ |
| Google Photos 추억 | 메타 필터 → 중복 제거 → 미학 → 테마 | ①②③ |

①③이 상용 컬링의 본체, ②는 보조, **④(부부의 세션 내 선택 + 앨범 커버리지)는 아무도 안 하는 영역**이다.

---

## 이유 문장 문제에 대한 함의

1. ①②는 논문이 원래 원인을 내놓지 않는다. 프롬프트에 "technical = 초점·노출·노이즈 계열 종합 품질,
   aesthetic = 구도·색·빛에 대한 일반 관람자 선호, **둘 다 스칼라**"라고 알려주면 층위가 맞는 말은
   나오지만 **세부 원인은 지어내게 되므로 금지**해야 한다.
2. ③④는 근거가 이미 측정·계산되어 있다. 연사 대표 사유·취향 축·장면 쿼터·다양성·유사 사진 —
   이걸 LLM 재료로 넘기는 것이 첫 번째 할 일이다.
3. 그 다음이 ①②의 분해: 고전 지표(초점·노출·노이즈)와 얼굴 bbox 배치는 모델 없이 되고,
   구도·색감·시선·순간은 VLM `strengths` 축으로 (A단계 변경, `feature-design.md` 계약 갱신).
