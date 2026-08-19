# photoselect 단계별 연구 조사 — 논문 출처·벤치마크·라이선스

`plan.md`의 4단계 퍼널 각각에 대해 논문 출처(학회·연도), 벤치마크 수치, 코드·가중치 공개
여부, 라이선스를 웹에서 검증한 조사 결과. **2026-08-19 기준.** 확인하지 못한 항목은
`(미검증)`으로 표기 — 채택 전 반드시 재확인.

---

## 요약 — 실무 결론 5가지

1. **정확도가 아니라 라이선스가 진짜 제약이다.** pyiqa 라이브러리(PolyForm NC), PIQ23,
   SER-FIQ/CR-FIQA, Q-Align(S-Lab), QualiCLIP, InsightFace 사전학습 모델이 전부
   **비상업 라이선스**다. 상업 서비스에 쓸 수 있는 깨끗한 경로는 원본 repo에서 직접 쓰는
   ARNIQA·MUSIQ·VILA·TANet·Charm(Apache-2.0), HyperIQA·LIQE·6DRepNet·L2CS-Net(MIT),
   MediaPipe(코드+모델 Apache-2.0), HSEmotion(Apache-2.0) 정도다.
2. **인물 특화 IQA는 아직 일반화가 약하다.** NTIRE 2024 인물 IQA 챌린지 1위가 미공개 테스트
   셋에서 SRCC 0.554에 그쳤다 (범용 IQA는 KonIQ에서 0.92+). 인물 특화 모델보다
   **범용 NR-IQA 점수 + 얼굴 영역 신호(랜드마크·얼굴 크롭 선명도)** 조합이 현재로선 더 믿을 만하다.
3. **③단계의 학계 합의는 "절대 점수가 아니라 쌍별(pairwise) 비교"다.** Photo Triage
   (SIGGRAPH 2016) 논문이 명시적으로 "절대 미학 점수는 유사 사진들에 비슷한 점수를 줘서
   시리즈 내 순위에 무력하다"고 논증했고, 이후 연구가 전부 siamese/ranking 구조다. Google
   Top Shot(프로덕션)의 신호 설계(눈 뜸 + 표정 + 블러 + 모션)가 우리 계획의 신호 목록을
   그대로 검증해 준다.
4. **개인화(PIAA)는 신호가 약하다는 게 정량적으로 확인됐다.** 개인 사진 10장 학습 기준
   실제 개인 앨범(REAL-CUR)에서 SRCC 0.44~0.58. 개인화는 보조 신호로만 쓰고 하드 필터
   금지 원칙을 유지하는 게 문헌과 일치한다. Top-K 다양성은 DPP(YouTube 프로덕션 적용,
   CIKM 2018)가 표준 템플릿.
5. **웨딩 특화 학술 논문은 전무하다.** 상용 컬링 툴(Aftershoot, FilterPixel, Imagen 등)만
   존재하고 방법론 공개는 없다. "실제 웨딩 갤러리 + 세션 내 선택만을 개인화 신호로 + Top-K"
   조합의 논문 니치는 실제로 비어 있다 (§6).

---

## ① Portrait IQA — 기술 품질

### 비교표

| 모델 | 학회/연도 | KonIQ-10k SRCC/PLCC | 백본 | 가중치·라이선스 | CPU Lambda |
|---|---|---|---|---|---|
| MUSIQ | ICCV 2021 | 0.916 / 0.928 | 멀티스케일 ViT (원본 해상도) | TF ckpt, **Apache-2.0** | 중 (해상도 비례) |
| TOPIQ | IEEE TIP 2024 | 0.926 / 0.939 | ResNet-50 CFANet | pyiqa 경유 → **비상업** | 좋음 (라이선스 문제) |
| LIQE | CVPR 2023 | 0.919 / 0.908 | CLIP ViT-B/32 | **MIT** | 중 (멀티크롭) |
| ARNIQA | WACV 2024 oral | — (SPAQ 0.905/0.910) | ResNet-50 + 선형 회귀 | **Apache-2.0**, torch.hub | **최적 조합** |
| HyperIQA | CVPR 2020 | 0.906 / 0.917 | ResNet-50 + hypernet | **MIT** | 좋음 |
| Q-Align | ICML 2024 | 0.940 / 0.941 | mPLUG-Owl2 ~8B LMM | S-Lab **비상업** + LLaMA-2 | 불가 (GPU) |
| LAR-IQA | ECCVW 2024 | (UHD-IQA 챌린지) | MobileNet급 | 미검증 | 탁월 (설계상) |

### 상세

- **MUSIQ** — "MUSIQ: Multi-scale Image Quality Transformer", Ke et al., Google Research,
  ICCV 2021. [arXiv:2108.05997](https://arxiv.org/abs/2108.05997) ·
  [코드](https://github.com/google-research/google-research/tree/master/musiq) (TensorFlow,
  Apache-2.0, KonIQ/SPAQ/PaQ-2-PiQ/AVA 체크포인트 공개, TF Hub 제공).
- **TOPIQ** — "TOPIQ: A Top-Down Approach From Semantics to Distortions for IQA",
  Chen et al., NTU S-Lab + SenseTime, IEEE TIP 2024.
  [arXiv:2308.03060](https://arxiv.org/abs/2308.03060). CLIVE 0.870/0.884, SPAQ 0.921/0.924.
  공식 배포가 pyiqa 안에 있어서 **PolyForm NC 라이선스에 묶인다** — 성능 대비 아까운 케이스.
  pyiqa에 GFIQA 학습 인물 변형 `topiq_nr-face` 존재 (미검증).
- **LIQE** — "Blind IQA via Vision-Language Correspondence", Zhang et al., SJTU+CityU,
  CVPR 2023. [arXiv:2303.14968](https://arxiv.org/abs/2303.14968) ·
  [코드](https://github.com/zwx8981/LIQE) (MIT, 가중치 공개). CLIVE 0.904/0.910.
- **ARNIQA** — "Learning Distortion Manifold for IQA", Agnolucci et al., Univ. of Florence,
  WACV 2024 oral. [arXiv:2310.14918](https://arxiv.org/abs/2310.14918) ·
  [코드](https://github.com/miccunifi/ARNIQA) (Apache-2.0, torch.hub 한 줄 로드,
  torchmetrics에도 수록). 자기지도 학습이라 소량 라벨로 일반화. **논문 본표에 KonIQ 수치는
  없음** (LIVE 0.966, CSIQ 0.962, KADID 0.908, SPAQ 0.905) — pyiqa의 koniq 회귀 변형은 별도.
- **HyperIQA** — "Blindly Assess Image Quality in the Wild…", Su et al., CVPR 2020.
  [코드](https://github.com/SSL92/hyperIQA) (MIT). PIQ23의 공식 베이스라인(SEM-HyperIQA)이
  이 모델 기반.
- **Q-Align** — Wu et al., ICML 2024,
  [proceedings](https://proceedings.mlr.press/v235/wu24ah.html) ·
  [arXiv:2312.17090](https://arxiv.org/abs/2312.17090). 정확도는 최고지만 8B LMM + 비상업
  라이선스. 배치용 불가 — 스파이크에서 **골든셋 라벨 보조용 오라클**로만 검토.
- **LAR-IQA** — ECCV 2024 Workshops. MobileNetV3급 듀얼 브랜치, "기존 최속 모델 대비
  ~5.7×" 주장 — CPU Lambda 관점에서 가장 직접적으로 관련.
  [arXiv:2408.17057](https://arxiv.org/abs/2408.17057). repo·라이선스 미검증.
- **인물 특화**: PIQ23 (DXOMARK, CVPR 2023, [arXiv:2304.05772](https://arxiv.org/abs/2304.05772))
  — 5,116장, 스마트폰 100종, 전문가 쌍별 비교(얼굴 디테일·얼굴 노출·종합), **비상업 전용**.
  후속 FHIQA (Pattern Recognition Letters 2025, [arXiv:2402.09178](https://arxiv.org/abs/2402.09178)).
  **NTIRE 2024 인물 IQA 챌린지** ([arXiv:2404.11159](https://arxiv.org/abs/2404.11159)):
  35팀 중 1위 SRCC 0.554 / PLCC 0.597 — 인물 IQA 일반화는 미해결 상태라는 핵심 근거.
- **얼굴 FIQA (SER-FIQ CVPR 2020, CR-FIQA CVPR 2023)** — 얼굴 "인식 가능성" 측정이라
  사진 품질과는 상관하되 동일 개념이 아니고, 둘 다 CC BY-NC(비상업). 채택 배제.
- **pyiqa (IQA-PyTorch)** — 위 모델 대부분 수록, 스파이크 비교에 편리. 단
  **PolyForm Noncommercial 1.0.0 + S-Lab 라이선스** — 상업 서비스 코드에 포함 불가.
  평가용 내부 사용도 그레이존이므로 스파이크는 원본 repo 기준으로 진행 권장.
- **2024–25 신규**: QualiCLIP ([arXiv:2403.11176](https://arxiv.org/abs/2403.11176), CC BY-NC),
  Compare2Score (NeurIPS 2024, LMM), DeQA-Score (CVPR 2025, LMM), DSL-FIQA
  ([arXiv:2406.09622](https://arxiv.org/abs/2406.09622), 범용 얼굴 IQA + GFIQA-20k, 미검증).

**①단계 스파이크 후보 확정**: ARNIQA(1순위, Apache+CPU 최적) vs HyperIQA(MIT 베이스라인)
vs LIQE(MIT, 무거움) + LAR-IQA(라이선스 확인 후). 얼굴 디테일은 ③의 MediaPipe 신호로 대체.

## ② Aesthetic Assessment — 미학

### 비교표 (AVA 기준)

| 모델 | 학회/연도 | AVA SRCC/PLCC | 백본/크기 | 라이선스 | CPU Lambda |
|---|---|---|---|---|---|
| NIMA | IEEE TIP 2018 | 0.612 / 0.636 (이진 81.5%) | MobileNet~Inception | 재구현 Apache/MIT | 탁월 |
| TANet | IJCAI 2022 | 0.758 / 0.765 | ResNet18+MobileNetV2 | **Apache-2.0** | 좋음 |
| VILA-R | CVPR 2023 | 0.774 / 0.774 | CoCa-Base (ViT-B/16), TF | **Apache-2.0** (TF Hub) | 중 |
| LAION Aesthetic v2 | (비논문) | 0.721 / 0.723 (Q-Align 측정) | CLIP ViT-L/14 + MLP | **Apache-2.0** | 중 (CLIP 재사용 시 ~무료) |
| UniQA | arXiv 2024 | 0.776 / 0.776 | CLIP-B/16 | 미검증 | 좋음 |
| Q-Align | ICML 2024 | **0.822 / 0.817** | ~8B LMM | **비상업** | 불가 |
| Charm | CVPR 2025 | (PARA 최고 성능 주장, 수치 미검증) | DINOv2-small ~22M | **Apache-2.0** | 탁월 |

### 상세

- **NIMA** — Talebi & Milanfar, Google, IEEE TIP 2018.
  [arXiv:1709.05424](https://arxiv.org/abs/1709.05424). 평점 분포(10빈, EMD loss) 예측.
  공식 가중치 미공개 — 재구현:
  [idealo](https://github.com/idealo/image-quality-assessment) (Apache-2.0, MobileNet 가중치
  포함, SRCC 0.609로 논문보다 약함), [titu1994](https://github.com/titu1994/neural-image-assessment) (MIT).
- **TANet** — "Rethinking Image Aesthetics Assessment", He et al., BUPT, IJCAI 2022.
  [proceedings](https://www.ijcai.org/proceedings/2022/132) ·
  [코드](https://github.com/woshidandan/TANet-image-aesthetics-and-quality-assessment)
  (Apache-2.0, 가중치 공개). TAD66K(47테마, 테마별 기준) 데이터셋 동반 — AVA보다 훨씬
  어려움(최고 SRCC ~0.51): **미학 기준이 테마 의존적**이라는 직접 증거.
- **VILA** — "Learning Image Aesthetics from User Comments with Vision-Language
  Pretraining", Ke et al., Google, CVPR 2023.
  [arXiv:2303.14302](https://arxiv.org/abs/2303.14302) ·
  [코드](https://github.com/google-research/google-research/tree/master/vila) (Apache-2.0,
  GCS 체크포인트 + TF Hub). AVA-Captions(사용자 코멘트) 사전학습 → 제로샷 SRCC 0.657.
  TensorFlow라 배포 스택이 무거운 게 단점.
- **LAION Aesthetic Predictor v2** — 논문 아님.
  [블로그](https://laion.ai/blog/laion-aesthetics/) ·
  [repo](https://github.com/christophschuhmann/improved-aesthetic-predictor) (Apache-2.0,
  가중치 in-repo). frozen CLIP ViT-L/14 + MLP, 학습 데이터 SAC 176K + LAION-Logos 15K +
  AVA 250K. **임베딩을 이미 뽑는다면 미학 점수가 사실상 공짜** — embedder 확장과 궁합 최고.
  단 SAC(AI 아트 평점) 편향으로 실사진 순위 검증은 부족.
- **Charm** — "The Missing Piece in ViT fine-tuning for IAA", Behrad et al., CVPR 2025.
  [arXiv:2504.02522](https://arxiv.org/abs/2504.02522) ·
  [코드](https://github.com/FBehrad/Charm) (Apache-2.0, HF 가중치). 구도·종횡비·멀티스케일
  보존 토크나이즈, DINOv2-small ~22M — **구도 인지 + 소형 + 상업 가능**으로 직접 관련.
- **UniQA** — IQA+IAA 통합, CLIP-B/16. [arXiv:2406.01069](https://arxiv.org/abs/2406.01069).
  NeurIPS 2024 주장 미검증.
- **인물 미학**: **HumanAesExpert** ([arXiv:2503.23907](https://arxiv.org/abs/2503.23907),
  Kuaishou) — 최초의 인물 이미지 미학(HIAA) 파운데이션 모델, HumanBeauty 108K(12차원 기준),
  1B/8B InternVL2, HF 가중치. README는 MIT지만 LICENSE 파일 부재 (미확실). 1B는 CPU 경계선.
  논문 스스로 "HIAA는 거의 탐구되지 않았다"고 명시.
- **도메인 갭 (핵심 유의)**: AVA는 DPChallenge 사진 공모전 데이터(255,530장, 평균 ~210표)
  — 의도적으로 연출된 단일 이미지 + "wow" 위주 투표. 이벤트·캔디드 사진 분포가 아니고
  근접 중복 구조도 없다. AVA 학습 모델은 극적인 풍경·HDR을 과대평가하고 수백 장의 비슷한
  인물 캔디드 구분에는 약할 것으로 예상 → **골든셋 상관 검증이 필수**인 이유.

**②단계 스파이크 후보 확정**: LAION v2(CLIP 재사용, 최저 원가) + Charm(구도 인지 소형) +
TANet(중형 베이스라인). VILA는 TF 스택 감수 시. 정확도 상한 참조용 오라클은 Q-Align.

## ③ Fine-grained — 근접 중복 시리즈 내 비교

### 앵커 논문과 프로덕션 증거

- **"Automatic Triage for a Photo Series"** — Chang et al., Princeton + Adobe,
  **SIGGRAPH (ACM ToG) 2016**. [DOI](https://dl.acm.org/doi/10.1145/2897824.2925908) ·
  [프리프린트](https://gfx.cs.princeton.edu/pubs/Chang_2016_ATF/chang2016-triage-preprint.pdf).
  15,545장 / 5,953 시리즈(2~8장), 크라우드소싱 쌍별 선호가 정답. VGG siamese가 인간 합의
  ≥70% 쌍에서 **73%** 정확도 (최고 기존 베이스라인 57%). **"절대 미학 점수는 유사 이미지에
  비슷한 점수를 줘서 시리즈 내 순위에 실패한다"를 명시 논증** — ③단계 존재 이유의 학술 근거.
  ⚠ 데이터셋 사이트(phototriage.cs.princeton.edu)는 2026-08 현재 DNS 사망 — 필요 시 저자
  연락 또는 Internet Archive.
- **Microsoft Best-of-Burst** — "Real-time Burst Photo Selection Using a Light-Head
  Adversarial Network" ([arXiv:1803.07212](https://arxiv.org/abs/1803.07212)) + 미국 특허
  **US 10,671,895** (Microsoft, 2020 등록). 사용자 선택 대비 top-1 ~64%, top-3 ~86%,
  0.47MB 모델. **특허 존재 — 유사 구현 시 특허 검토 필요.**
- **Google Top Shot** (Pixel 3) —
  [Google Research 블로그 2018](https://research.google/blog/top-shot-on-pixel-3/).
  커스텀 MobileNet으로 블러(얕은 층)·**눈 뜸·표정**(깊은 층) + 모션 saliency + 3A 신호를
  해석 가능한 GAM으로 합성. **우리가 계획한 신호 목록(눈·표정·블러)이 프로덕션에서 검증된
  설계임을 확인.**
- 후속 학술: Multi-view Graph SPS (ICME 2022, [arXiv:2203.09736](https://arxiv.org/abs/2203.09736),
  수치 미검증), 쌍별 랭킹 원조 Kong et al. ECCV 2016 (AADB 데이터셋 동반,
  [arXiv:1606.01621](https://arxiv.org/abs/1606.01621)). 학술 밀도는 낮고 상용 제품
  (Aftershoot, FilterPixel DeepCull — 웨딩 시퀀스 인지 주장)으로 이동한 영역.

### 얼굴 신호 도구 (채택안)

| 신호 | 도구 | 근거 | 라이선스 |
|---|---|---|---|
| 눈 뜸/감김 | **MediaPipe Face Landmarker** — 478 랜드마크 + 52 blendshape (`eyeBlinkLeft/Right`) | EAR 원조: Soukupová & Čech, CVWW 2016 | **Apache-2.0 (코드+모델)** |
| 미소/표정(경량) | MediaPipe `mouthSmile*` blendshape | 상동 | Apache-2.0 |
| 표정(정밀) | **HSEmotion** (EfficientNet-B0/B2, ONNX, pip) — AffectNet-7 66.3% | SoftwareX 2022 | **Apache-2.0** (상업 명시 허용) |
| 얼굴 방향 | **6DRepNet** — AFLW2000 MAE 3.47° | ICIP 2022 / TIP 2024 | **MIT** (pip sixdrepnet) |
| 시선 | **L2CS-Net** — MPIIGaze 3.92° | arXiv 2022 | **MIT** |

주의사항:
- **dlib 68 랜드마크 금지** — 학습 데이터 iBUG 300-W가 상업 사용 배제 (EAR 튜토리얼 대부분이
  이걸 씀). MediaPipe 랜드마크로 EAR 계산.
- **InsightFace 배제** — 코드는 MIT지만 사전학습 모델(buffalo_l 등) 전부 비상업. 게다가
  눈 상태·표정 속성 자체가 없음(성별·나이만).
- FER 정밀도 상한: AffectNet 7클래스 63~67%가 분야 한계. 웨딩 셀렉에는 미소/valence 정도만
  필요해서 MediaPipe blendshape로 충분할 가능성 — 스파이크에서 판단.
- DDAMFN (Electronics 2023, RAF-DB 91.4%)이 더 정확하지만 라이선스 미확인.

**③단계 채택안 확정**: MediaPipe 단일 스택(검출+랜드마크+눈+미소) 기본, 부족하면
HSEmotion(표정)·6DRepNet(방향) 추가. 학습형 siamese 랭커는 골든셋이 쌓인 뒤의 후속 연구 주제.

## ④ Personalized Curation — 개인화 Top-K

### PIAA (개인화 미학) — 신호 강도의 현실

FLICKR-AES(40K장/210명) · REAL-CUR(실제 개인 앨범 14개, 소유자 평가 — 우리 설정과 가장
유사) 벤치마크, SRCC (개인 사진 10장 / 100장 학습):

| 방법 | 학회/연도 | FLICKR-AES | REAL-CUR |
|---|---|---|---|
| PAM/FPMF (원조, Ren et al.) | ICCV 2017 | 0.513 / 0.524 | — |
| PA-IAA | IEEE TIP 2020 | 0.543 / 0.639 | 0.443 / 0.562 |
| BLG-PIAA (메타러닝) | IEEE T-Cyb 2020 | 0.561 / 0.669 | 0.448 / 0.578 |
| TAPP-PIAA (전이 전파) | ACM MM 2022 | 0.591 / 0.685 | ~0.580 (100) |
| Task-vector PIAA | arXiv 2024 (venue 미검증) | 0.668 / 0.748 | 0.577 / 0.621 |

- PARA 데이터셋 — "Personalized Image Aesthetics Assessment with Rich Attributes",
  Yang et al., OPPO+Xidian, **CVPR 2022**. [arXiv:2203.16754](https://arxiv.org/abs/2203.16754).
  31,220장 × 438명(장당 ~25명), 객관 속성 9종(구도·빛·색·DoF…) + 주관 속성 4종 + 평가자
  성격 프로필. 속성 단위 감독의 최고 데이터셋. 학술용 배포 (정확한 라이선스 문구 미검증).
  현재 PARA 개인화 SRCC 상한 ~0.6.
- **시사점**: 세션 내 10장 안팎의 선택으로 얻는 개인화 신호는 SRCC 0.44~0.58 수준(실제 앨범
  기준). **개인화는 재정렬 보조 신호로만, 초기값은 ①~③ 점수** — plan.md의 콜드스타트
  설계와 일치.

### Top-K 구성 (다양성·커버리지)

- **DPP 정본**: Kulesza & Taskar, "Determinantal Point Processes for Machine Learning",
  Foundations & Trends in ML 2012. [arXiv:1207.6083](https://arxiv.org/abs/1207.6083).
  k-DPP = 고정 크기 부분집합.
- **프로덕션 적용 템플릿**: Wilhelm et al., "Practical Diversified Recommendations on
  YouTube with DPPs", **CIKM 2018**. [PDF](https://jgillenw.com/cikm2018.pdf) — 품질 점수
  벡터 × 유사도 커널 → 재정렬. "품질 + 다양성 Top-K"의 표준 엔지니어링 레퍼런스.
- **서브모듈러 커버리지**: Tschiatschek et al., "Learning Mixtures of Submodular Functions
  for Image Collection Summarization", NIPS 2014. 그리디 보장은 Nemhauser et al. 1978의
  (1−1/e). → 1차 구현은 그리디 + 장면/타임라인 커버리지 제약으로 충분.
- **앨범 큐레이션**: CUFED / "Event-Specific Image Importance" (Wang et al., **CVPR 2016** +
  [arXiv:1707.05911](https://arxiv.org/abs/1707.05911)) — 1,883 앨범, 23 이벤트 타입(웨딩
  포함), ~94,797장. ⚠ 공식 UCSD 배포 페이지 접속 불가(2026-08) — 입수 가능성 재확인 필요.

### 실시간 재추천 (세션 내 선택 반영)

- **가장 가까운 선행**: Yeh & Barsky et al., "Personalized Photograph Ranking and Selection
  System Considering Positive and Negative User Feedback", **ACM TOMM 2014**.
  [DOI](https://dl.acm.org/doi/10.1145/2584105) — 사용자의 선택/제외 예시 기반 재정렬,
  예시 기반이 특징 기반보다 2:1로 선호됨, 이진 정확도 ~93%. **"고른 사진으로 남은 후보
  재정렬"의 직접 선행 연구.**
- 온라인 학습 정본: LinUCB (Li et al., WWW 2010), Dueling Bandits (Yue & Joachims).
  1차 구현은 밴딧까지 갈 필요 없이 **선택 사진과의 임베딩 유사도 가중 재정렬**로 시작.
- 상용 시스템 공개 자료: Google Photos Memories
  ([PAIR 블로그](https://medium.com/people-ai-research/a-snapshot-of-ai-powered-reminiscing-in-google-photos-5a05d2f2aa46)
  — 메타데이터 필터 → 중복 제거 → 미학 점수 → 테마), Apple ANSA
  ([Apple ML Research 2022](https://machinelearning.apple.com/research/on-device-scene-analysis)
  — 단일 백본으로 태깅+미학+중복 제거). 논문 수준 공개는 없음.

## ⑤ 상용 AI 사진 서비스의 판단 기준 — 퍼널 대응 확인

상용 서비스들이 실제로 쓰는 판단 기준을 조사한 결과, **우리 4단계 퍼널과 사실상 동일한
기준**을 쓴다 (기준의 타당성 검증). 단 방법론을 논문으로 공개한 곳은 없다.

| 서비스 | 공개된 판단 기준 | 퍼널 대응 |
|---|---|---|
| **Aftershoot** (웨딩 컬링) | 사진당 1~100점, "30개 이상 요인" — 눈 감김·블러(객관) + 조명·배경·구도(주관). 유사 사진 그룹핑 후 그룹 내 베스트 선택. 결과 필터: Selected/Highlights/**Duplicates/Blurry/Closed Eyes** | ①+②+③ |
| **Narrative Select** (웨딩 컬링) | 초점·눈 평가를 신호등으로 표시 (초록=눈 뜸·선명 / 노랑=실눈·소프트 포커스 / 빨강=눈 감김·못 쓸 블러), 장면(scene) 그룹 내 선명도 랭킹으로 베스트 프레임 우선 표시. **"AI는 정보만, 최종 선택은 사진가"를 명시** | ①+③ |
| **FilterPixel DeepCull** | 웨딩 특화 학습 주장 (본식·축사·퍼스트댄스 시퀀스 인지) — 마케팅 수준 | ①+③ |
| **Google Top Shot** (Pixel) | 눈 뜸 + 표정 + 블러 + 모션을 GAM으로 합성해 연사 베스트 프레임 선택 | ③ |
| **Google Photos 추억** | 메타데이터 필터 → 문서/영수증 제거 → 근접 중복 제거 → 미학 점수(블러·조명) → 테마 구성 | ①②③→테마 |
| **Apple 사진 앱** (ANSA) | 단일 백본으로 태깅 + 미학 점수 + 임베딩 중복 제거 → "추억" 큐레이션 | ②③④ 일부 |

시사점:

1. **①③(기술 품질 + 중복 그룹 베스트)이 상용 컬링의 본체**고 ②(미학)는 보조 — 우리 원가
   배분(①③ 경량 전수, ② 임베딩 재사용)과 방향이 같다.
2. **④(개인화·앨범 커버리지)는 상용 컬링 툴이 하지 않는 영역** — 컬링까지만 하고 앨범
   구성은 사진가 몫으로 남긴다. 부부의 세션 내 선택 기반 실시간 재추천 + 장면 커버리지가
   실질적 차별 지점.
3. Narrative Select의 "AI 어시스트, 결정은 사람" 포지셔닝은 우리 하드 필터 금지 원칙과
   동일 — 시장에서 검증된 포지션이라는 방증.

출처: [Aftershoot Selects](https://aftershoot.com/selects/) ·
[Aftershoot 컬링 가이드](https://support.aftershoot.com/en/articles/5223473-get-started-with-aftershoot-culling) ·
[Aftershoot 컬링 환경설정](https://support.aftershoot.com/en/articles/6508163-setting-your-ai-automated-culling-preferences-in-aftershoot) ·
[Narrative Select](https://narrative.so/select) ·
[SLR Lounge Narrative 리뷰](https://www.slrlounge.com/narrative-select-review/) ·
Google/Apple 출처는 §③·§④ 참조.

## ⑥ 논문 니치 확인 (K-Wedding Photo Curation)

- "wedding photo selection/curation" 계열 학술 검색 결과: **동료평가 논문 0건.** 존재하는 것은
  상용 툴 마케팅, 소셜 정보 기반 앨범 특허(US 9,349,052 등), 그리고 웨딩을 23개 이벤트 중
  하나로 포함하는 CUFED뿐.
- 가장 가까운 학술 이웃: REAL-CUR(개인 앨범 14개), Yeh & Barsky TOMM 2014, Photo Triage.
- 앨범 수준 VLM 벤치마크가 CVPR 2026에 등장(Beyond Single Images) — related work 앵커로 유용.
- **(a) 실제 웨딩 갤러리 (b) 세션 내 선택만을 개인화 신호로 (c) 품질+선호+다양성+커버리지
  Top-K를 작가/부부 정답과 대조 평가**하는 논문은 빈 니치가 맞다. `../dataset` 골든셋
  스키마를 벤치마크 재사용 가능하게 설계할 가치가 있다.

## ⑦ 라이선스 매트릭스 (채택 판단 기준)

| 사용 가능 (상업) | 라이선스 | 사용 불가/위험 | 사유 |
|---|---|---|---|
| ARNIQA, MUSIQ, VILA, TANet, Charm, LAION v2, idealo-NIMA, MediaPipe(모델 포함), HSEmotion | Apache-2.0 | pyiqa 라이브러리 | PolyForm NC |
| HyperIQA, LIQE, 6DRepNet, L2CS-Net, titu1994-NIMA | MIT | Q-Align, QualiCLIP, SER-FIQ, CR-FIQA, PIQ23 | 비상업 조항 |
| — | — | InsightFace 사전학습 모델, dlib 68 랜드마크 | 모델만 비상업 / iBUG 300-W |
| — | — | Microsoft BoB 유사 구현 | 특허 US 10,671,895 |

기타 그레이존: AVA 데이터셋 자체는 라이선스 불명(관행상 학습 모델의 상업 사용은 통용 —
LAION·idealo 선례), HumanAesExpert(README MIT vs LICENSE 부재), EfficientFace·DDAMFN
라이선스 미확인, CUFED·Princeton Triage 데이터셋 호스팅 사망.

## ⑧ plan.md 반영 사항

1. §3 ①의 SER-FIQ/CR-FIQA는 **후보에서 제외** (비상업 + 인식 유틸리티 측정) → 얼굴 신호는
   MediaPipe로 일원화.
2. §3 ①②의 스파이크는 pyiqa 경유가 아니라 **원본 repo 직접 사용** 기준으로.
3. §3 ②에 Charm(CVPR 2025)·LAION v2(CLIP 재사용 시 무료) 추가.
4. §6 스파이크 선정 기준 (b)에 특허(US 10,671,895) 확인 추가.
5. §7 리스크에 데이터셋 호스팅 유실(Princeton Triage, CUFED) 반영 — 골든셋 자체 구축의
   중요성 상승.
