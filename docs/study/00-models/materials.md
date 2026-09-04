# 00 모델 카탈로그 — 학습 자료

> 이 영역의 본체는 `catalog.md`와 `lab/`이다. 이 목록은 **더 알고 싶을 때** 보는 것이고,
> 전부 읽을 필요는 없다. `catalog.md`의 각 모델 절에서 필요한 항목만 찾아 읽으면 된다.
>
> 링크는 원 문서에서 옮겨 온 것이라 **접근 검증은 각자 할 것**(다른 영역 자료와 달리
> 일괄 확인을 안 했다). 죽은 링크는 발견하는 대로 지운다.

## 먼저 — 읽기 전에 돌려 보기

문서보다 `lab/step1`·`lab/step2`가 먼저다. **무엇에 반응하는지 본 뒤에 읽으면
논문이 훨씬 빨리 읽힌다.** 반대 순서로 하면 대부분 기억에 안 남는다.

---

## 1. MediaPipe Face Landmarker

- [ ] **공식 가이드 — Face Landmarker** · 20~30분 · 영어
  https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker
  입출력 형식과 옵션. 우리 러너가 쓰는 `output_face_blendshapes=True`가 여기 있다.
- [ ] **blendshape 52개 목록** · 10분 · 위 문서 내 "Face landmarker model" 절
  `eyeBlinkLeft`, `mouthSmileRight` 같은 이름을 직접 보면 `eyes_open` 계산식이 이해된다.
- [ ] (선택) **MediaPipe Face Mesh 논문** · https://arxiv.org/abs/1907.06724
  랜드마크 478개가 어떻게 나오는지. 우리 용도에는 과하다.

---

## 2. ARNIQA — 기술 품질(NR-IQA)

- [ ] **ARNIQA 논문 (WACV 2024)** · https://arxiv.org/abs/2310.14918
  §3(방법)과 그림 2만 봐도 충분하다 — **"왜곡 공간을 자기지도로 학습한다"** 는 아이디어.
- [ ] **공식 repo** · https://github.com/miccunifi/ARNIQA
  우리 러너가 `torch.hub`로 부르는 그 코드. `single_image_inference` 데모를 보면
  우리가 왜 전체 이미지와 half-scale 두 개를 넣는지 알 수 있다.
- [ ] (선택) **NR-IQA 개관** — "no-reference image quality assessment" 로 검색.
  참조 이미지 없이 품질을 매기는 문제군 전체의 지도.

**대안 후보** (1순위가 부족할 때만, `spike-report.md` 남은 작업)
HyperIQA · LIQE · MUSIQ · Charm

---

## 3. LAION Aesthetic Predictor v2 — 미학

- [ ] **improved-aesthetic-predictor repo** · https://github.com/christophschuhmann/improved-aesthetic-predictor
  MLP 구조가 그대로 있다. 우리 러너의 `_MLP` 클래스가 이것을 옮긴 것.
- [ ] **CLIP 논문 리뷰 (한국어)** · 30~40분 — 03 영역 `materials.md`의 ffighting.net 항목
  **미학 점수의 무거운 절반이 CLIP이다.** 03과 겹치므로 거기서 한 번만 읽으면 된다.
- [ ] (선택) **AVA 데이터셋** — 미학 점수의 정답이 어디서 왔는지.
  "일반 사진 데이터셋이지 웨딩 사진이 아니다"를 확인하는 용도.

---

## 4. 오픈 VLM (Gemma 3 / Qwen2.5-VL)

04 영역과 겹친다. **여기서는 "어떤 모델을 고를까"만 보고, 원리는 04에서 본다.**

- [ ] **Gemma 3 발표 블로그** · https://blog.google/technology/developers/gemma-3/
- [ ] **gemma-3-12b-it 모델 카드** · https://huggingface.co/google/gemma-3-12b-it
  **라이선스(Gemma ToU) 상업 조건을 여기서 확인한다** — 채택 전 필수 항목.
- [ ] **Qwen2.5-VL 블로그** · https://qwenlm.github.io/blog/qwen2.5-vl/
- [ ] **Qwen2.5-VL-7B 모델 카드** · https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- [ ] **vLLM Structured Outputs** · https://docs.vllm.ai/en/latest/features/structured_outputs.html
  `catalog.md`가 말한 guided decoding의 프로덕션 쪽. 우리가 Ollama로 흉내 낸 것의 원본.

---

## 5. DINOv2 — 임베딩

03 영역이 본체다. 여기서는 카탈로그용으로만.

- [ ] **Meta AI — DINOv2 공식 블로그** · 15~20분 · https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/
- [ ] **kimjy99 — DINOv2 논문 리뷰 (한국어)** · 30~40분 · https://kimjy99.github.io/논문리뷰/dinov2/

---

## 공통 배경 — "남이 학습한 모델을 가져다 쓴다"는 것

- [ ] **Hugging Face — Model Cards 개념** · 15분 · https://huggingface.co/docs/hub/model-cards
  모델 카드에서 무엇을 확인해야 하는가 — 학습 데이터·의도된 용도·한계·라이선스.
  `catalog.md`의 다섯 칸이 사실상 이 형식을 우리 문제에 맞춘 것이다.
- [ ] (선택) **transfer learning / feature extraction 개관** — StatQuest나 Andrew Ng Course 2.
  "무거운 backbone + 작은 head" 구조(LAION Aesthetic이 정확히 그것)의 일반형.

---

## 우리 repo 안의 1차 자료 — 이게 가장 정확하다

문서보다 **실제로 돌아가는 코드**가 먼저다. 전부 주석이 촘촘하다.

| 파일 | 무엇 |
|---|---|
| `photoselect/scripts/spike/runners/mediapipe_faces.py` | MediaPipe 호출과 `eyes_open` 계산식 |
| `photoselect/scripts/spike/runners/arniqa.py` | ARNIQA 입력 두 개(전체·half-scale)의 이유 |
| `photoselect/scripts/spike/runners/laion_aesthetic.py` | CLIP + MLP 두 층 구조 |
| `photoselect/scripts/spike/runners/common.py` | **미리보기 1600px 고정** — lab/step2가 왜 중요한지의 근거 |
| `photoselect/scripts/spike/vlm_tag.py` | 고정 축 어휘와 enum 강제 |
| `photoselect/docs/spike-report.md` | 다섯 모델의 **실측치와 미해결 항목** |
| `photoselect/docs/tech-stack.md` §2 | 모델 선정 근거와 후보 비교표 |
