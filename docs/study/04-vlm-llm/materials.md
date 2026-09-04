# 04 VLM/LLM 동작 원리 — 학습 자료

> 모든 URL은 2026-08-23 실제 접근 검증됨 (미검증 섹션 제외). 학습 완료한 자료는 체크한다.
> 권장 총 시간: 필수만 ~3시간, 전체 6~9시간.

## 필수

- [ ] **3Blue1Brown — Transformers (Deep Learning Ch.5)** · 영상 27분 · 영어(한글 자막)
  https://www.youtube.com/watch?v=wjZofJX0v4M (글 버전: https://www.3blue1brown.com/lessons/gpt)
  토큰·임베딩·"다음 토큰 예측기"로서의 LLM. 수식 없는 최고의 시각적 입문.
- [ ] **3Blue1Brown — Attention step-by-step (Ch.6)** · 영상 26분 · 영어(한글 자막)
  https://www.youtube.com/watch?v=eMlx5fFNoYc (글: https://www.3blue1brown.com/lessons/attention)
  Q/K/V, multi-head, masking. "attention = 문맥에 따라 임베딩을 갱신하는 메커니즘" 직관.
- [ ] **HF Blog — Vision Language Models Explained** · 블로그 40분~1시간 · 영어
  https://huggingface.co/blog/vlms
  **VLM 파트는 이 글 하나로 개요 충족** — image encoder + projector + text decoder 표준
  구조, LLaVA 예시(CLIP 인코더 + projector + Vicuna), projector 우선 학습 방식.
- [ ] **vLLM 공식 문서 — Structured Outputs** · 문서 30~40분 · 영어
  https://docs.vllm.ai/en/latest/features/structured_outputs.html
  choice/enum·JSON schema·regex, 백엔드(xgrammar·guidance). 우리 "고정 축 enum 태그"의
  바로 그 기능. ⚠️ v0.12.0에서 구 API(`guided_json`/`guided_choice`)가
  `structured_outputs` 파라미터로 통일됨 — 옛 블로그 예제 따라하지 말 것.

## 선택

- [ ] **Illustrated Transformer 한글 번역** · 블로그 1시간 · 한국어 · https://nlpinkorean.github.io/illustrated-transformer/
  3B1B 후 행렬 흐름을 한 단계 구체화. 원문: https://jalammar.github.io/illustrated-transformer/
- [ ] **LLaVA 논문** · 논문 · 영어 · https://arxiv.org/abs/2304.08485
  §Architecture 그림 하나(vision encoder → projection W → LLM)만 봐도 충분. 원전 확인용.
- [ ] **HF LLM Course Ch.1** · 강좌 · 영어 · https://huggingface.co/learn/llm-course/chapter1/1
  개념→코드 연결. Ch.1만 우선, 나머지는 필요할 때 발췌.
- [ ] **혁펜하임×테디노트 — 왜 RNN보다 트랜스포머인가** · 유튜브 ~1시간 · 한국어
  https://www.youtube.com/watch?v=slqdel6HoCI — attention이 왜 필요했는지 역사적 맥락.

## 후보 모델 1차 소스 (스파이크 직결, ~1.5시간)

- [ ] **Gemma 3 발표 블로그** · https://blog.google/technology/developers/gemma-3/
  1B/4B/12B/27B, 4B 이상만 비전, 단일 GPU 설계.
- [ ] **gemma-3-12b-it 모델 카드** · https://huggingface.co/google/gemma-3-12b-it
  이미지 896×896 → **256 토큰** — 처리량(1,000장 ≤ 6분) 계산의 기초 데이터.
- [ ] **Qwen2.5-VL 블로그** · https://qwenlm.github.io/blog/qwen2.5-vl/
  동적 해상도, 안정적 JSON 출력 강조 — 고정 축 태깅과 궁합.
- [ ] **Qwen2.5-VL-7B 모델 카드** · https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
  Apache 2.0, vLLM 서빙 예제 포함 — 실습 진입점.

**"파라미터 수 vs 성능" 학습법**: 단일 개요 자료가 없다. gemma-3-12b-it vs 27b-it 모델
카드의 동일 벤치마크(DocVQA·ChartQA·MMMU) 표를 직접 비교하라 — 지식·추론에서 격차가
크고 단순 인식·포맷 준수(태깅류)에서 좁아지는 패턴이 수치로 보인다. 이것이 "태깅은
소형 모델로 충분할 수 있다"는 스파이크 가설의 근거.

## 미검증 (직접 확인 필요)

- 위키독스 딥러닝 NLP 입문 https://wikidocs.net/book/2155 — 봇 차단으로 검증 불가,
  브라우저로는 정상 접속으로 알려짐. attention 챕터: https://wikidocs.net/22893
- 혁펜하임 트랜스포머 강의(인프런 유료 "직관적으로 이해하는 딥러닝 트랜스포머" 존재
  가능) — 가격·커리큘럼 미확인
- Qwen2.5-VL-32B 모델 카드 https://huggingface.co/Qwen/Qwen2.5-VL-32B-Instruct —
  "31B급 비교" 대상 추정, 미검증
