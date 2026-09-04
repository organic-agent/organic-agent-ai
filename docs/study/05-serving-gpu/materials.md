# 05 추론 서빙과 GPU — 학습 자료

> 모든 URL은 2026-08-23 실제 접근 검증됨 (미검증 섹션 제외). 학습 완료한 자료는 체크한다.
> 워크로드 기준: g6.xlarge(L4 24GB) + vLLM, 배치 사진 태깅 — **지연시간이 아니라 처리량 게임**.

## 필수

- [ ] **vLLM 공식 Quickstart** · 문서+실습 1~2시간 · 영어
  https://docs.vllm.ai/en/latest/getting_started/quickstart.html
  **오프라인 배치 추론(`LLM` 클래스)** — 배치 A와 정확히 같은 사용 패턴. 첫 예제가 곧
  프로덕션 코드의 뼈대.
- [ ] **Anyscale — How continuous batching enables 23x throughput** · 블로그 40~60분 · 영어
  https://www.anyscale.com/blog/continuous-batching-llm-inference
  static vs continuous batching, "왜 vLLM인가"의 표준 레퍼런스. 처리량 직관의 출발점.
- [ ] **최신 LLM 양자화 기술 총정리 (2026)** · 블로그 1시간 · **한국어**
  https://www.youngju.dev/blog/2026-07-08-llm-quantization-2026
  GPTQ/AWQ/SmoothQuant/FP8/KV 캐시 양자화 + 실측("4bit 양자화해도 메모리는 절반쯤만 준다").
  태스크별 평가셋 검증 강조 — 우리 골든셋 평가 방침과 일치.

## GPU 메모리 계산

- [ ] **NVIDIA L4 제품 페이지** · 10분 · https://www.nvidia.com/en-us/data-center/l4/
  24GB GDDR6, **대역폭 300GB/s**(A100의 1/6) — 이 숫자가 배치 크기·처리량 설계의 핵심 제약.
- [ ] **HF Model Memory Calculator** · 도구 10분 · https://huggingface.co/spaces/hf-accelerate/model-memory-usage
  Qwen2.5-VL 7B / Gemma 3 12B를 dtype별로 넣어보고 손계산(파라미터×바이트) 검산.
- [ ] **kipply — Transformer Inference Arithmetic** · 심층 블로그 2~3시간 · 영어 (난이도 높음)
  https://kipp.ly/transformer-inference-arithmetic/
  KV 캐시 공식, FLOPs ≈ 2P, **memory-bandwidth-bound vs compute-bound 판별** — 백엔드의
  "용량 산정" 감각을 GPU로 이식. 멘토 질문 "L4에서 0.3초/장 현실적인가"에 답하는 핵심 도구.

## 양자화

- [ ] **Maarten Grootendorst — A Visual Guide to Quantization** · 1~1.5시간 · 영어
  https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization
  그림 50+장. FP32~INT8 표현, PTQ vs QAT, GPTQ. ⚠️ AWQ는 미포함 — 한국어 총정리로 보완.
- [ ] **vLLM 공식 Quantization 문서** · 30~40분 · https://docs.vllm.ai/en/latest/features/quantization/index.html
  **L4(Ada)에서 실제 지원되는 양자화의 최종 답** — Marlin 커널로 GPTQ/AWQ/FP8, Quantized KV Cache.
- [ ] **HF — natively supported quantization schemes** · 30분 · 2023년 글이라 최신 글과 병행
  https://huggingface.co/blog/overview-quantization-transformers

## vLLM 내부 (PagedAttention)

- [ ] **vLLM 공식 발표 블로그** · 30분 · https://vllm.ai/blog/2023-06-20-vllm
  KV 캐시 비연속 블록화(OS 페이징 착안), 메모리 낭비 60~80% → 4% 미만. 1차 소스.
- [ ] **스캐터랩 — 최대 24배 빠른 vLLM의 비밀** · 1.5~2시간 · **한국어**
  https://blog.scatterlab.co.kr/vllm-implementation-details
  국내에서 가장 깊은 코드 레벨 분석. v0.1.2 기준이지만 개념 구조는 유효.
- [ ] **PagedAttention 논문 한국어 리뷰** · 1시간 · https://pangyoalto.com/pagedattetion-review/
- [ ] **SK DevOcean — vLLM로 효율적인 모델 서빙하기** · 40분 · **한국어**
  https://devocean.sk.com/blog/techBoardDetail.do?ID=167138&boardType=techBlog
  배칭 전략 3종 비교 + 온라인 vs 오프라인(배치) 서빙 구분 — 배치 A 관점과 맞닿음.
- [ ] **NVIDIA — Mastering LLM Inference Optimization** · 1.5시간 · 영어 (전체 지도·복습용)
  https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/

## 유료 (선택)

- **FastCampus — vLLM을 활용한 고성능 저비용 LLM 서빙** · 한국어 강의 56클립 ~8시간 ·
  확인 시점 ₩293,000 · https://fastcampus.co.kr/data_online_vllm
  무료 자료로 충분히 커버 가능해 우선순위 낮음. 한국어로 몰아서 배우고 싶을 때만.

## 미검증 (직접 확인 필요)

- vLLM 완벽 가이드 (YouTube) https://www.youtube.com/watch?v=4eaiFbB6psk
- DevOcean — vLLM 기술적 혁신 (2025-04) https://devocean.sk.com/blog/techBoardDetail.do?ID=167416&boardType=techBlog
- AWQ 원논문 https://arxiv.org/abs/2306.00978 · vLLM/PagedAttention 원논문(SOSP'23) https://arxiv.org/abs/2309.06180

## 권장 순서

1. **메모리 감각**: L4 스펙 → 계산기 → kipply(KV 캐시·bandwidth-bound)
2. **vLLM 원리**: 공식 블로그 → Anyscale continuous batching → (한국어 심화) 스캐터랩/논문 리뷰
3. **양자화**: Visual Guide → 한국어 총정리 → vLLM 문서로 L4 지원 확인
4. **손 실행**: Quickstart 오프라인 배치 → g6에서 처리량 실측("1,000장 ≤ 6분" 검증 직결)
5. **정리**: NVIDIA 종합 글로 복습

**설계 메모**: L4는 대역폭이 약점 → 12B급은 AWQ 4bit + FP8 KV 캐시 조합이 사실상 전제.
`max_num_seqs`·`gpu_memory_utilization`을 키워 GPU를 놀리지 않게 하는 것이 목표.
