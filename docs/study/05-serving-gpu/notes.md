# 05 추론 서빙과 GPU — 노트

기존 강점(ECS·GPU 인스턴스·Terraform)과 시너지가 가장 큰 영역. 멘토 질문 "L4에서
0.3초/장이 현실적인가"에 스스로 답하는 것이 목표. `/study-quiz 05`가 출제 근거로 쓴다.

## 개념 체크리스트

- [ ] GPU 메모리 계산 — 가중치 = 파라미터 수 × 정밀도 바이트, KV cache
      (L4 24GB에 12B FP16 = 24GB → 빠듯, 양자화 필요라는 결론 직접 유도)
- [ ] 양자화 — AWQ, GPTQ, FP8 개념과 품질-메모리 트레이드오프
- [ ] vLLM 핵심 — PagedAttention, continuous batching
- [ ] throughput vs latency — 배치 처리(트랙 A)는 throughput 최적화 문제

## 정리

<!-- 개념별로: ## 개념명 / 내 설명 / 설계안 연결 -->
