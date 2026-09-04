# 04 VLM/LLM 동작 원리 — 노트

개요 수준. 세부 수식은 불필요 — 12B vs 31B 선택, enum 강제 태깅, 캡션 품질을 **판단**할
수 있으면 된다. `/study-quiz 04`가 이 파일을 출제 근거로 쓴다.

## 개념 체크리스트

- [ ] Transformer attention 개념 (개요)
- [ ] VLM 표준 구조 — vision encoder + projector + LLM (LLaVA)
- [ ] 구조화 출력 — vLLM guided decoding, enum 강제의 원리와 한계
- [ ] 파라미터 수와 성능 — 왜 31B가 12B보다 나은 경향, 어떤 태스크에서 차이가 줄어드는가

## 정리

<!-- 개념별로: ## 개념명 / 내 설명 / 설계안 연결 -->
