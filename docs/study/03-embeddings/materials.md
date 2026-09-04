# 03 임베딩과 표현학습 — 학습 자료

> 모든 URL은 2026-08-23 실제 접근 검증됨 (미검증 섹션 제외). 학습 완료한 자료는 체크한다.
> 권장 총 시간: 핵심 코스 5~6시간, 심화 포함 10~12시간.

## 필수

- [ ] **3Blue1Brown — Essence of Linear Algebra (Ch.1·2·9 최소 코스)** · 영상 ~2시간 · 영어
  Ch.1 벡터: https://www.youtube.com/watch?v=fNk_zzaMoSs (웹 텍스트판: https://www.3blue1brown.com/lessons/vectors)
  벡터·선형결합·**내적(Ch.9)**의 기하학적 의미. 코사인 유사도 = 정규화된 내적이라는 직관.
  영어 청취 부담이면 웹 텍스트판 권장.
- [ ] **ffighting.net — CLIP 논문 리뷰** · 블로그 30~40분 · 한국어
  https://ffighting.net/deep-learning-paper-review/multimodal-model/clip/
  contrastive learning, 코사인 유사도 zero-shot 분류 — 논문 원문 전에 읽는 최적 선행 자료.
- [ ] **How HDBSCAN Works (공식 문서)** · 문서 40~60분 · 영어
  https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
  mutual reachability → MST → 계층 → 안정성 추출 5단계. 이미 쓴 `min_cluster_size`가
  알고리즘 어느 단계에 꽂히는지 그림으로 이해. k-means와의 근본 차이가 자연히 드러남.

## 선택

- [ ] **Meta AI — DINOv2 공식 블로그** · 15~20분 · 영어
  https://ai.meta.com/blog/dino-v2-computer-vision-self-supervised-learning/
  "caption 기반 특징은 국소 정보 이해가 부족" — CLIP=의미 vs DINOv2=시각 구조라는 핵심
  질문에 Meta가 직접 답하는 글. 논문 전 진입점.
- [ ] **Dale Seo — pgvector로 PostgreSQL 벡터 검색** · 블로그 30~40분 · 한국어
  https://daleseo.com/pgvector/
  거리 연산자 `<->`(유클리드)/`<#>`(내적)/`<=>`(코사인) 선택 기준, HNSW vs IVFFlat —
  우리 스택 그대로 이론과 운영을 연결.
- [ ] **kimjy99 — DINOv2 논문 리뷰 (한국어)** · 30~40분 · https://kimjy99.github.io/논문리뷰/dinov2/
- [ ] **에스코어 — 벡터 임베딩 모델의 이해와 활용** · 20~30분 · 한국어
  https://s-core.co.kr/insight/view/벡터-임베딩-모델의-이해와-활용-최적의-임베딩-모델/
  임베딩 개념·차원-비용 트레이드오프·MTEB — 백엔드 개발자의 언어로 쓰인 글.

## 심화 (원전)

- [ ] **CLIP 논문** · https://arxiv.org/abs/2103.00020 — §2(방법)+그림 1~3만 발췌 독 1~2시간
- [ ] **DINOv2 논문** · https://arxiv.org/abs/2304.07193 — 서론+방법 개요 발췌 독 1~2시간
- [ ] **HF Computer Vision Course — Unit 4 Multimodal** · 강좌 4~6시간 · 영어
  https://huggingface.co/learn/computer-vision-course/en/unit4/multimodal-models/clip-and-relatives/clip
  "Losses" 섹션이 contrastive vs self-distillation 차이를 이론적으로 정리. Unit 3(ViT)은
  DINOv2 백본 이해용.

## 미검증 (직접 확인 필요)

- OpenAI CLIP 공식 블로그 https://openai.com/index/clip/ — Cloudflare 차단, 브라우저로는 열릴 가능성 높음
- 3Blue1Brown 한국어 더빙 채널 — 유튜브에서 "3Blue1Brown 한국어" 검색
- HDBSCAN docs — Comparing Python Clustering Algorithms
  https://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html (k-means 대비 직접 비교)
- Luna AI Blog DINOv2 리뷰 https://lunaleee.github.io/posts/dinov2/ · greeksharifa CLIP 리뷰

## 권장 순서

1. 3B1B Ch.1·2·9 → 2. pgvector 글(배운 내적·코사인을 실제 연산자와 연결) →
3. ffighting CLIP 리뷰 + Meta DINOv2 블로그("왜 둘 다 쓰는가"의 답) →
4. How HDBSCAN Works → 5. (심화) 논문 발췌 독 + HF CV Course Unit 4
