"""preference — 부부의 최종 선택(photo_selection_items)으로 계속 학습하는 선호 가중치 층.

산출물은 ML 모델 파일이 아니라 **가중치 벡터 한 행**(스칼라 11 + 임베딩 1536 + bias)이다. 학습은 이 모듈(numpy·scipy,
torch 없음)이 갤러리 CLOSED 때 하고, 추론은 wes 가 사진마다 내적 한 번으로 한다:

    score = z(prior) + λ(n) · z(w·x + b)          prior = 0.5·tech_pct + 0.5·aes_pct (지금 그대로)

입력은 전부 DB 에 있는 것이다 — embedder 의 embedding(DINOv3), score 의 clip_embedding·pct·subjects·cluster.
photo_ratings 는 읽지 않는다(CLAUDE.md). photo_selection_items 는 읽기만 한다.

패키지는 embedder 와 같은 층으로 나뉜다 — controller → service → repository, 모두가 domain 을 본다.
특징 순서 계약 `FEATURE_SPEC` 은 `domain/features.py` 에 있다. 설계: docs/photoselect/plan-preference-layer.md (로컬).
"""
