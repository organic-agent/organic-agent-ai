"""categorize — 갤러리 단위 그룹화·이름 Lambda. **torch 없음** (numpy · scipy · Bedrock).
embedder·score 와 같은 평탄 구조: handler(Lambda) / __main__(CLI) → job.run → pipeline.

    pipeline     백분위 · 연사 클러스터 · concat(DINOv3⊕CLIP) 임베딩 그룹 → photo_analysis (pct · cluster · group)
                 → naming
    naming       ② 이름(Bedrock, 부모는 닫힌 목록) ③ 소그룹 최근접 배정 ④ 저장된 clip_parent 다수결 검증
                 → ai_concept_assignments
    cluster      연사(burst) union-find — 카메라 파티션 ∧ 순서 창 ∧ 코사인 임계
    concept      임베딩 그룹 경계 — 평균연결 계층 클러스터, 적응 임계
    llm          BedrockClient.complete_json (JSON 스키마 강제, 이미지 블록)
    job          잡: ai_analysis_jobs RUNNING → DONE/FAILED (체인의 끝)
    handler      Lambda: {galleryId, jobId} — score 가 체인으로(FULL), wes 가 직접(NAMING)
    config · db · store · gallery · storage · jobs

입력은 전부 DB 에 저장된 것이다 — embedder 의 embedding, score 의 sub_scores·subjects·clip_embedding·model_version.
실제 폴더(concept_folders/detail_folders)는 wes 가 ai_concept_assignments 를 읽어 만든다(POST /concept-folders/ai).
score 와 공유하는 계약: MODEL_VERSION · PARENTS · photo_analysis 컬럼 경계 (config.py 머리말).
"""
