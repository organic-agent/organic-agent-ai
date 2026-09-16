"""categorize — 갤러리 단위 그룹화·이름 Lambda. **torch 없음** (numpy · scipy · Bedrock).
embedder·score 와 같은 층 구조(#131): controller → service → repository · infrastructure, 모두가 domain 을 본다.

    controller/handler      Lambda: {galleryId, jobId} → service.job.run (Bedrock 클라이언트 주입). __main__ 은 로컬 CLI
    service/job             갤러리 하나: 대상 조회 → pipeline → (실패면 ai_analysis_jobs.error). 상태 전이는 wes(#95)
    service/pipeline        백분위 · 연사 클러스터 · concat(DINOv3⊕CLIP) 임베딩 그룹 → photo_analysis (pct · cluster · group) → naming
    service/naming          ② 이름(Bedrock, 부모는 닫힌 목록) ③ 소그룹 최근접 배정 ④ 저장된 clip_parent 다수결 검증 → ai_concept_assignments
    service/cluster         연사(burst) union-find — 카메라 파티션 ∧ 순서 창 ∧ 코사인 임계
    service/concept         임베딩 그룹 경계 — 평균연결 계층 클러스터, 적응 임계
    domain/                 photo(PhotoRef) · analysis(PhotoAnalysis · ConceptAssignment · GalleryRead · Store) · run(Grouped · CategorizeResult)
    repository/             connection · analysis(DbStore) · local(LocalStore) · photos(대상 조회) · jobs(error 한 컬럼) · storage(S3)
    infrastructure/bedrock  BedrockClient.complete_json (JSON 스키마 강제, 이미지 블록)
    config/settings         Settings · Knobs · LlmKnobs · MODEL_VERSION · PARENTS

입력은 전부 DB 에 저장된 것이다 — embedder 의 embedding, score 의 sub_scores·subjects·clip_embedding·model_version.
실제 폴더(concept_folders/detail_folders)는 wes 가 ai_concept_assignments 를 읽어 만든다(POST /concept-folders/ai).
score 와 공유하는 계약: MODEL_VERSION · PARENTS · photo_analysis 컬럼 경계 (config/settings.py 머리말).
"""
