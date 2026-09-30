"""categorize — 갤러리 단위 컨셉 폴더(1층·2층) Lambda. **torch 없음** (numpy · scipy · Bedrock).
embedder·score 와 같은 층 구조(#131): controller → service → repository · infrastructure, 모두가 domain 을 본다.

    controller/handler      Lambda: {galleryId, jobId, conceptCount?} → service.job.run (Bedrock 클라이언트 주입). __main__ 은 로컬 CLI
    service/job             갤러리 하나: 대상 조회 → pipeline → (실패면 analysis_jobs.error). 상태 전이는 wes(#95)
    service/pipeline        백분위 · 연사 · 구간 → 1층(naming) → 1층 안 2층 그룹 → photo_analysis → 2층 이름
    service/segment         1층 구간 — 촬영 시각 공백, 시각이 없으면 DINOv3 Ward
    service/naming          ① 구간 타일 → VLM 컨셉 묶기·이름 ② 1층별 2층 이름 · 최근접 · 검증 → concept_assignments
    service/burst           연사 union-find — 카메라 파티션 ∧ 순서 창 ∧ 코사인 임계
    service/grouping        2층 그룹 경계 — 1층마다 평균연결 계층 클러스터, 과분할 가드
    domain/                 photo(PhotoRef) · analysis(PhotoAnalysis · ConceptAssignment · GalleryRead · Store) · run(Grouped · CategorizeResult)
    repository/             connection · analysis(DbStore) · local(LocalStore) · photos(대상 조회) · jobs(error 한 컬럼) · storage(S3)
    infrastructure/bedrock  BedrockClient.complete_json (JSON 스키마 강제, 이미지 블록)
    config/settings         Settings · Knobs · LlmKnobs · PIPELINE_VERSION

입력은 전부 DB 에 저장된 것이다 — embedder 의 embedding, score 의 sub_scores·subjects·clip_embedding·pipeline_version.
실제 폴더(concept_folders/detail_folders)는 wes 가 concept_assignments 를 읽어 만든다(POST /concept-folders/ai).
score 와 공유하는 계약: PIPELINE_VERSION · photo_analysis 컬럼 경계 (config/settings.py 머리말).
"""
