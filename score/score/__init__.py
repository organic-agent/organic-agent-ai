"""score — 사진별 점수. handler(Lambda) / __main__(CLI) → service.job.run → service.pipeline. embedder(#121)와 같은 층 구조.

    controller/      handler.py(Lambda: {galleryId, photoIds} 하나) · sagemaker.py(SageMaker train — GPU 벤치마크)
    service/         job(사진 목록·갤러리 → 다운로드 → pipeline) · pipeline(SCORE 본체: CLIP → 미학·피사체·부모, ARNIQA, 고전 지표)
                     · worker(GPU 집기 루프: photo_analysis SKIP LOCKED 32장 → 점수 → commit, 유휴면 자기 정지)
                     · classical(Laplacian 선명도 · 노출 클립 · bg_luma) · subjects(CLIP zero-shot 태거)
    domain/          photo(PhotoRef · PhotoAnalysis) · run(ScoreResult) · errors(PREVIEW_MISSING · SCORE_FAILED)
    repository/      connection(Postgres 접속) · photos(대상 조회) · store(LocalStore · DbStore — write_scores · claim_batch)
                     · storage(S3 미리보기 · download_previews) · dataset(로컬 데이터셋 폴더)
    infrastructure/  runners/(ArniqaRunner · LaionRunner — torch 는 여기만) · device · images(PIL) · ec2(IMDS · StopInstances) · gpu(NVML)
    config/          settings(Settings · Knobs · MODEL_VERSION · PARENTS · PARENT_PROMPTS)

의존은 한 방향이다 — controller → service → repository, service → infrastructure, 모두가 domain·config 를 본다.

v2(#98): 잡 상태·categorize 호출·재시도는 **wes 가 소유한다**. score 는 점수만 쓴다 — 운영에서는 GPU 워커가 집어 처리하고,
워커가 없거나 못 따라가면 wes 가 Lambda 를 {galleryId, photoIds} 로 부른다. 그룹·이름은 categorize 모듈(#35).
categorize 와 공유하는 계약: MODEL_VERSION · PARENTS · photo_analysis 컬럼 경계 (config/settings.py 머리말).
"""
