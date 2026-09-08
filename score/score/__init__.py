"""score — 사진별 점수 Lambda. embedder 와 같은 평탄 구조: handler(Lambda) / __main__(CLI) → job.run → pipeline.

    pipeline     SCORE 본체: CLIP ViT-L/14 → LAION 미학 · zero-shot 피사체 · zero-shot 부모 라벨(clip_parent),
                 ARNIQA 기술, 고전 지표 → photo_analysis (subjects · sub_scores · clip_embedding · model_version).
                 사진 단위 재개, write_batch 마다 commit, 데드라인 정지
    job          사진 목록(운영 폴백) 또는 갤러리 전체(로컬·벤치마크) → 미리보기 다운로드 → pipeline
    handler      Lambda: {galleryId, photoIds} 하나. 데드라인 앞에서 배치 경계 정지
    gpu_worker   GPU 인스턴스 워커 루프: photo_analysis SKIP LOCKED 32장 집기 → 점수 → commit, 유휴면 자기 정지
    runners/     ArniqaRunner · LaionRunner (torch)
    subjects     CLIP zero-shot — SubjectsTagger · ParentTagger
    classical    Laplacian 선명도 · 노출 클립
    config · db · store · gallery · storage · device

v2(#98): 잡 상태·categorize 호출·재시도는 **wes 가 소유한다**. score 는 점수만 쓴다 — 운영에서는 GPU 워커가 집어 처리하고,
워커가 없거나 못 따라가면 wes 가 Lambda 를 {galleryId, photoIds} 로 부른다. 그룹·이름은 categorize 모듈(#35).
categorize 와 공유하는 계약: MODEL_VERSION · PARENTS · photo_analysis 컬럼 경계 (config.py 머리말).
"""
