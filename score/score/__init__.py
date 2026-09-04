"""score — 사진별 점수 Lambda. embedder 와 같은 평탄 구조: handler(Lambda) / __main__(CLI) → job.run → pipeline.

    pipeline     SCORE 본체: CLIP ViT-L/14 → LAION 미학 · zero-shot 피사체 · zero-shot 부모 라벨(clip_parent),
                 ARNIQA 기술, 고전 지표 → photo_analysis (subjects · sub_scores · clip_embedding · model_version).
                 사진 단위 재개, write_batch 마다 commit, 데드라인 정지
    job          갤러리 잡: advisory lock · ai_analysis_jobs RUNNING · EMBEDDED 사진 + 미리보기 다운로드 · 결과 기록
    chain        다음 칸(categorize) 호출 — Lambda EVENT 또는 로컬 서브프로세스
    handler      Lambda: 데드라인 앞 정지 → 자기 재호출, 끝나면 chain
    worker       로컬 폴링 워커 (wes 에 invoker 가 생기면 삭제)
    runners/     ArniqaRunner · LaionRunner (torch)
    subjects     CLIP zero-shot — SubjectsTagger · ParentTagger
    classical    Laplacian 선명도 · 노출 클립
    config · db · store · gallery · storage · jobs

체인: wes → score {galleryId, jobId} → categorize {galleryId, jobId} → DONE. 그룹·이름은 categorize 모듈(#35).
categorize 와 공유하는 계약: MODEL_VERSION · PARENTS · photo_analysis 컬럼 경계 (config.py 머리말).
"""
