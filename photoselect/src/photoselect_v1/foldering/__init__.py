"""AI 클러스터링 폴더화 — 갤러리당 1회 배치, 두 잡(#26).

    score       SCORE: 사진별 기술(ARNIQA)·미학(CLIP+LAION)·고전 지표, CLIP zero-shot 피사체·부모 라벨
                → photo_analysis (clip_embedding 저장). torch. 사진 단위 재개
    categorize  CATEGORIZE: 백분위 · 연사 클러스터 · concat(DINOv3⊕CLIP) 임베딩 그룹 → naming
                → photo_analysis (pct·cluster·group) · ai_concept_assignments. torch 없음
    naming      ② 이름(VLM, 부모는 닫힌 목록) ③ 소그룹 최근접 배정 ④ 저장된 부모 라벨 다수결 검증
    cluster     연사(burst) union-find — 카메라 파티션 ∧ 순서 창 ∧ 코사인 임계
    concept     임베딩 그룹 경계 — 평균연결 계층 클러스터, 적응 임계
    classical   선명도·노출 클립 등 고전 지표
    runners/    ARNIQA·CLIP+LAION 모델 러너 (torch — score 만 쓴다)

wes 잡 mode 는 그대로다: FULL = score → categorize, NAMING = categorize (worker 가 매핑).
실제 폴더(concept_folders/detail_folders)는 wes가 ai_concept_assignments 를 읽어 만든다.
"""
