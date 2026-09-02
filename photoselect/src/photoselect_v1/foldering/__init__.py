"""① AI 클러스터링 폴더화 — 갤러리당 1회 배치.

    analyze    FULL: 사진별 기술(ARNIQA)·미학(CLIP+LAION)·고전 지표, 연사 클러스터,
               concat(DINOv3⊕CLIP) 임베딩 그룹 → photo_analysis (clip_embedding 저장)
    naming     ② 이름(VLM, 부모는 닫힌 목록) ③ 소그룹 최근접 배정 ④ CLIP zero-shot 부모 검증
               → ai_concept_assignments
    cluster    연사(burst) union-find — 순서 창 ∧ 코사인 임계
    concept    임베딩 그룹 경계 — 평균연결 계층 클러스터, 적응 임계
    classical  선명도·노출 클립 등 고전 지표
    runners/   ARNIQA·CLIP+LAION 모델 러너 (torch — 이 기능만 무겁다)

실제 폴더(photo_folder_*)는 wes가 ai_concept_assignments 를 읽어 만든다.
"""
