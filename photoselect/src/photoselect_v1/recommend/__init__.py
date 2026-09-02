"""② 폴더별 사진 n장 추천 + 상세보기 이유 — 셀렉당 배치(draft/refine).

    draft      폴더 세트 읽기 → 폴더마다 후보·n_f → 선택 → 2단계 이유 오케스트레이션
    rerank     folder_quota(비례·최소 1·절반 상한) + select_in_folder(연사 대표 1장 → MMR)
    scoring    prior = 0.5·tech_pct + 0.5·aes_pct (+4단계 균형·선호) — 갤러리 내 백분위 그대로
    reasons    사실(수치·셈) + 사진 → LLM "셀렉터의 말". reason NULL INSERT 뒤 UPDATE, 실패 시 템플릿

폴더 세트가 없으면 store.FolderSetMissing → wes 가 409 로 옮긴다. 품질로 사진을 제거하는 단계는 없다.
"""
