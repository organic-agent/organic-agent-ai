"""photoselect-v1 — AI 클러스터링 폴더화 배치 (기능별 구조).

    foldering/   score(SCORE: 사진별 점수·CLIP 벡터·라벨, torch) → categorize(CATEGORIZE: 백분위·연사·
                 임베딩 그룹 → naming, torch 없음). wes mode FULL = 둘 다, NAMING = categorize (#26).
                 산출물: photo_analysis · ai_concept_assignments. 폴더 생성은 wes(POST /concept-folders/ai).

공용(루트): config(설정·손잡이) · store(DB/로컬 저장소) · gallery(미리보기 로드) · llm(Bedrock) ·
subjects(피사체 유형) · jobs(잡 전이) · db · storage. 진입점: worker(잡 폴링) · __main__(CLI).

폴더별 추천과 비교샷은 wes로 이관됐다(#25, wes `docs/plans/ai-feature-migration-to-wes.md`) —
경계는 "갤러리 전수에 torch 모델 추론이 필요한가"다. 저장된 숫자를 셈하고 미리보기 몇 장을
Bedrock에 보여 주는 일은 wes가 한다. DB 계약(V45 스키마, model_version "photoselect-v3-a-0.1",
결과 payload의 pipeline="v3")은 그대로다.
"""
