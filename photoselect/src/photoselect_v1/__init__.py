"""photoselect-v1 — 확정된 세 기능만 담은 패키지 (기능별 구조).

    foldering/   ① AI 클러스터링 폴더화
                 analyze(FULL: 사진별 점수·연사 클러스터·임베딩 그룹) → naming(VLM 이름·배정·검증)
                 산출물: photo_analysis · ai_concept_assignments. 폴더 생성은 wes(POST /folder-groups/ai).
    recommend/   ② 폴더별 사진 n장 추천 + 상세보기 이유
                 draft(폴더 쿼터·MMR) → reasons(2단계: reason NULL INSERT → LLM UPDATE)
                 산출물: ai_recommendations.
    compare/     ③ 비교샷 — 두 장 중 한 장 AI 판정 (동기, torch 미사용)
                 산출물: ai_pair_verdicts.

공용(루트): config(설정·손잡이) · store(DB/로컬 저장소) · gallery(미리보기 로드) · llm(Bedrock) ·
subjects(피사체 유형) · jobs(잡 전이) · db · storage. 진입점: handler(Lambda) · worker(잡 폴링) · __main__(CLI).

옛 `photoselect` 패키지(v1·v2·v3)에서 v3 확정분만 옮겨 온 것이다 — DB 계약(V45 스키마,
model_version "photoselect-v3-a-0.1", 결과 payload의 pipeline="v3")은 그대로 유지한다.
"""
