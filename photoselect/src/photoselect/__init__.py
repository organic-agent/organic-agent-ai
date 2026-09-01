"""photoselect — AI 셀렉터 서버 (organic-agent-ai).

    analyze/   A  갤러리당 1회 전수 분석 (경량 3종 + VLM 태그 + 클러스터)
    draft/     B  초안·재계산 (prior + λ·취향 + 커버리지 + MMR)
    review/       초안 검수 HTML — 로컬 테스트 루프
    llm/       C  (미구현) Bedrock 텍스트 호출

진입점: `python -m photoselect …` (로컬) / `handler.py` (Lambda). 둘 다 같은 job.run()을 부른다.
"""
