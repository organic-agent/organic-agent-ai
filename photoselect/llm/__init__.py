"""C — Bedrock 텍스트 호출. 이미지는 절대 보내지 않는다 (CLAUDE.md 데이터 원칙).

    reasons.py    초안 확정 시 이유 문장 1회 일괄 (plan.md §3-C)
    feedback.py   자연어 피드백 → {axis, tag, delta} 번역
    client.py     AnthropicBedrockMantle 래퍼 + structured output

가드레일 (plan.md §3-C): LLM은 선택을 바꿀 수 없다. photo_id가 요청과 다르면 거부, 빠지면 템플릿
폴백, 번역 실패면 무시. 전부 코드에서 강제한다 — 프롬프트로 부탁하지 않는다.
"""
