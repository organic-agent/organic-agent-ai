"""③ 비교샷 — 두 장 중 한 장 AI 판정 + 이유. 유일한 동기(사용자 대기) 기능.

    verdict.run(store, gallery, settings, photo_a, photo_b, selection_id=…, llm=…)
        사실 수집(DB·PIL) → Sonnet 판정(타임아웃 8s, 재시도 0) → ai_pair_verdicts 저장·응답.
        실패·초과는 템플릿 판정 폴백(source='template'). 캐시는 순서 무관 쌍 + model_version.

torch 를 import 하지 않는다 — Lambda RequestResponse·서브프로세스 시작이 가벼워야 한다.
"""

from photoselect_v1.compare.verdict import run  # noqa: F401 — 기능의 공개 진입점
