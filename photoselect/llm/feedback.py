"""자연어 피드백 → 축 가중치 변경 (plan.md §3-C).

"가족 사진 더", "너무 클로즈업이 많아요" → [{axis, tag, delta}]. 이것만 받는다.
LLM은 사진을 고르지도, 빼지도 못한다 — 선호 분포의 숫자 몇 개를 움직일 뿐이라
틀려도 순위가 조금 흔들릴 뿐 사고가 안 난다 (study/01 step6 §4).

가드레일: axis·tag가 어휘 밖이면 버림, delta는 ±0.5로 클램프, 실패면 빈 목록(👍/👎만 반영).
"""

from __future__ import annotations

import logging

from photoselect.axes import AXES, LABELS_KO
from photoselect.llm.client import LlmClient

log = logging.getLogger(__name__)

MAX_ABS_DELTA = 0.5

SYSTEM = """\
고객이 추천 사진에 남긴 한국어 피드백을 읽고, 아래 축·값 중 무엇을 더 원하는지(+) 덜 원하는지(−)
구조로 바꾼다. 피드백에 근거가 없는 축은 넣지 않는다. 사진을 고르거나 빼는 일은 하지 않는다.

축과 값:
""" + "\n".join(
    f"- {ax}: " + ", ".join(f"{v}({LABELS_KO[ax].get(v) or v})" for v in vals)
    for ax, vals in AXES.items()
) + """

delta는 -0.5 ~ +0.5. "더/많이" 계열은 +0.3, "너무 많다/줄여" 계열은 -0.3을 기본으로,
강조가 세면 ±0.5.\
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "changes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "axis": {"type": "string", "enum": list(AXES)},
                    "tag": {"type": "string"},
                    "delta": {"type": "number"},
                },
                "required": ["axis", "tag", "delta"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["changes"],
    "additionalProperties": False,
}


def translate(client: LlmClient, text: str, max_tokens: int = 512) -> list[tuple[str, str, float]]:
    """(axis, tag, delta) 목록. 어휘 밖·실패는 조용히 버린다."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = client.complete_json(SYSTEM, f"피드백: {text}", SCHEMA, max_tokens)
    except Exception as exc:  # noqa: BLE001
        log.warning("피드백 번역 실패 — 무시: %s", exc)
        return []
    out: list[tuple[str, str, float]] = []
    for c in data.get("changes", []):
        ax, tag = c.get("axis"), c.get("tag")
        if ax not in AXES or tag not in AXES[ax]:
            log.warning("어휘 밖 피드백 무시: %s=%s", ax, tag)
            continue
        try:
            delta = float(c.get("delta", 0.0))
        except (TypeError, ValueError):
            continue
        delta = max(-MAX_ABS_DELTA, min(MAX_ABS_DELTA, delta))
        if delta != 0.0:
            out.append((ax, tag, delta))
    return out
