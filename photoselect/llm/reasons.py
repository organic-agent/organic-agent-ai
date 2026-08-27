"""이유 문장 일괄 생성 — 초안 확정 시 1회 (plan.md §3-C).

입력은 사진마다 **텍스트 신호만**이다: 캡션·태그·대표 선정 근거·취향 근거·품질 백분위.
이미지는 안 간다. LLM이 하는 일은 그 재료를 고객이 읽을 한 문장으로 다듬는 것뿐이고,
**무엇을 골랐는지는 이미 정해져 있다.**

가드레일
  · photo_id가 요청에 없는 것 → 버린다
  · 요청한 photo_id가 응답에 없음 → 템플릿 문장(호출자가 준 fallback)으로
  · 길이 초과·빈 문장 → 템플릿으로
  · 호출 자체 실패 → 전부 템플릿. 초안은 나간다
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from photoselect.axes import LABELS_KO
from photoselect.llm.client import LlmClient

log = logging.getLogger(__name__)

MAX_REASON_CHARS = 60

SYSTEM = """\
당신은 웨딩 사진 셀렉 도우미다. 이미 고른 사진마다 "왜 이 사진을 추천하는지"를 신랑신부에게
말하듯 한국어 한 문장으로 쓴다.

규칙
- 주어진 재료(캡션·태그·근거)에 있는 것만 쓴다. 사진을 본 적 없으니 추측하지 않는다.
- 취향 근거가 주어진 사진은 그것을 앞세운다 ("전신 컷을 선호하셔서 …").
- 40자 안팎, 60자를 넘기지 않는다. 존댓말, 마침표 없이 끝낸다. 감탄·과장 금지.
- photo_id는 그대로 돌려준다. 빠뜨리거나 바꾸지 않는다.\
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "reasons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"photo_id": {"type": "string"}, "reason": {"type": "string"}},
                "required": ["photo_id", "reason"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["reasons"],
    "additionalProperties": False,
}


@dataclass
class ReasonInput:
    photo_id: str
    caption: str
    tags: dict[str, str]
    rank_reason_code: str
    preference_labels: list[str]     # 이미 정밀도·conf 관문을 통과한 것만 (draft/job.py)
    technical_pct: float
    aesthetic_pct: float
    fallback: str                    # 템플릿 문장


def _material(item: ReasonInput) -> str:
    tags = " · ".join(f"{ax}={LABELS_KO.get(ax, {}).get(v, v)}" for ax, v in item.tags.items()
                      if LABELS_KO.get(ax, {}).get(v, v))
    lines = [f"photo_id: {item.photo_id}"]
    if item.caption:
        lines.append(f"캡션: {item.caption}")
    lines.append(f"태그: {tags}")
    if item.rank_reason_code == "eyes_open":
        lines.append("근거: 같은 순간의 연사 중 눈을 뜬 컷")
    elif item.rank_reason_code in ("technical", "aesthetic"):
        lines.append("근거: 같은 순간의 연사 중 대표 컷")
    if item.preference_labels:
        lines.append(f"취향 근거: 고객이 {'·'.join(item.preference_labels)} 스타일을 선호함")
    lines.append(f"품질: 기술 상위 {100 - item.technical_pct:.0f}% · 미학 상위 {100 - item.aesthetic_pct:.0f}%")
    return "\n".join(lines)


def generate(client: LlmClient, items: list[ReasonInput], batch_size: int = 40,
             max_tokens: int = 4096) -> dict[str, str]:
    """photo_id → 이유 문장. 실패한 것은 fallback으로 채워 **항상 전부 돌려준다.**"""
    out = {it.photo_id: it.fallback for it in items}
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        wanted = {it.photo_id for it in chunk}
        user = "다음 사진들의 추천 이유를 각각 한 문장으로.\n\n" + "\n\n".join(_material(it) for it in chunk)
        try:
            data = client.complete_json(SYSTEM, user, SCHEMA, max_tokens)
        except Exception as exc:  # noqa: BLE001 — 이유 문장 실패로 초안이 죽으면 안 된다
            log.warning("이유 문장 호출 실패 (%d장 템플릿 폴백): %s", len(chunk), exc)
            continue
        got = 0
        for r in data.get("reasons", []):
            pid, text = r.get("photo_id"), (r.get("reason") or "").strip()
            if pid not in wanted:
                log.warning("응답에 요청하지 않은 photo_id — 무시: %s", pid)
                continue
            if not text or len(text) > MAX_REASON_CHARS:
                continue
            out[pid] = text
            got += 1
        if got < len(chunk):
            log.info("이유 문장 %d/%d 수신, 나머지는 템플릿", got, len(chunk))
    return out
