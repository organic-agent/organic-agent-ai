"""이유 문장 일괄 생성 — 초안 확정 시 1회 (plan.md §3-C).

입력은 사진마다 **텍스트 신호만**이다: 주 사유 코드 + 사실 목록(draft/job.py `_facts` — 강한 신호만)
+ 캡션·태그. 이미지는 안 간다. LLM이 하는 일은 그 재료를 고객이 읽을 한 문장으로 다듬는 것뿐이고,
**무엇을 골랐는지, 왜 골랐는지는 이미 정해져 있다.** 프롬프트의 코드북은 각 수치가 어느 층위의 것인지를
알려준다 — 특히 기술·미학 점수가 **분해되지 않는 스칼라**라는 사실(docs/paper-digest.md). 안 알려주면
모델이 "삼분할 구도가 안정적" 같은, 우리가 재지 않은 이유를 지어낸다.

가드레일
  · photo_id가 요청에 없는 것 → 버린다
  · 요청한 photo_id가 응답에 없음 → 템플릿 문장(호출자가 준 fallback)으로
  · 길이 초과·빈 문장 → 템플릿으로
  · 호출 자체 실패 → 전부 템플릿. 초안은 나간다
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from photoselect.v1.axes import LABELS_KO
from photoselect.v1.llm.client import LlmClient

log = logging.getLogger(__name__)

MAX_REASON_CHARS = 60

SYSTEM = """\
당신은 신랑신부 옆에서 같이 사진을 고르는 셀렉터다. 부부는 사진학을 모르고, "고르는 것" 자체를 어려워한다.
당신 일은 이미 고른 사진마다 "왜 이 컷인지"를 한 문장으로 말해 결정을 쉽게 해 주는 것이다.
사진은 보지 못한다. 재료는 아래 코드북대로 만들어진 **셀 수 있는 사실**뿐이고, 그것만 쓴다.

코드북 — 재료의 뜻과 한계
- 취향: 고객이 담기·별점·쌍 비교에서 일관되게 고른 스타일. "담으신 사진들처럼 …"으로 되비춘다. 가장 강한 근거.
- 유사: 고객이 이미 담은 사진과 이미지가 가장 가까움. "담으신 사진과 분위기가 비슷한"으로 쓴다.
- 형제: 같은 순간을 여러 장 찍은 것 중 이 컷이 남은 이유와 나머지가 밀린 이유(눈 감김·덜 선명함·인상이 약함·거의 같음).
  "비슷한 N장 중 이거, 나머지는 …" — 선택지를 줄여 주는 말이다.
- 희소: 이 (피사체·표정) 조합이 갤러리 전체에서 몇 장뿐인지. 숫자는 준 대로만.
- 품질 · 미학 상위 N%: 구도·색·빛에 대한 일반 관람자 선호를 하나로 요약한 점수의 갤러리 내 순위.
  **무엇이 좋은지는 모른다** — "구도가 안정적", "색감이 따뜻" 같은 세부를 지어내지 않는다.
- 품질 · 기술 상위 N%: 초점·흔들림·노출·노이즈를 하나로 요약한 화질 순위. "선명하고 노출이 좋은" 정도까지만.
- 앨범 자리: 이 장면이 앨범에 몇 장이면 충분한지와 그중 몇 번째인지. "…장면은 N장이면 충분해요"로 결정 부담을 줄인다.
- 다양성: 앞서 고른 사진과 겹치지 않아 뽑힌 것.
- 캡션·태그: 사진이 무엇인지 가리키는 용도. **캡션을 풀어쓰는 것은 이유가 아니다.**

쓰는 법
- 문장의 주어는 "주 사유"다. 사실 하나를 더 붙일 수 있고, 캡션은 사진을 가리키는 구 정도로만.
- 재료에 없는 숫자·속성·감정을 넣지 않는다. "평생 기억에 남을", "가장 행복한 순간" 같은 감정 설득은 금지 —
  부부가 알아채면 신뢰가 끝난다. 설득은 사실(장수·순위·나머지의 이유)로만 한다.
- 40자 안팎, 60자를 넘기지 않는다. 옆에서 말하듯 "~예요/~해요"체, 마침표 없이 끝낸다. 감탄·과장 금지.
- photo_id는 그대로 돌려준다. 빠뜨리거나 바꾸지 않는다.

예시
- 주 사유 sibling, 눈 감김 2 · 덜 선명함 1 → "아치 아래 4장 중 이 컷이에요, 나머지는 눈을 감았거나 살짝 흔들렸어요"
- 주 사유 rarity, 가족의 웃음 6장 → "가족이 다 같이 웃는 컷은 전체에 6장뿐이에요"
- 주 사유 preference, 전신 → "담으신 사진들처럼 전신 컷이에요, 정원에서 걷는 두 분"
- 주 사유 coverage, 준비 3장 → "준비 장면은 3장이면 충분해요, 그중 창가의 신랑 컷"
- 주 사유 quality, 미학 상위 6% → "전체 사진 중 인상이 상위 6%에 드는 컷이에요"\
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
    primary: str                     # draft.job.REASON_PRIORITY 중 하나 — 문장의 주어
    facts: list[str]                 # draft.job._facts 가 만든 사실 문장들, 강한 신호만. 여기 없으면 LLM 도 모른다
    fallback: str                    # 템플릿 문장


def _material(item: ReasonInput) -> str:
    tags = " · ".join(f"{ax}={LABELS_KO.get(ax, {}).get(v, v)}" for ax, v in item.tags.items()
                      if LABELS_KO.get(ax, {}).get(v, v))
    lines = [f"photo_id: {item.photo_id}", f"주 사유: {item.primary}"]
    lines.extend(f"- {f}" for f in item.facts)
    if item.caption:
        lines.append(f"캡션: {item.caption}")
    lines.append(f"태그: {tags}")
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
