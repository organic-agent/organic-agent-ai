"""v2 근거 — 사실 → 결정적 템플릿 → (선택) LLM 다듬기.

근거 종류는 넷뿐이고 재료는 전부 검증된 수치다 (docs/plan-v2-slim.md §3.3):

    balance    담은 사진에서 이 유형이 부족  (4단계, subjects_trusted 일 때만)
    quality    기술/미학 백분위 상위 15% + 고전 지표(초점·노출). 형제가 있으면 "연사 N장 중 …"을 덧붙인다
    sibling    같은 연사 N장 중 이 컷 — 나머지가 밀린 이유(덜 선명함·화질·인상·거의 같음)
    concept    같은 배경·구도 N장 중 점수 1위
    score      기술·미학 종합이 평균보다 뚜렷이 높음 (prior_z > 1, 위 넷이 없을 때)
    diversity  위 어느 것도 아님 — 앞서 고른 사진과 배경이 겹치지 않아 뽑힌 컷. 점수 얘기는 하지 않는다

사진에 없는 것(사람·표정·장소 이름·감정)은 재료에 없으므로 LLM 도 말할 수 없다. 과장은 허용,
거짓은 구조적으로 불가 — 그게 이 설계의 핵심이다. 모델이 *무엇을 재는지*는 코드북으로 준다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from photoselect.v2.llm.client import LlmClient

log = logging.getLogger(__name__)

REASON_PRIORITY = ("balance", "quality", "sibling", "concept", "score", "diversity")
MAX_REASON_CHARS = 60

SYSTEM = """\
당신은 신랑신부 옆에서 같이 사진을 고르는 셀렉터다. 이미 고른 사진마다 "왜 이 컷인지"를 한 문장으로 말해
결정을 쉽게 해 준다. 사진은 보지 못한다. 재료는 아래 코드북대로 잰 **수치와 셈**뿐이고, 그것만 쓴다.

코드북 — 우리가 재는 것과 그 한계
- 미학 점수: 구도·색감·피사체 배치에 대한 일반 관람자 선호를 수십만 장의 평가로 학습한 모델(LAION Aesthetic)의
  점수. 갤러리 안 순위(상위 N%)로 준다. **무엇이 좋은지는 모른다** — "삼분할 구도", "따뜻한 색감" 같은 세부는 지어내지 않는다.
- 기술 점수: 초점·흔들림·노이즈·노출 이상을 재는 모델(ARNIQA)의 점수. 갤러리 안 순위로 준다.
- 초점(선명도): 사진에서 직접 잰 디테일 양. "그룹/연사 안에서 가장 또렷"처럼 상대 비교로만 준다.
- 노출: 날아간 밝은 부분·뭉개진 어두운 부분의 비율. "노출이 안정적"까지만.
- 형제: 같은 순간을 연사로 찍은 N장 중 이 컷이 남은 이유와 나머지가 밀린 이유. 선택지를 줄여 주는 말이다.
- 배경 그룹: 같은 배경·구도로 찍은 사진 묶음. 이름은 없다 — "이 배경의 N장 중"이라고만 한다.
- 균형: 담은 사진에서 어떤 유형(신부 단독·신랑 단독·두 분·단체)이 부족한지. 숫자는 준 대로.
- 다양성: 점수는 평범하지만 앞서 고른 사진들과 배경이 겹치지 않아 뽑힌 컷. **점수가 높다고 말하지 않는다.**

쓰는 법
- 문장의 주어는 "주 사유"다. 사실 하나를 더 붙일 수 있다.
- 재료에 없는 숫자·속성·사람·장소·감정을 넣지 않는다. "평생 기억에 남을" 같은 감정 설득 금지.
  설득은 사실(장수·순위·나머지의 이유)로만 한다. 과장은 괜찮지만 없는 것을 말하면 안 된다.
- 40자 안팎, 60자를 넘기지 않는다. "~예요/~해요"체, 마침표 없이. photo_id는 그대로 돌려준다.

예시
- quality 미학 상위 1%·기술 상위 4% + 형제 5장 → "전체 인상 상위 1%에 화질도 상위권, 연사 5장 중 가장 또렷해요"
- sibling, 4장 중, 나머지 덜 선명함 2·거의 같음 1 → "연사 4장 중 초점이 가장 또렷한 컷이에요, 나머지는 살짝 흐려요"
- diversity → "앞서 고른 사진들과 배경이 다른 컷이라 골랐어요"
- quality, 미학 상위 5%·기술 상위 12% → "전체 중 인상이 상위 5%, 화질도 상위권인 컷이에요"
- concept, 37장 중 1위 → "이 배경으로 찍은 37장 중 점수가 가장 높은 컷이에요"
- balance, 신부 단독 2장 부족 → "담으신 사진에 신부 단독 컷이 신랑보다 2장 적어요, 이 컷으로 채워요"\
"""

SCHEMA = {
    "type": "object",
    "properties": {"reasons": {"type": "array", "items": {
        "type": "object",
        "properties": {"photo_id": {"type": "string"}, "reason": {"type": "string"}},
        "required": ["photo_id", "reason"], "additionalProperties": False}}},
    "required": ["reasons"], "additionalProperties": False,
}


@dataclass
class ReasonInput:
    photo_id: str
    primary: str
    facts: list[str] = field(default_factory=list)
    fallback: str = ""


def top_pct(pct: float) -> int:
    return max(1, round(100 - pct))


def _sibling_clause(m: dict) -> str:
    lead = "초점이 가장 또렷한" if m.get("sharpest") else "점수가 가장 높은"
    return f"연사 {m['n']}장 중 {lead} 컷"


def template(primary: str, material: dict) -> str:
    """LLM 없이도 납득되는 한 문장. material 은 draft 가 만든 dict (아래 키만 쓴다)."""
    sib = material.get("sibling")
    if primary == "balance":
        m = material["balance"]
        return f"담으신 사진에 {m['label']} 컷이 부족해요 ({m['sel']}장/{m['total_sel']}장), 이 컷으로 채워요"
    if primary == "quality":
        m = material["quality"]
        parts = []
        if "aesthetic_top" in m:
            parts.append(f"인상 상위 {m['aesthetic_top']}%")
        if "technical_top" in m:
            parts.append(f"화질 상위 {m['technical_top']}%")
        s = "전체 중 " + "·".join(parts)
        if sib:
            return f"{s}, {_sibling_clause(sib)}이에요"
        extra = " · ".join(m.get("descriptors", []))
        return f"{s}인 컷이에요, {extra}" if extra else f"{s}인 컷이에요"
    if primary == "sibling":
        why = " · ".join(f"{k} {v}" for k, v in sib["why_counts"].items())
        return f"{_sibling_clause(sib)}이에요, 나머지는 {why}"
    if primary == "concept":
        m = material["concept"]
        return f"이 배경으로 찍은 {m['size']}장 중 점수가 가장 높은 컷이에요"
    if primary == "score":
        return "기술·미학 점수가 갤러리 평균보다 뚜렷이 높은 컷이에요"
    return "앞서 고른 사진들과 배경·구도가 겹치지 않아 고른 컷이에요"


def facts_of(material: dict) -> tuple[str, list[str]]:
    """(주 사유, 사실 문장들). LLM 에 주는 재료 — 여기 없는 건 LLM 도 모른다."""
    facts: dict[str, str] = {}
    if "balance" in material:
        m = material["balance"]
        facts["balance"] = (f"균형: 담은 {m['total_sel']}장 중 {m['label']} {m['sel']}장 "
                            f"(갤러리 비율대로면 {m['expected']}장) — 부족")
    if "sibling" in material:
        m = material["sibling"]
        why = " · ".join(f"{k} {v}" for k, v in m["why_counts"].items())
        facts["sibling"] = f"형제: 같은 순간 연사 {m['n']}장 중 이 컷. 나머지는 {why}"
    if "quality" in material:
        m = material["quality"]
        q = []
        if "aesthetic_top" in m:
            q.append(f"미학 상위 {m['aesthetic_top']}%")
        if "technical_top" in m:
            q.append(f"기술 상위 {m['technical_top']}%")
        q.extend(m.get("descriptors", []))
        facts["quality"] = "품질: " + " · ".join(q) + " (갤러리 안 순위)"
    if "concept" in material:
        m = material["concept"]
        facts["concept"] = f"배경 그룹: 같은 배경·구도 {m['size']}장 중 점수 1위"
    if material.get("prior_z", 0) > 1.0 and "quality" not in facts:
        facts["score"] = "점수: 기술·미학 종합이 갤러리 평균보다 뚜렷이 높음"
    if not facts:
        facts["diversity"] = "다양성: 앞서 고른 사진들과 배경·구도가 겹치지 않아 뽑힘 (점수는 평범)"
    primary = next((c for c in REASON_PRIORITY if c in facts), "diversity")
    return primary, [facts[c] for c in REASON_PRIORITY if c in facts]


def _material_text(item: ReasonInput) -> str:
    lines = [f"photo_id: {item.photo_id}", f"주 사유: {item.primary}"]
    lines.extend(f"- {f}" for f in item.facts)
    return "\n".join(lines)


def generate(client: LlmClient, items: list[ReasonInput], batch_size: int = 40,
             max_tokens: int = 4096) -> dict[str, str]:
    """photo_id → 문장. 실패는 fallback. 요청에 없는 photo_id·빈 문장·60자 초과는 버린다."""
    out = {it.photo_id: it.fallback for it in items}
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        wanted = {it.photo_id for it in chunk}
        user = "다음 사진들의 추천 이유를 각각 한 문장으로.\n\n" + "\n\n".join(_material_text(it) for it in chunk)
        try:
            data = client.complete_json(SYSTEM, user, SCHEMA, max_tokens)
        except Exception as exc:  # noqa: BLE001
            log.warning("이유 문장 호출 실패 (%d장 템플릿 폴백): %s", len(chunk), exc)
            continue
        for r in data.get("reasons", []):
            pid, text = r.get("photo_id"), (r.get("reason") or "").strip()
            if pid not in wanted or not text or len(text) > MAX_REASON_CHARS:
                continue
            out[pid] = text
    return out
