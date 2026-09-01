"""v3 근거 — 사실(수치·셈) + 사진 → LLM 이 "셀렉터의 말"로. 실패하면 결정적 템플릿.

v2 reasons 의 폴더판 (docs/plan-v3-folder-compare.md §2.3). 달라진 것:
  · 재료에 **폴더 사실** 한 줄이 추가된다 — `폴더: "{부모} › {컨셉}" N장 중 M위 (추천 n장)`.
    폴더 이름은 naming 잡(VLM)이나 사용자가 붙인 것이라 문장에 써도 된다 — 임베딩 그룹과 달리
    이름이 있다. needs_review 폴더라도 그 말은 재료에 넣지 않는다(추천 이유와 무관) — 배지로만.
  · concept(이름 없는 배경 그룹) 재료는 폴더가 대신한다.
  · 이유는 2단계다 — draft 가 reason NULL 로 먼저 적재하고, 여기서 만든 문장으로 UPDATE 한다.

재료 두 층(수치·셈 + 사진)과 원칙(숫자는 재료에 있는 것만, 눈에 보이는 것은 사진에서 본 것만)은
v2 와 같다. 이름은 반드시 한국어로 쓴다.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from photoselect.v3.llm.client import LlmClient, Part, jpeg_bytes

log = logging.getLogger(__name__)

REASON_PRIORITY = ("balance", "quality", "sibling", "folder", "score", "diversity")
#: 사진을 보고 쓰는 문장의 상한. 템플릿·텍스트 전용 문장은 이보다 훨씬 짧다.
MAX_REASON_CHARS = 600
#: 사진 없이(텍스트 재료만) 만든 문장의 상한.
MAX_TEXT_ONLY_CHARS = 60

SYSTEM = """\
당신은 신랑신부 옆에 앉아 함께 사진을 고르는 베테랑 셀렉터다. 두 분은 고르기를 어려워하고, 당신의 일은
"이 컷이면 됩니다"라고 확신을 실어 주는 것이다. 이미 뽑힌 사진마다 **왜 이 컷인지** 말한다.

말투 — 아낌없이 칭찬하고 설득한다
- 두 분을 한껏 띄운다. "조명이 정말 예술입니다", "이 시선이 제일 좋았습니다"처럼 감탄을 앞세운다.
- 그러나 근거 없는 감탄은 아니다. 감탄 뒤에는 반드시 **사진에서 보이는 것** 또는 **수치**가 따라온다.
- "~입니다/~요"를 섞은 구어체. 부부에게 직접 말하듯. 사진 id 같은 내부 표기는 문장에 넣지 않는다.
- 모든 표현은 한국어로 쓴다.

재료 두 가지
1. 사진 — 이 컷과, 있으면 같은 순간에 찍힌 형제 컷 몇 장. 조명(역광·림라이트·그림자), 구도와 레이어(앞·가운데·뒤),
   시선과 표정, 자세의 대칭, 색감(드레스·부케 톤), 흑백/컬러 같은 **눈에 보이는 것**은 직접 보고 말한다.
   형제 컷이 있으면 "저 컷은 시선이 빗나갔고, 이 컷은 마주쳤어요"처럼 **비교로** 설득한다 — 이게 가장 세다.
2. 수치·셈 — 아래 코드북대로 잰 것. **숫자·순위·장수는 여기 적힌 것만 쓴다.** 없는 숫자를 만들지 않는다.

코드북 — 우리가 재는 것과 그 한계
- 미학 점수: 구도·색감·피사체 배치에 대한 일반 관람자 선호를 학습한 모델(LAION Aesthetic)의 점수, 갤러리 안 순위(상위 N%).
- 기술 점수: 초점·흔들림·노이즈·노출 이상을 재는 모델(ARNIQA)의 점수, 갤러리 안 순위.
- 초점(선명도): 사진에서 직접 잰 디테일 양. "연사 안에서 가장 또렷"처럼 상대 비교.
- 노출: 날아간 밝은 부분·뭉개진 어두운 부분의 비율. "노출이 안정적"까지만.
- 형제: 같은 순간 연사 N장 중 이 컷이 남은 이유와 나머지가 밀린 이유(덜 선명함·화질·인상·거의 같음).
- 폴더: 같은 배경·컨셉으로 묶인 폴더의 이름과 그 안에서의 순위. 폴더 이름은 문장에 그대로 써도 된다.
- 균형: 담은 사진에서 어떤 유형(신부 단독·신랑 단독·두 분·단체)이 부족한지. 숫자는 준 대로.
- 다양성: 점수는 평범하지만 앞서 고른 사진들과 배경이 겹치지 않아 뽑힌 컷. **점수가 높다고 말하지 않는다.**

쓰는 법
- 주 사유가 뼈대다. 그 위에 사진에서 본 것 2~3가지를 얹어 살을 붙인다. 형제 컷이 있으면 비교를 한 문단 넣는다.
- 사진이 온 컷: 3~5문장, 200~400자. 문단은 최대 둘.
- 사진이 안 온 컷("(사진 없음)"): 재료만으로 한 문장, 60자 안.
- 사람 이름·장소 이름·촬영 상황(누가 왜)처럼 사진에도 재료에도 없는 것은 지어내지 않는다. 과장은 좋다, 거짓은 안 된다.
- photo_id는 받은 그대로 돌려준다. 문장 안에는 넣지 않는다.

예시 (사진이 온 컷, 형제 2장과 비교)
"일단 조명이 정말 예술입니다. 역광인데도 두 분 얼굴에 그림자가 안 지고 림라이트가 살아 있어요 — 타이밍이 완벽했다는
뜻이거든요. 앞쪽 부케, 가운데 두 분, 뒤쪽 조명 줄까지 레이어가 겹겹이 쌓여서 깊이감이 확 삽니다. 무엇보다 두 분 시선이
진짜로 마주치고 있어요, 그게 제일 좋았습니다. 같은 순간의 나머지 두 장은 한 장은 살짝 흐리고, 한 장은 흑백이라 드레스의
파스텔 블루가 안 보여서 아까웠어요. 연사 3장 중 초점도 이 컷이 가장 또렷합니다."

예시 (사진 없음)
- quality 미학 상위 1%·기술 상위 4% + 형제 5장 → "전체 인상 상위 1%에 화질도 상위권, 연사 5장 중 가장 또렷해요"
- folder "야외 자연 › 해변 모래사장" 41장 중 1위 → "해변 모래사장 폴더 41장 중 점수가 가장 높은 컷이에요"\
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
class SiblingImage:
    photo_id: str
    why_not: str
    path: str


@dataclass
class ReasonInput:
    photo_id: str
    primary: str
    facts: list[str] = field(default_factory=list)
    fallback: str = ""
    #: 이 컷의 이미지 경로. None 이면 텍스트 재료만 보낸다.
    image_path: str | None = None
    #: 비교용 형제 컷 (draft 가 why_not 순으로 몇 장 골라 준다).
    siblings: list[SiblingImage] = field(default_factory=list)


def top_pct(pct: float) -> int:
    return max(1, round(100 - pct))


def _sibling_clause(m: dict) -> str:
    lead = "초점이 가장 또렷한" if m.get("sharpest") else "점수가 가장 높은"
    return f"연사 {m['n']}장 중 {lead} 컷"


def _folder_clause(m: dict) -> str:
    return f"\"{m['parent']} › {m['name']}\" 폴더 {m['size']}장 중 {m['rank']}위"


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
    if primary == "folder":
        return f"{_folder_clause(material['folder'])}인 컷이에요"
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
    folder_top = material.get("folder", {}).get("rank") == 1
    if "folder" in material:
        m = material["folder"]
        facts["folder"] = f"폴더: {_folder_clause(m)} (추천 {m['quota']}장)"
    if material.get("prior_z", 0) > 1.0 and "quality" not in facts:
        facts["score"] = "점수: 기술·미학 종합이 갤러리 평균보다 뚜렷이 높음"
    # 주 사유 후보: folder 는 폴더 1위일 때만. 전부 탈락하면 diversity(MMR 이 끌어올린 컷).
    eligible = {c for c in facts if c != "folder" or folder_top}
    primary = next((c for c in REASON_PRIORITY if c in eligible), "diversity")
    if primary == "diversity":
        facts["diversity"] = "다양성: 앞서 고른 사진들과 배경·구도가 겹치지 않아 뽑힘 (점수는 평범)"
    return primary, [facts[c] for c in REASON_PRIORITY if c in facts]


def _parts(item: ReasonInput, long_edge: int) -> list[Part]:
    """한 사진의 user 조각들: 헤더 텍스트 → (이미지) → 형제들 → 사실 목록."""
    parts: list[Part] = [("text", f"### photo_id: {item.photo_id}\n주 사유: {item.primary}")]
    if item.image_path:
        try:
            parts.append(("text", "이 컷:"))
            parts.append(("image", jpeg_bytes(item.image_path, long_edge)))
        except Exception as exc:  # noqa: BLE001 — 사진 하나 못 읽었다고 라운드를 죽이지 않는다
            log.warning("이미지 읽기 실패 %s: %s", item.image_path, exc)
            parts = parts[:1] + [("text", "(사진 없음)")]
        else:
            for sib in item.siblings:
                try:
                    data = jpeg_bytes(sib.path, long_edge)
                except Exception as exc:  # noqa: BLE001
                    log.warning("형제 이미지 읽기 실패 %s: %s", sib.path, exc)
                    continue
                parts.append(("text", f"같은 순간의 다른 컷 (탈락, 이유: {sib.why_not}):"))
                parts.append(("image", data))
    else:
        parts.append(("text", "(사진 없음)"))
    parts.append(("text", "재료:\n" + "\n".join(f"- {f}" for f in item.facts)))
    return parts


def _user(chunk: list[ReasonInput], vision: bool, long_edge: int) -> str | list[Part]:
    intro = "다음 사진들의 추천 이유를 각각 써 주세요. photo_id 는 그대로 돌려주세요.\n"
    if not vision:
        lines = []
        for it in chunk:
            lines.append(f"photo_id: {it.photo_id}\n주 사유: {it.primary}\n(사진 없음)\n" + "\n".join(f"- {f}" for f in it.facts))
        return intro + "\n" + "\n\n".join(lines)
    parts: list[Part] = [("text", intro)]
    for it in chunk:
        parts.extend(_parts(it, long_edge))
    return parts


def generate(client: LlmClient, items: list[ReasonInput], batch_size: int = 10,
             max_tokens: int = 8192, vision: bool = True, image_long_edge: int = 768) -> dict[str, str]:
    """photo_id → 문장. 실패는 fallback. 요청에 없는 photo_id·빈 문장·상한 초과는 버린다.

    상한은 사진을 보냈는지에 따라 다르다: 사진을 본 컷은 MAX_REASON_CHARS, 텍스트만 준 컷은 MAX_TEXT_ONLY_CHARS.
    """
    out = {it.photo_id: it.fallback for it in items}
    for start in range(0, len(items), batch_size):
        chunk = items[start:start + batch_size]
        by_id = {it.photo_id: it for it in chunk}
        try:
            data = client.complete_json(SYSTEM, _user(chunk, vision, image_long_edge), SCHEMA, max_tokens)
        except Exception as exc:  # noqa: BLE001
            log.warning("이유 문장 호출 실패 (%d장 템플릿 폴백): %s", len(chunk), exc)
            continue
        for r in data.get("reasons", []):
            pid, text = r.get("photo_id"), (r.get("reason") or "").strip()
            it = by_id.get(pid)
            if it is None or not text:
                continue
            limit = MAX_REASON_CHARS if (vision and it.image_path) else MAX_TEXT_ONLY_CHARS
            if len(text) > limit:
                log.info("이유 문장 상한 초과 → 템플릿 (%s, %d자)", pid, len(text))
                continue
            out[pid] = text
    return out
