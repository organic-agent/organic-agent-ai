"""비교샷 — 두 사진 중 AI가 하나를 고르고 이유를 쓴다 (docs/plan-v3-folder-compare.md §3).

첫 대화형(동기) 워크로드다. 사용자가 답을 기다리므로:
  · **torch 를 import 하지 않는다** — DB + PIL + Bedrock 만. 서브프로세스/Lambda 시작이 가벼워야 한다.
  · LLM 은 타임아웃(compare_timeout_s)·재시도 없음으로 부르고, 실패하면 **템플릿 판정**으로
    응답한다(source='template') — "AI가 못 골랐어요"가 아니라 "기준으로만 골랐어요".
  · 같은 (selection, a, b) 재요청은 캐시 반환 — 순서 무관, 모델·프롬프트 세대(model_version)가
    바뀌면 다시 판정해 덮는다.

판정 원칙(§3.1): **항상 하나를 고른다.** 거의 같으면 confidence='slight'. 사진에서 확인되는
차이(초점·눈·시선·표정)를 우선하고, 백분위 차는 5pt 이상일 때만 근거로 쓴다. 흑백/컬러·구도
같은 취향 문제는 "두 분이 정하실 몫"이라고 말하되 선택은 한다.

사람의 답(pair_comparison_events)과는 분리 저장 — 누가 골랐는지가 흐려지면 안 된다.
"""

from __future__ import annotations

import logging
import time

from photoselect_v1.config import Settings
from photoselect_v1.llm import LlmClient, Part, jpeg_bytes
from photoselect_v1.store import PairVerdict, Store

log = logging.getLogger(__name__)

#: 프롬프트·사실 수집 규칙의 세대. 바꾸면 캐시가 무효화되도록 model_version 에 들어간다.
PROMPT_VERSION = "compare-p2"
#: 이유 문장 상한 — 상세 카드 하나에 들어갈 길이.
MAX_REASON_CHARS = 400
#: 선명도 비가 이보다 커야 "더 또렷"이라고 말한다 (draft._why_not 과 같은 기준).
SHARPNESS_RATIO = 1.25
#: 백분위 차가 이보다 작으면 근거로 쓰지 않는다 (§3.1).
PCT_GAP = 5.0

SYSTEM = """\
당신은 신랑신부 옆에 앉아 함께 사진을 고르는 베테랑 셀렉터다. 두 분이 사진 두 장을 놓고
고민하다가 당신에게 "어느 쪽이 나아요?"라고 물었다. 당신의 일은 **반드시 한 장을 고르고**,
왜 그 컷인지 두 분이 납득할 말로 설명하는 것이다.

규칙
- 반드시 a 또는 b 하나를 고른다. 기권은 없다. 두 장이 거의 같으면 confidence 를 "slight" 로
  하고 "거의 같아요, 굳이 고르면"의 톤으로 말한다. 차이가 분명하면 "clear".
- 판단 순서: ① 사진에서 직접 확인되는 차이 — 초점, 눈 감김, 시선, 표정, 흔들림, 잘림.
  ② 재료의 수치 — 백분위 차는 재료에 적힌 것만, 5포인트 이상일 때만 근거로 쓴다.
  ③ 흑백/컬러, 구도 취향, 분위기는 **취향**이다 — "이건 두 분이 정하실 몫"이라고 말하되,
  그래도 선택은 한다.
- 숫자·순위·장수는 재료에 적힌 것만 쓴다. 없는 숫자를 만들지 않는다.
- 말투: "~입니다/~요" 구어체, 부부에게 직접. 탈락한 쪽도 깎아내리지 않는다 — "저 컷도 좋지만"
  으로 시작해도 좋다. 모든 표현은 한국어로 쓴다.
- a/b 는 내부 라벨이다 — **문장에는 절대 쓰지 않는다.** 고른 쪽은 "이 컷", 다른 쪽은
  "다른 컷"이라고 부른다. 두 분 화면에는 a/b 표시가 없다.
- 길이: 2~4문장, 300자 안.

응답은 {chosen: "a"|"b", confidence: "clear"|"slight", reason} 하나다.\
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "chosen": {"type": "string", "enum": ["a", "b"]},
        "confidence": {"type": "string", "enum": ["clear", "slight"]},
        "reason": {"type": "string"},
    },
    "required": ["chosen", "confidence", "reason"],
    "additionalProperties": False,
}


def collect_facts(ra, rb, folder_a: str | None, folder_b: str | None,
                  selected: set[str]) -> tuple[list[str], dict]:
    """(LLM 재료 문장들, 저장용 dict). 숫자는 전부 여기서 나온다 — LLM 은 이것만 안다."""
    facts: list[str] = []
    d: dict = {}

    sa, sb = ra.sub_scores.get("sharpness") or 0.0, rb.sub_scores.get("sharpness") or 0.0
    if sa > 0 and sb > 0:
        ratio = sa / sb if sa >= sb else sb / sa
        if ratio >= SHARPNESS_RATIO:
            lead = "a" if sa >= sb else "b"
            facts.append(f"초점: 사진 {lead} 가 더 또렷하다 (선명도 {ratio:.1f}배)")
            d["sharper"] = {"photo": lead, "ratio": round(ratio, 2)}

    for name, key in (("기술(화질)", "technical_pct"), ("미학(인상)", "aesthetic_pct")):
        va, vb = getattr(ra, key), getattr(rb, key)
        gap = va - vb
        if abs(gap) >= PCT_GAP:
            lead = "a" if gap > 0 else "b"
            facts.append(f"{name} 백분위: 사진 {lead} 가 {abs(gap):.0f}포인트 높다 "
                         f"(a {va:.0f} / b {vb:.0f}, 갤러리 안 순위)")
            d[key] = {"a": round(va, 1), "b": round(vb, 1)}

    for label, row in (("a", ra), ("b", rb)):
        if row.sub_scores.get("highlight_clip", 0.0) >= 0.01:
            facts.append(f"노출: 사진 {label} 는 밝은 부분이 일부 날아갔다")
            d.setdefault("exposure", {})[label] = "highlight_clip"

    if ra.cluster_id == rb.cluster_id and ra.cluster_id >= 0:
        best = "a" if ra.cluster_rank <= rb.cluster_rank else "b"
        facts.append(f"연사: 같은 순간에 찍힌 연속 촬영 컷이다. 점수 기준 대표는 사진 {best}")
        d["same_burst"] = {"best": best}

    if folder_a or folder_b:
        if folder_a == folder_b:
            facts.append(f"폴더: 두 장 다 \"{folder_a}\" 폴더다")
        else:
            facts.append(f"폴더: 사진 a 는 \"{folder_a or '미분류'}\", 사진 b 는 \"{folder_b or '미분류'}\"")
        d["folders"] = {"a": folder_a, "b": folder_b}

    if ra.subjects != rb.subjects and "unknown" not in (ra.subjects, rb.subjects):
        facts.append(f"유형: 사진 a 는 {ra.subjects}, 사진 b 는 {rb.subjects} — 담는 용도가 다를 수 있다")
        d["subjects"] = {"a": ra.subjects, "b": rb.subjects}

    for label, row in (("a", ra), ("b", rb)):
        if row.photo_id in selected:
            facts.append(f"참고: 사진 {label} 는 이미 담은 사진이다")
            d.setdefault("already_selected", []).append(label)

    if not facts:
        facts.append("측정된 차이 없음 — 두 장의 수치가 거의 같다. 사진에서 보이는 것으로 판단하라")
    return facts, d


def template_verdict(ra, rb) -> tuple[str, str]:
    """LLM 없이 사실만으로 고르는 결정적 판정 — 초점 ▸ 화질 ▸ 미학 ▸ a (§3.2)."""
    sa, sb = ra.sub_scores.get("sharpness") or 0.0, rb.sub_scores.get("sharpness") or 0.0
    if sa > 0 and sb > 0 and max(sa, sb) / max(min(sa, sb), 1e-9) >= SHARPNESS_RATIO:
        c = "a" if sa >= sb else "b"
        return c, "초점이 더 또렷한 쪽을 골랐어요. 눈으로도 한번 비교해 보세요"
    if abs(ra.technical_pct - rb.technical_pct) >= PCT_GAP:
        c = "a" if ra.technical_pct >= rb.technical_pct else "b"
        return c, "화질 점수가 더 높은 쪽을 골랐어요"
    if abs(ra.aesthetic_pct - rb.aesthetic_pct) >= PCT_GAP:
        c = "a" if ra.aesthetic_pct >= rb.aesthetic_pct else "b"
        return c, "전체 인상 점수가 더 높은 쪽을 골랐어요"
    return "a", "두 장의 측정치가 거의 같아요 — 기준상 앞의 사진을 골랐지만 두 분 취향이 정답이에요"


def run(store: Store, gallery: str, settings: Settings, photo_a: str, photo_b: str,
        selection_id: str | None = None, llm: LlmClient | None = None) -> dict:
    started = time.monotonic()
    photo_a, photo_b = str(photo_a), str(photo_b)
    if photo_a == photo_b:
        raise SystemExit("같은 사진 두 장은 비교할 수 없다")
    model_version = f"{settings.llm.model_id}+{PROMPT_VERSION}"

    cached = store.read_pair_verdict(gallery, photo_a, photo_b, model_version)
    if cached is not None:
        return _response(cached, cached=True, started=started)

    by_id = {r.photo_id: r for r in store.read_analysis(gallery)
             if r.photo_id in (photo_a, photo_b)}
    if photo_a not in by_id or photo_b not in by_id:
        missing = [p for p in (photo_a, photo_b) if p not in by_id]
        raise SystemExit(f"분석되지 않은 사진: {missing} — FULL 분석이 먼저다")
    ra, rb = by_id[photo_a], by_id[photo_b]

    folders = store.folder_names(gallery, [photo_a, photo_b])
    selected = set(store.read_evidence(gallery, selection_id).selected)
    facts, facts_d = collect_facts(ra, rb, folders.get(photo_a), folders.get(photo_b), selected)

    chosen, reason = template_verdict(ra, rb)
    confidence, source = "slight", "template"

    if llm is not None:
        try:
            parts: list[Part] = [("text", "두 장 중 어느 쪽을 담을지 골라 주세요.\n사진 a:")]
            parts.append(("image", _image(store, gallery, photo_a, settings)))
            parts.append(("text", "사진 b:"))
            parts.append(("image", _image(store, gallery, photo_b, settings)))
            parts.append(("text", "재료(측정된 사실):\n" + "\n".join(f"- {f}" for f in facts)))
            out = llm.complete_json(SYSTEM, parts, SCHEMA, settings.llm.compare_max_tokens)
            c, conf, text = out.get("chosen"), out.get("confidence"), (out.get("reason") or "").strip()
            if c in ("a", "b") and conf in ("clear", "slight") and text:
                chosen, confidence, source = c, conf, "llm"
                reason = text[:MAX_REASON_CHARS]
            else:
                log.warning("compare LLM 응답이 계약을 벗어남 (%r) — 템플릿 판정", out)
        except Exception as exc:  # noqa: BLE001 — 타임아웃 포함. 동기 경로는 템플릿으로 응답한다
            log.warning("compare LLM 실패 → 템플릿 판정: %s", exc)

    verdict = PairVerdict(
        photo_a=photo_a, photo_b=photo_b,
        chosen_photo_id=photo_a if chosen == "a" else photo_b,
        confidence=confidence, reason=reason, facts=facts_d,
        model_version=model_version, source=source,
    )
    store.write_pair_verdict(gallery, verdict)
    return _response(verdict, cached=False, started=started)


def _image(store: Store, gallery: str, photo_id: str, settings: Settings) -> bytes:
    path = store.preview_path(gallery, photo_id)
    if path is None:
        raise RuntimeError(f"미리보기가 없다: {photo_id}")
    return jpeg_bytes(path, settings.llm.compare_image_long_edge)


def _response(v: PairVerdict, *, cached: bool, started: float) -> dict:
    return {
        "pipeline": "v3", "mode": "compare",
        "photoA": v.photo_a, "photoB": v.photo_b,
        "chosenPhotoId": v.chosen_photo_id, "confidence": v.confidence,
        "reason": v.reason, "source": v.source, "cached": cached,
        "elapsedSeconds": round(time.monotonic() - started, 2),
    }
