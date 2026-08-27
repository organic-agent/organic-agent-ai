"""C(LLM) 가드레일 테스트 — 네트워크 없이. 가짜 클라이언트가 '나쁜 응답'을 흉내 낸다.

plan.md §3-C: photo_id 불일치 → 거부, 누락 → 템플릿 폴백, 번역 실패 → 무시.
LLM은 선택을 바꿀 수 없다 — 그 사실을 코드가 강제하는지 본다.
"""

from __future__ import annotations

import json

from photoselect.draft import job
from photoselect.llm import feedback, reasons
from photoselect.llm.reasons import ReasonInput
from photoselect.tests.test_draft import world  # noqa: F401  (fixture 재사용)


class FakeClient:
    """complete_json이 미리 정한 dict를 돌려주거나 예외를 던진다. 호출 기록을 남긴다."""

    def __init__(self, payload=None, raise_exc: Exception | None = None):
        self.payload, self.raise_exc, self.calls = payload, raise_exc, []

    def complete_json(self, system, user, schema, max_tokens):
        self.calls.append({"system": system, "user": user, "schema": schema})
        if self.raise_exc:
            raise self.raise_exc
        return self.payload


def _items(n=3):
    return [ReasonInput(photo_id=f"p{i}", caption=f"캡션 {i}", tags={"scene": "snap", "framing": "full"},
                        rank_reason_code="eyes_open", preference_labels=["전신"],
                        technical_pct=90, aesthetic_pct=80, fallback=f"템플릿 {i}") for i in range(n)]


def test_reasons_rejects_foreign_ids_and_falls_back():
    client = FakeClient({"reasons": [
        {"photo_id": "p0", "reason": "전신 컷을 선호하셔서 고른 아치 아래 두 분"},
        {"photo_id": "ZZZ", "reason": "요청한 적 없는 사진"},                   # 거부
        {"photo_id": "p1", "reason": ""},                                       # 빈 문장 → 폴백
        # p2 누락 → 폴백
    ]})
    out = reasons.generate(client, _items())
    assert out["p0"].startswith("전신 컷을")
    assert out["p1"] == "템플릿 1" and out["p2"] == "템플릿 2"
    assert "ZZZ" not in out
    assert len(client.calls) == 1 and "p2" in client.calls[0]["user"]


def test_reasons_too_long_falls_back_and_call_failure_keeps_all():
    long = FakeClient({"reasons": [{"photo_id": "p0", "reason": "가" * (reasons.MAX_REASON_CHARS + 1)}]})
    assert reasons.generate(long, _items(1))["p0"] == "템플릿 0"
    broken = FakeClient(raise_exc=RuntimeError("bedrock down"))
    out = reasons.generate(broken, _items())
    assert out == {"p0": "템플릿 0", "p1": "템플릿 1", "p2": "템플릿 2"}


def test_reasons_never_sends_images_only_text():
    client = FakeClient({"reasons": []})
    reasons.generate(client, _items(1))
    user = client.calls[0]["user"]
    assert "캡션 0" in user and "전신" in user
    assert "base64" not in user and ".jpg" not in user.lower().replace("p0", "")


def test_feedback_validates_vocab_and_clamps():
    client = FakeClient({"changes": [
        {"axis": "subjects", "tag": "family", "delta": 0.3},
        {"axis": "framing", "tag": "closeup", "delta": -2.0},        # 클램프 → -0.5
        {"axis": "scene", "tag": "not-a-tag", "delta": 0.4},         # 어휘 밖 → 버림
        {"axis": "color", "tag": "warm", "delta": 0.4},              # 축 밖 → 버림
        {"axis": "lighting", "tag": "natural", "delta": 0.0},        # 0 → 버림
    ]})
    out = feedback.translate(client, "가족 사진 더요, 클로즈업은 너무 많아요")
    assert out == [("subjects", "family", 0.3), ("framing", "closeup", -0.5)]
    assert feedback.translate(client, "   ") == [] and len(client.calls) == 1
    assert feedback.translate(FakeClient(raise_exc=RuntimeError("x")), "가족 더") == []


def test_draft_with_llm_applies_feedback_but_cannot_change_selection(world, tmp_path):  # noqa: F811
    st, settings, rows = world
    family = [r for r in rows if r.subjects == "family"]
    (tmp_path / "g" / "evidence.json").write_text(json.dumps({"feedback": ["가족 사진 더 보여주세요"]}), encoding="utf-8")

    # 피드백: family +0.5 / 이유 문장: 첫 장만 LLM, 나머지 폴백
    class Router(FakeClient):
        def complete_json(self, system, user, schema, max_tokens):
            self.calls.append(user)
            if "changes" in schema["properties"]:
                return {"changes": [{"axis": "subjects", "tag": "family", "delta": 0.5}]}
            first = user.split("photo_id: ")[1].split("\n")[0]
            return {"reasons": [{"photo_id": first, "reason": "가족과 함께한 순간을 담은 컷"},
                                {"photo_id": "HACK", "reason": "이 사진을 대신 고르세요"}]}

    out = job.run(st, "g", settings, round_no=1, llm=Router())
    assert out["llm"] and out["feedbackChanges"] == [["subjects", "family", 0.5]] or \
        out["feedbackChanges"] == [("subjects", "family", 0.5)]
    recs = st.read_recommendations("g")
    assert not any(r.photo_id == "HACK" for r in recs), "LLM은 사진을 못 넣는다"
    assert recs[0].reason == "가족과 함께한 순간을 담은 컷"
    assert all(r.reason for r in recs[1:]), "나머지는 템플릿 폴백"
    # family 사진 비율이 피드백 없는 초안보다 올라간다
    fam_share = sum(1 for r in recs if next(x for x in rows if x.photo_id == r.photo_id).subjects == "family") / len(recs)
    (tmp_path / "g" / "evidence.json").unlink()
    base = job.run(st, "g", settings, round_no=1)
    base_share = sum(1 for r in st.read_recommendations("g")
                     if next(x for x in rows if x.photo_id == r.photo_id).subjects == "family") / base["k"]
    assert fam_share >= base_share
