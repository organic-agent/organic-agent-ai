"""C(LLM) 가드레일 테스트 — 네트워크 없이. 가짜 클라이언트가 '나쁜 응답'을 흉내 낸다.

plan.md §3-C: photo_id 불일치 → 거부, 누락 → 템플릿 폴백, 번역 실패 → 무시.
LLM은 선택을 바꿀 수 없다 — 그 사실을 코드가 강제하는지 본다.
"""

from __future__ import annotations

import json

from photoselect.v1.draft import job
from photoselect.v1.llm import feedback, reasons
from photoselect.v1.llm.reasons import ReasonInput
from tests.v1.test_draft import world  # noqa: F401  (fixture 재사용)


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
                        primary="preference",
                        facts=["취향: 고객이 담기·비교에서 전신 스타일을 일관되게 선호",
                               "연사: 같은 순간을 여러 장 찍은 것 중 눈을 뜬 컷"],
                        fallback=f"템플릿 {i}") for i in range(n)]


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


def test_material_carries_only_strong_signals_and_codebook(world):  # noqa: F811
    """이유 문장 재료: 약한 백분위(상위 68%)는 재료에 없고, 강한 것(상위 6%)만 있다. 코드북이 프롬프트에 있다."""
    from photoselect.v1.draft.job import REASON_PRIORITY
    st, settings, rows = world
    client = FakeClient({"reasons": []})
    job.run(st, "g", settings, round_no=1, llm=client)
    system, user = client.calls[0]["system"], client.calls[0]["user"]
    assert "코드북" in system and "스칼라" not in user
    assert "지어내지 않는다" in system
    top = 100 - settings.score.reason_quality_top_pct
    for r in st.read_recommendations("g"):
        row = next(x for x in rows if x.photo_id == r.photo_id)
        block = user.split(f"photo_id: {r.photo_id}\n")[1].split("\n\n")[0]
        assert block.startswith("주 사유: ") and block.split("\n")[0][len("주 사유: "):] in REASON_PRIORITY
        aes = f"미학 상위 {max(1, round(100 - row.aesthetic_pct))}%"
        assert (aes in block) == (row.aesthetic_pct >= top), (r.photo_id, row.aesthetic_pct, block)
        tech = f"기술 상위 {max(1, round(100 - row.technical_pct))}%"
        assert (tech in block) == (row.technical_pct >= top)
        assert r.score_breakdown["primary_reason"] == block.split("\n")[0][len("주 사유: "):]
        assert r.score_breakdown["slot"] in ("quota", "diversity", "fill")


def test_template_reason_leads_with_primary_and_never_cites_weak_quality(world):  # noqa: F811
    st, settings, rows = world
    job.run(st, "g", settings, round_no=1)
    top = 100 - settings.score.reason_quality_top_pct
    for r in st.read_recommendations("g"):
        row = next(x for x in rows if x.photo_id == r.photo_id)
        assert r.reason and "추천" != r.reason
        if row.aesthetic_pct < top and row.technical_pct < top:
            assert "상위" not in r.reason, r.reason
        if r.score_breakdown["primary_reason"] == "quality":
            assert r.reason.startswith(("미학 상위", "기술 상위"))
        if r.score_breakdown["primary_reason"] == "coverage":
            assert "장이면 충분해요" in r.reason


def test_phase2_similar_to_selected_becomes_a_fact(world, tmp_path):  # noqa: F811
    """담은 사진과 같은 클러스터(닮은 임베딩)의 컷이 후보에 남아 있으면 '유사' 근거가 붙는다."""
    st, settings, rows = world
    # 클러스터 크기 ≥2 인 것 하나: 첫 장을 담고, 나머지가 후보에 남는다
    by_cluster: dict[int, list] = {}
    for r in rows:
        by_cluster.setdefault(r.cluster_id, []).append(r)
    group = next(g for g in by_cluster.values() if len(g) >= 2)
    (tmp_path / "g" / "evidence.json").write_text(json.dumps({"selected": [group[0].photo_id]}), encoding="utf-8")
    job.run(st, "g", settings, round_no=1, top_k=len(rows), target=len(rows))   # 전부 제시 → 형제 컷 포함
    sibling = next(r for r in st.read_recommendations("g") if r.photo_id in {x.photo_id for x in group[1:]})
    assert sibling.score_breakdown["similar_to"] == group[0].photo_id
    assert sibling.score_breakdown["primary_reason"] == "similar"
    assert sibling.reason.startswith("담으신 사진과 분위기가 가장 가까운 컷")


def test_relations_alternatives_rarity_album_role(world):  # noqa: F811
    """순간 단위 재료: 형제 컷 + 탈락 사유, 희소 조합, 앨범 자리 — 전부 셀 수 있는 사실이고 breakdown 에 실린다."""
    from photoselect.v1.draft.job import RARITY_MAX_SHARE
    st, settings, rows = world
    by_id = {r.photo_id: r for r in rows}
    combo: dict = {}
    for r in rows:
        combo[(r.subjects, r.expression)] = combo.get((r.subjects, r.expression), 0) + 1
    client = FakeClient({"reasons": []})
    job.run(st, "g", settings, round_no=1, llm=client)
    system = client.calls[0]["system"]
    assert "셀렉터" in system and "감정 설득은 금지" in system
    recs = st.read_recommendations("g")
    seen_sibling = False
    for r in recs:
        b, row = r.score_breakdown, by_id[r.photo_id]
        members = [x for x in rows if x.cluster_id == row.cluster_id]
        alts = b["alternatives"]
        assert {a["photo_id"] for a in alts} == {x.photo_id for x in members} - {r.photo_id}
        assert all(a["why_not"] in ("눈 감김", "덜 선명함", "인상이 약함", "거의 같은 컷") for a in alts)
        if alts:
            seen_sibling = True
            if b["primary_reason"] == "sibling":
                assert r.reason.startswith(f"비슷한 {len(alts) + 1}장 중 이 컷이에요")
        m = combo[(row.subjects, row.expression)]
        rare = m <= max(1, int(RARITY_MAX_SHARE * len(rows))) and row.subjects != "none" and row.expression not in ("none", "eyes_closed")
        assert ("rarity" in b) == rare, (r.photo_id, m)
        if rare:
            assert b["rarity"]["count"] == m and b["rarity"]["total"] == len(rows)
        assert b["album_role"]["scene"] == row.scene and b["album_role"]["position"] >= 1
        assert b["moment"] == row.caption
    assert seen_sibling, "합성 갤러리엔 크기 ≥2 클러스터가 있다"
    # 장면 안 순서는 1..n 연속
    pos: dict = {}
    for r in recs:
        pos.setdefault(r.score_breakdown["album_role"]["scene"], []).append(r.score_breakdown["album_role"]["position"])
    assert all(sorted(v) == list(range(1, len(v) + 1)) for v in pos.values())
