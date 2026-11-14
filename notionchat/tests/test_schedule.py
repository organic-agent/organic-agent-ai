"""팀 일정 '지금 이후' 선별 테스트 — 네트워크 없이 순수 함수만 검증한다."""

from datetime import datetime

from api.notion.schedule import KST, format_upcoming, select_upcoming

NOW = datetime(2026, 11, 14, 15, 25, tzinfo=KST)  # 토요일 오후


def _row(title: str, start: str, end: str | None = None, memo: str = "") -> dict:
    props = {
        "이름": {"type": "title", "title": [{"plain_text": title, "href": None}]},
        "날짜": {"type": "date", "date": {"start": start, "end": end}},
    }
    if memo:
        props["메모"] = {"type": "rich_text", "rich_text": [{"plain_text": memo, "href": None}]}
    return {"id": title, "properties": props}


def _titles(rows, now=NOW, keyword=None):
    return [e.title for e in select_upcoming(rows, now, keyword, lambda r: r["id"])]


def test_today_evening_comes_before_tomorrow():
    # 실제 오답 사례: "다음 멘토링"에 오늘 20:00 대신 내일을 답했다
    rows = [
        _row("[김재훈] 멘토링", "2026-11-15"),
        _row("[한유일] 멘토링", "2026-11-14T20:00:00.000+09:00"),
        _row("[정인호] 멘토링", "2026-11-13T18:30:00.000+09:00"),
    ]
    assert _titles(rows) == ["[한유일] 멘토링", "[김재훈] 멘토링"]


def test_past_today_and_ongoing():
    rows = [
        _row("오늘 아침", "2026-11-14T09:00:00.000+09:00"),  # 지남
        _row("방금 시작", "2026-11-14T15:00:00.000+09:00"),  # 종료 없음, 1시간 안 → 진행 중
        _row("오늘 종일", "2026-11-14"),  # 종일 일정은 오늘이면 남긴다
        _row("끝난 범위", "2026-11-14T13:00:00.000+09:00", "2026-11-14T15:00:00.000+09:00"),
    ]
    events = select_upcoming(rows, NOW, None, lambda r: r["id"])
    assert [e.title for e in events] == ["오늘 종일", "방금 시작"]
    assert events[1].ongoing


def test_multi_day_range_in_progress_is_kept():
    rows = [_row("활동비 신청", "2026-11-10", "2026-11-16"), _row("지난 신청", "2026-11-01", "2026-11-07")]
    assert _titles(rows) == ["활동비 신청"]


def test_keyword_filters_title_and_detail():
    rows = [
        _row("[한유일] 멘토링", "2026-11-14T20:00:00.000+09:00"),
        _row("프로젝트 활동비 신청", "2026-11-16"),
        _row("미팅", "2026-11-17", memo="멘토링 보충"),
    ]
    assert _titles(rows, keyword="멘토링") == ["[한유일] 멘토링", "미팅"]


def test_format_has_weekday_relative_and_weekend_range():
    rows = [
        _row("[김재훈] 멘토링", "2026-11-15", memo="13:00 웹엑스"),
        _row("[한유일] 멘토링", "2026-11-14T20:00:00.000+09:00"),
        _row("[정인호] 멘토링", "2026-11-21T13:00:00.000+09:00"),
    ]
    text = format_upcoming(select_upcoming(rows, NOW, None, lambda r: r["id"]), NOW, limit=2)
    assert "기준 시각: 2026-11-14(토) 15:25 KST" in text
    assert "이번 주말: 2026-11-14(토) ~ 2026-11-15(일)" in text
    assert "1. 2026-11-14(토) 20:00 [오늘] [한유일] 멘토링" in text
    assert "2. 2026-11-15(일) 종일 [내일] [김재훈] 멘토링 / 메모: 13:00 웹엑스" in text
    assert "이후 1건 더 있음" in text


def test_empty():
    assert "예정된 일정 없음" in format_upcoming([], NOW, limit=10)
