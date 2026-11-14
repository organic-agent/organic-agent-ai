"""팀 일정 DB에서 '지금 이후' 일정을 코드로 골라낸다.

"다음 멘토링", "이번 주말 일정" 같은 질문에서 날짜 비교·정렬·요일 계산을 모델에 맡기면
오늘 일정을 빼거나(다음 = 내일부터로 해석) 요일을 틀린다. 그래서 여기서 결정적으로 계산하고,
모델은 결과의 맨 위부터 읽기만 하게 한다.
"""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

from api.notion.blocks import properties_to_text, rich_text

KST = ZoneInfo("Asia/Seoul")
WEEKDAYS = "월화수목금토일"

# 팀 일정 DB (workspace_map.md의 "ASM 부산 17 일정")
SCHEDULE_DB_ID = "35bac24a664c805cbd74db676e16dfbc"
DATE_PROPERTY = "날짜"

# 종료 시각이 없는 일정은 시작 후 이 시간까지 '진행 중'으로 남긴다
ONGOING_GRACE = timedelta(hours=1)
# 여러 날에 걸친 일정을 놓치지 않도록 시작일 기준으로 넉넉히 거슬러 조회한다
LOOKBACK_DAYS = 31
MAX_PAGES = 5


@dataclass
class Event:
    title: str
    start: datetime | date
    end: datetime | date | None
    ongoing: bool
    detail: str
    url: str


def now_kst() -> datetime:
    return datetime.now(KST)


def weekday_label(d: date) -> str:
    return f"{d:%Y-%m-%d}({WEEKDAYS[d.weekday()]})"


def _parse(value: str | None) -> datetime | date | None:
    if not value:
        return None
    if "T" in value:
        dt = datetime.fromisoformat(value)
        return dt if dt.tzinfo else dt.replace(tzinfo=KST)
    return date.fromisoformat(value)


def _as_date(value: datetime | date) -> date:
    return value.astimezone(KST).date() if isinstance(value, datetime) else value


def _is_upcoming(start, end, now: datetime) -> tuple[bool, bool]:
    """(남길지, 이미 시작해 진행 중인지)."""
    last = end or start
    if isinstance(last, datetime):
        if end is None:
            # 종료 시각 없음: 시작 후 ONGOING_GRACE 까지만 진행 중으로 본다
            return start + ONGOING_GRACE >= now, start < now
        return last >= now, _start_before(start, now)
    # 종일 일정: 오늘 날짜면 아직 유효하다
    return last >= now.date(), _start_before(start, now)


def _start_before(start, now: datetime) -> bool:
    if isinstance(start, datetime):
        return start < now
    return start < now.date()


def _title(properties: dict) -> str:
    for prop in properties.values():
        if prop.get("type") == "title":
            return rich_text(prop.get("title", [])) or "(제목 없음)"
    return "(제목 없음)"


def select_upcoming(rows: list[dict], now: datetime, keyword: str | None, url_of) -> list[Event]:
    """DB 행에서 지금 이후(또는 진행 중) 일정만 골라 시작 순으로 정렬한다."""
    events = []
    kw = (keyword or "").strip().lower()
    for row in rows:
        props = row.get("properties", {})
        date_prop = (props.get(DATE_PROPERTY) or {}).get("date") or {}
        start = _parse(date_prop.get("start"))
        if start is None:
            continue
        end = _parse(date_prop.get("end"))
        keep, ongoing = _is_upcoming(start, end, now)
        if not keep:
            continue
        title = _title(props)
        rest = {k: v for k, v in props.items() if k != DATE_PROPERTY and v.get("type") != "title"}
        detail = properties_to_text(rest)
        if kw and kw not in title.lower() and kw not in detail.lower():
            continue
        events.append(Event(title, start, end, ongoing, detail, url_of(row)))

    def sort_key(e: Event):
        # 종일 일정은 그날 00:00 으로 본다 — 같은 날의 시각 있는 일정보다 앞에 둔다
        s = e.start if isinstance(e.start, datetime) else datetime.combine(e.start, datetime.min.time(), KST)
        return s.astimezone(KST)

    return sorted(events, key=sort_key)


def _when(e: Event) -> str:
    def one(v):
        if isinstance(v, datetime):
            v = v.astimezone(KST)
            return f"{weekday_label(v.date())} {v:%H:%M}"
        return f"{weekday_label(v)} 종일"

    text = one(e.start)
    if e.end is not None:
        same_day = _as_date(e.end) == _as_date(e.start)
        if same_day and isinstance(e.end, datetime):
            text += f"~{e.end.astimezone(KST):%H:%M}"
        else:
            text += f" ~ {one(e.end)}"
    return text


def _relative(e: Event, now: datetime) -> str:
    if e.ongoing:
        return "진행 중"
    days = (_as_date(e.start) - now.date()).days
    return {0: "오늘", 1: "내일", 2: "모레"}.get(days, f"{days}일 후")


def format_upcoming(events: list[Event], now: datetime, limit: int) -> str:
    today = now.date()
    monday = today - timedelta(days=today.weekday())
    sunday = monday + timedelta(days=6)
    saturday = monday + timedelta(days=5)
    header = (
        f"기준 시각: {weekday_label(today)} {now.astimezone(KST):%H:%M} KST\n"
        f"이번 주(월~일): {weekday_label(monday)} ~ {weekday_label(sunday)} / "
        f"이번 주말: {weekday_label(saturday)} ~ {weekday_label(sunday)}\n"
        "아래는 이미 끝난 일정을 뺀 예정 일정이며, 시작이 빠른 순이다. 맨 위가 가장 빠른(다음) 일정이다.\n---"
    )
    if not events:
        return f"{header}\n예정된 일정 없음."
    lines = []
    for i, e in enumerate(events[:limit], 1):
        line = f"{i}. {_when(e)} [{_relative(e, now)}] {e.title}"
        if e.detail:
            line += f" / {e.detail}"
        lines.append(f"{line}\n   url: {e.url}")
    more = len(events) - limit
    if more > 0:
        lines.append(f"(이후 {more}건 더 있음 — limit을 늘려 다시 조회)")
    return f"{header}\n" + "\n".join(lines)


def fetch_upcoming_rows(client, now: datetime) -> list[dict]:
    """시작일이 LOOKBACK_DAYS 이내 과거부터인 행을 시작 순으로 모두 가져온다."""
    since = (now.date() - timedelta(days=LOOKBACK_DAYS)).isoformat()
    rows, cursor = [], None
    for _ in range(MAX_PAGES):
        data = client.query_database(
            SCHEDULE_DB_ID,
            filter={"property": DATE_PROPERTY, "date": {"on_or_after": since}},
            sorts=[{"property": DATE_PROPERTY, "direction": "ascending"}],
            page_size=100,
            start_cursor=cursor,
        )
        rows.extend(data.get("results", []))
        if not data.get("has_more"):
            break
        cursor = data.get("next_cursor")
    return rows
