"""골든 셋 평가 러너 — tests/eval_questions.jsonl의 질문을 실제로 수행해 채점한다.

실행: notionchat/ 에서
  python -m tests.run_eval              # 전체 1회
  python -m tests.run_eval --repeat 5   # 문항당 5회 (라우팅 일관성 측정)
  python -m tests.run_eval --id next-expense-period   # 특정 문항만

실제 Bedrock·Notion을 호출하므로 비용이 든다 (문항당 수 원).
프롬프트·도구·지도 변경 후 반드시 이걸 돌려 회귀 여부를 확인한다 (docs/retrieval-improvement-plan.md).

문항 필드:
  question           질문
  today              (선택) 날짜 고정 "YYYY-MM-DD(요일)" — 시간 상대 질문의 정답 불변화
  expect_contains    답변에 있어야 할 것들. 각 항목은 "a|b" 대안 허용 (정규식)
  expect_not_contains 답변에 있으면 안 되는 것들 (정규식)
  expect_tool        호출됐어야 할 도구 ("a|b" 대안 허용)
  expect_not_tool    호출되면 안 되는 도구
"""

import argparse
import json
import re
import sys
from pathlib import Path

from api.core.provider import ChatError, run_turn

QUESTIONS = Path(__file__).resolve().parent / "eval_questions.jsonl"


def load_items() -> list[dict]:
    items = []
    for line in QUESTIONS.read_text().splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def judge(item: dict, answer: str, tools: list[str]) -> list[str]:
    """실패 사유 목록을 반환한다. 비면 통과."""
    fails = []
    for pattern in item.get("expect_contains", []):
        if not re.search(pattern, answer):
            fails.append(f"누락: /{pattern}/")
    for pattern in item.get("expect_not_contains", []):
        # 줄 단위 매칭 — 표의 다른 행까지 걸리지 않게 .이 줄바꿈을 넘지 않는다
        if re.search(pattern, answer):
            fails.append(f"금지 표현: /{pattern}/")
    if pattern := item.get("expect_tool"):
        if not any(re.fullmatch(pattern, t) for t in tools):
            fails.append(f"도구 미호출: {pattern} (실제: {tools or '없음'})")
    if pattern := item.get("expect_not_tool"):
        if any(re.fullmatch(pattern, t) for t in tools):
            fails.append(f"금지 도구 호출: {pattern}")
    return fails


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=1, help="문항당 반복 횟수")
    parser.add_argument("--id", help="이 id의 문항만 실행")
    args = parser.parse_args()

    items = load_items()
    if args.id:
        items = [i for i in items if i["id"] == args.id]
        if not items:
            sys.exit(f"해당 id 없음: {args.id}")

    total_pass = total_run = 0
    total_tokens = 0
    for item in items:
        passes = 0
        last_fails: list[str] = []
        for _ in range(args.repeat):
            tools: list[str] = []
            try:
                answer, _, usage = run_turn(
                    item["question"], [],
                    on_log=lambda m, t=tools: t.append(m.split("(")[0]),
                    today=item.get("today"),
                )
                total_tokens += usage["inputTokens"] + usage["outputTokens"]
            except ChatError as e:
                last_fails = [f"실행 오류: {e}"]
                continue
            fails = judge(item, answer, tools)
            if fails:
                last_fails = fails
            else:
                passes += 1
        total_run += args.repeat
        total_pass += passes
        mark = "✅" if passes == args.repeat else ("⚠️" if passes else "❌")
        print(f"{mark} {item['id']}: {passes}/{args.repeat}")
        if last_fails:
            for f in last_fails:
                print(f"     └ {f}")

    print(f"\n합계: {total_pass}/{total_run} 통과 · 토큰 {total_tokens:,} (입력+출력)")
    sys.exit(0 if total_pass == total_run else 1)


if __name__ == "__main__":
    main()
