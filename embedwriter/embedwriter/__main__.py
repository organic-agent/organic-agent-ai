"""로컬 실행 진입점. Lambda 와 같은 handle() 을 부른다 — SQS 이벤트 JSON 을 파일이나 stdin 으로 받는다.

    python -m embedwriter --event event.json      # {"Records": [{"messageId": "...", "body": "<메시지 JSON 문자열>"}]}
    python -m embedwriter --message msg.json      # 메시지 본문 하나(계약 v1)를 이벤트로 감싸서

접속은 DB_* 환경변수(로컬 터널이면 DB_SSLMODE=require).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(prog="embedwriter")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--event", help="SQS 이벤트 JSON 파일. 없으면 stdin")
    source.add_argument("--message", help="메시지 본문 JSON 파일 하나")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")

    if args.message:
        with open(args.message, encoding="utf-8") as file:
            event = {"Records": [{"messageId": "local-1", "body": file.read()}]}
    elif args.event:
        with open(args.event, encoding="utf-8") as file:
            event = json.load(file)
    else:
        event = json.load(sys.stdin)

    # 설정은 handler import 때 읽힌다(Lambda 와 같다).
    from embedwriter import handler

    print(json.dumps(handler.handler(event), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
