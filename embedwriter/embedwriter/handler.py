"""Lambda 진입점 — SQS 결과 큐 이벤트 소스 매핑(ReportBatchItemFailures)(wes #286).

임베더가 모델 배치마다 보낸 메시지(계약 v1, `message.py`)를 이벤트 하나 단위로 모아 **한 트랜잭션**으로 쓴다.
동시 실행 ≤ 2(인프라 maximum_concurrency)라 DB 쓰기 세션이 임베더 동시 수(72)가 아니라 둘로 묶인다.

실패 처리:
    - 모양이 틀린 메시지(JSON 아님 · v ≠ 1 · 필드 틀림)는 SQL 전에 걸러 그 메시지만 실패로 돌린다.
    - 묶음 전체 쓰기가 실패하면 메시지마다 따로(각자 트랜잭션) 다시 쓰고, 그래도 실패한 메시지만 `batchItemFailures` 로.
      SQS 가 그 메시지만 다시 보내고, maxReceiveCount 를 넘기면 DLQ 로 간다. UPSERT 라 다시 써도 결과가 같다.
    - DB 접속 자체가 안 되면 묶음·메시지별 시도가 모두 실패하므로 전 메시지가 `batchItemFailures` 로 돌아가 다시 온다.
"""

from __future__ import annotations

import logging
import time

from embedwriter import connection, message, writer
from embedwriter.settings import Settings

logging.getLogger().setLevel(logging.INFO)
log = logging.getLogger(__name__)

_SETTINGS = Settings.from_env()


def handler(event: dict, context=None) -> dict:
    return handle(event, _SETTINGS)


def handle(event: dict, settings: Settings) -> dict:
    started = time.monotonic()
    records = event.get("Records") or []
    failures: list[str] = []
    parsed: list[tuple[str, list[message.PhotoRow]]] = []

    for record in records:
        message_id = record.get("messageId")
        try:
            parsed.append((message_id, message.parse(record.get("body"), settings.embed_dim)))
        except message.InvalidMessage as error:
            log.error("메시지 모양이 틀렸습니다 — 실패로 돌립니다(반복되면 DLQ): messageId=%s %s", message_id, error)
            failures.append(message_id)

    photos = sum(len(rows) for _, rows in parsed)
    written = 0
    bulk = True
    if parsed:
        try:
            written = _write(settings, [row for _, rows in parsed for row in rows])
        except Exception:
            log.exception("묶음 쓰기 실패(메시지 %s개 · 사진 %s장) — 메시지마다 다시 씁니다", len(parsed), photos)
            bulk = False
            for message_id, rows in parsed:
                try:
                    written += _write(settings, rows)
                except Exception:
                    log.exception("메시지 쓰기 실패: messageId=%s (사진 %s장)", message_id, len(rows))
                    failures.append(message_id)

    log.info(
        "embedwriter records=%s photos=%s written=%s failed=%s bulk=%s seconds=%.2f",
        len(records), photos, written, len(failures), bulk, time.monotonic() - started,
    )
    return {"batchItemFailures": [{"itemIdentifier": message_id} for message_id in failures]}


def _write(settings: Settings, rows: list[message.PhotoRow]) -> int:
    """한 트랜잭션. 실패하면 롤백(접속이 깨졌으면 버려 다음 시도가 새로 맺게) 하고 예외를 다시 던진다."""
    conn = connection.shared(settings)
    try:
        written = writer.write(conn, rows)
        conn.commit()
        return written
    except Exception:
        if conn.broken:
            connection.discard()
        else:
            try:
                conn.rollback()
            except Exception:
                connection.discard()
        raise
