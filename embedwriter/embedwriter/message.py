"""결과 큐 메시지 계약 v1 읽기 — 짝은 embedder `repository/results.py`.

    {"v": 1, "galleryId": 12, "embeddingModel": "...",
     "photos": [{"photoId", "storageKey", "previewKey", "vector": base64(little-endian float32 × embed_dim),
                 "metadata": {"takenAt", "cameraMake", "cameraModel", "exposureTime", "fNumber", "iso",
                              "width", "height", "byteSize"} | null}]}

모양이 틀린 메시지는 `InvalidMessage` 로 거절한다. handler 가 그 메시지만 실패로 돌려 몇 번 뒤 DLQ 로 간다 —
다른 메시지까지 같은 묶음에서 막히지 않게 SQL 에 닿기 전에 걸러 낸다.
"""

from __future__ import annotations

import base64
import binascii
import json
import struct
from dataclasses import dataclass
from datetime import datetime

MESSAGE_VERSION = 1

#: photos 컬럼 길이(V1 baseline). 넘는 값은 UPDATE 가 묶음째 실패하므로 여기서 메시지를 거절한다.
_MAX_KEY = 500
_MAX_TEXT = 100
_MAX_EXPOSURE = 30
_MAX_MODEL = 60  # photo_analysis.embedding_model


class InvalidMessage(ValueError):
    pass


@dataclass(frozen=True)
class PhotoRow:
    """photo_analysis · photos 에 쓸 한 장. `embedding` 은 pgvector 텍스트(`'[0.1,…]'`)."""

    photo_id: int
    storage_key: str
    preview_key: str
    embedding: str
    embedding_model: str
    taken_at: datetime | None = None
    camera_make: str | None = None
    camera_model: str | None = None
    exposure_time: str | None = None
    f_number: float | None = None
    iso: int | None = None
    width: int | None = None
    height: int | None = None
    byte_size: int | None = None


def parse(body: str, embed_dim: int) -> list[PhotoRow]:
    try:
        message = json.loads(body)
    except (TypeError, ValueError) as error:
        raise InvalidMessage(f"JSON 이 아닙니다: {error}") from error
    if not isinstance(message, dict):
        raise InvalidMessage("메시지가 객체가 아닙니다")
    if message.get("v") != MESSAGE_VERSION:
        raise InvalidMessage(f"모르는 계약 버전: {message.get('v')!r}")
    _int(message, "galleryId")
    model = _text(message, "embeddingModel", _MAX_MODEL, required=True)
    photos = message.get("photos")
    if not isinstance(photos, list):
        raise InvalidMessage("photos 가 목록이 아닙니다")
    return [_photo(photo, model, embed_dim) for photo in photos]


def _photo(photo, model: str, embed_dim: int) -> PhotoRow:
    if not isinstance(photo, dict):
        raise InvalidMessage("photos 원소가 객체가 아닙니다")
    meta = photo.get("metadata")
    if meta is not None and not isinstance(meta, dict):
        raise InvalidMessage("metadata 가 객체가 아닙니다")
    meta = meta or {}
    return PhotoRow(
        photo_id=_int(photo, "photoId"),
        storage_key=_text(photo, "storageKey", _MAX_KEY, required=True),
        preview_key=_text(photo, "previewKey", _MAX_KEY, required=True),
        embedding=_vector(photo.get("vector"), embed_dim),
        embedding_model=model,
        taken_at=_taken_at(meta.get("takenAt")),
        camera_make=_text(meta, "cameraMake", _MAX_TEXT),
        camera_model=_text(meta, "cameraModel", _MAX_TEXT),
        exposure_time=_text(meta, "exposureTime", _MAX_EXPOSURE),
        f_number=_float(meta, "fNumber"),
        iso=_int(meta, "iso", required=False),
        width=_int(meta, "width", required=False),
        height=_int(meta, "height", required=False),
        byte_size=_int(meta, "byteSize", required=False),
    )


def _vector(value, embed_dim: int) -> str:
    if not isinstance(value, str):
        raise InvalidMessage("vector 가 base64 문자열이 아닙니다")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as error:
        raise InvalidMessage(f"vector base64 를 풀지 못했습니다: {error}") from error
    if len(raw) != embed_dim * 4:
        raise InvalidMessage(f"vector 길이 {len(raw)}바이트 ≠ float32 × {embed_dim}")
    floats = struct.unpack(f"<{embed_dim}f", raw)
    # 9 유효숫자면 float32 가 그대로 돌아온다(pgvector 는 float4 로 읽는다). NaN·inf 는 vector 가 받지 않는다.
    if any(f != f or f in (float("inf"), float("-inf")) for f in floats):
        raise InvalidMessage("vector 에 NaN·inf 가 있습니다")
    return "[" + ",".join(f"{f:.9g}" for f in floats) + "]"


def _taken_at(value) -> datetime | None:
    """ISO-8601. 컬럼이 timestamp without time zone(EXIF 벽시계)이라 오프셋이 붙어 와도 벽시계 값만 쓴다."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidMessage("takenAt 이 문자열이 아닙니다")
    try:
        return datetime.fromisoformat(value).replace(tzinfo=None)
    except ValueError as error:
        raise InvalidMessage(f"takenAt 형식이 틀렸습니다: {value!r}") from error


def _int(container: dict, name: str, required: bool = True) -> int | None:
    value = container.get(name)
    if value is None and not required:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidMessage(f"{name} 이(가) 정수가 아닙니다: {value!r}")
    return value


def _float(container: dict, name: str) -> float | None:
    value = container.get(name)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InvalidMessage(f"{name} 이(가) 숫자가 아닙니다: {value!r}")
    return float(value)


def _text(container: dict, name: str, limit: int, required: bool = False) -> str | None:
    value = container.get(name)
    if value is None and not required:
        return None
    if not isinstance(value, str) or (required and not value):
        raise InvalidMessage(f"{name} 이(가) 문자열이 아닙니다: {value!r}")
    if len(value) > limit:
        raise InvalidMessage(f"{name} 길이 {len(value)} > {limit}")
    return value
