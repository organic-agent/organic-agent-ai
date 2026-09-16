"""S3 미리보기 읽기. 이 서버는 원본을 열지 않는다 — embedder가 만든 파생 JPEG만 내려받는다."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import boto3
from botocore.config import Config

from score.domain.photo import PhotoRef

log = logging.getLogger(__name__)


class PreviewStorage:
    def __init__(self, bucket: str) -> None:
        self.bucket = bucket
        # 커넥션 풀은 다운로드 스레드(SCORE_DOWNLOAD_WORKERS, 워커 16)보다 커야 "pool is full" 경고 없이 병렬이 산다(#81).
        self._client = boto3.client("s3", config=Config(retries={"max_attempts": 5, "mode": "standard"},
                                                        max_pool_connections=32))

    def download(self, key: str, dest: Path) -> Path:
        """이미 있으면 다시 받지 않는다 — 재실행·--force 때 갤러리를 통째로 다시 내려받지 않게."""
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._client.download_file(self.bucket, key, str(dest))
        return dest


#: S3 가 "그 키가 없다"고 답한 오류 코드. 이건 다시 받아도 안 되는 결정적 실패라 배치를 죽이지 않고 그 장만 뺀다(#85).
_MISSING_CODES = {"404", "NoSuchKey", "NotFound"}


def _is_missing(exc: Exception) -> bool:
    code = getattr(exc, "response", {}).get("Error", {}).get("Code") if hasattr(exc, "response") else None
    return code in _MISSING_CODES


def download_previews(storage, refs: list[PhotoRef], dest_dir: Path, workers: int = 8,
                      missing: list[str] | None = None) -> list[PhotoRef]:
    """path 가 없는 ref 의 미리보기를 내려받아 path 를 채운 새 목록. 샤드가 자기 몫만 받을 때 쓴다(#54).

    `workers` 스레드로 동시에 받는다(#68): 장당 0.2MB 라 시간은 전송량이 아니라 S3 왕복(~80ms)이 정한다 — 한 프로세스가
    7,000장을 한 장씩 받으면 9분이지만 8개 동시면 1분 남짓. boto3 클라이언트는 스레드에서 같이 써도 된다.

    `missing` 을 주면 S3 에 없는 키(404)는 예외 대신 그 photo_id 를 여기에 담고 결과에서 뺀다(#85) — 호출자가
    `photo_analysis.error='PREVIEW_MISSING'` 을 쓴다. 그 외 오류(접속·권한·스로틀)는 일시적일 수 있어 지금처럼 예외로 올린다.
    `missing` 이 None 이면 옛 동작(모든 오류가 예외)."""

    def fetch(ref: PhotoRef) -> PhotoRef | None:
        if ref.path is None and ref.preview_key:
            try:
                return replace(ref, path=str(storage.download(ref.preview_key, dest_dir / f"{ref.photo_id}.jpg")))
            except Exception as exc:  # noqa: BLE001
                if missing is not None and _is_missing(exc):
                    log.warning("미리보기 없음 %s (%s) — error 로 표시하고 건너뛴다", ref.preview_key, ref.photo_id)
                    missing.append(ref.photo_id)
                    return None
                raise
        return ref

    if workers <= 1:
        fetched = [fetch(ref) for ref in refs]
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            fetched = list(pool.map(fetch, refs))
    return [ref for ref in fetched if ref is not None]
