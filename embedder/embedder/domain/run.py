"""배치 임베딩 한 번의 결과. service 가 만들어 controller 로 돌려준다.

run() 이 도는 동안 누적하므로 frozen 이 아니다. `to_dict()` 는 로그·stdout 용이고 아직 읽는 계약은 없다.
"""

from __future__ import annotations

from dataclasses import dataclass, field



@dataclass
class RunResult:
    gallery_id: int
    targets: int = 0
    #: 손댄 사진 수(성공 + 실패). targets - attempted 가 아직 시도하지 않은 수.
    attempted: int = 0
    processed: int = 0
    #: 원본 GET·미리보기 PUT·재디코드 중 어디서 실패했든 벡터가 없으므로 다음 호출이 다시 집는다.
    failed: list[str] = field(default_factory=list)
    #: 벡터·미리보기는 나왔지만 촬영 정보만 못 읽은 사진.
    metadata_failed: list[str] = field(default_factory=list)
    #: 데드라인 때문에 배치 경계에서 멈췄다. 남은 사진은 wes 가 다시 배정한다.
    stopped: bool = False
    elapsed_seconds: float = 0.0
    #: 사진 목록 호출이면 요청 장수. 갤러리 호출(로컬 CLI)이면 None.
    photo_ids: int | None = None

    @property
    def remaining(self) -> int:
        return self.targets - self.attempted

    def to_dict(self) -> dict:
        out = {
            "galleryId": self.gallery_id,
            "targets": self.targets,
            "processed": self.processed,
            "failed": self.failed,
            "metadataFailed": self.metadata_failed,
            "stopped": self.stopped,
            "remaining": self.remaining,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
        }
        if self.photo_ids is not None:
            out["photoIds"] = self.photo_ids
        return out
