"""실행 결과 값 타입. service 가 만들어 controller(handler · __main__)로 돌려준다.

    RunResult        배치 임베딩 한 번의 결과. run() 이 도는 동안 누적하므로 frozen 이 아니다.
    AdminJobResult   관리자 사진 교체 한 건의 결과.

`to_dict()` 의 camelCase 키는 바깥에 보이는 모양이다. 지금은 읽는 쪽이 없다 — Lambda 는 EVENT 호출이라 wes 가
응답을 받지 않고, handler 는 로그에, __main__ 은 stdout 에 찍을 뿐이다. 누가 이 값을 계약으로 읽게 되면
그때 직렬화를 controller 로 뺀다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from embedder.domain.admin import AdminPhotoEvent


@dataclass
class RunResult:
    gallery_id: int
    targets: int = 0
    #: 이번 실행이 손댄 사진 수 (성공 + 실패). targets - attempted 가 아직 시도하지 않은 수다.
    attempted: int = 0
    processed: int = 0
    #: 원본을 못 읽었거나, 미리보기를 못 올렸거나, 올린 파일을 못 열었거나 -- 어느 쪽이든 벡터가
    #: 없으므로 다음 호출의 fetch_targets가 다시 집어 온다. 미리보기 실패를 따로 세지 않는
    #: 이유가 이것이다: 미리보기가 곧 임베딩 입력이라 둘은 같은 실패다.
    failed: list[str] = field(default_factory=list)
    #: 벡터·미리보기는 나왔지만 촬영 정보만 읽지 못한 사진. 상세 화면에 정보가 덜 나올 뿐
    #: 사진은 멀쩡히 보이고 임베딩도 끝나 있다.
    metadata_failed: list[str] = field(default_factory=list)
    #: 데드라인 때문에 배치 경계에서 멈췄다. 남은 사진은 remaining — wes 가 다시 배정한다.
    stopped: bool = False
    elapsed_seconds: float = 0.0
    #: 사진 목록 호출이면 요청 장수(#73). 갤러리 호출(로컬 CLI)이면 None.
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


@dataclass(frozen=True)
class AdminJobResult:
    """관리자 사진 교체 한 건의 결과. `status` 는 SUCCEEDED · FAILED · IGNORED 셋 중 하나다.

    `detail` 은 상태에 따라 다르다 — 성공이면 `previewKey` 또는 `embeddingDimension`, 실패·무시면 `failureCode`.
    """

    event: AdminPhotoEvent
    status: str
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "jobId": self.event.job_id,
            "attemptCount": self.event.attempt_count,
            "jobType": self.event.job_type,
            "photoId": self.event.photo_id,
            "revisionId": self.event.revision_id,
            "status": self.status,
            **self.detail,
        }
