"""한 번의 점수 실행 결과 — pipeline.run 이 채우고 handler · CLI · 워커가 dict 로 받는다."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ScoreResult:
    gallery: str
    pipeline: str = "v3"
    mode: str = "score"
    targets: int = 0
    processed: int = 0
    skipped: int = 0
    failed: list[str] = field(default_factory=list)
    #: 데드라인 때문에 배치 경계에서 멈췄다. 남은 사진은 remaining.
    stopped: bool = False
    remaining: int = 0
    subjects_used: bool = False
    #: 점수 단계 — full(한 번에) · fast(1단계) · quality(2단계, wes #274 2물결).
    stage: str = "full"
    #: 2단계에서 계산하지 못한 사진. `error` 에 쓰지 않고 빈 점수로 끝 표시만 남긴다 — 폴더에서 빠지지 않게.
    quality_failed: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0
    per_stage_seconds: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "gallery": self.gallery, "pipeline": self.pipeline, "mode": self.mode,
            "targets": self.targets, "processed": self.processed, "skipped": self.skipped,
            "failed": self.failed, "stopped": self.stopped, "remaining": self.remaining,
            "subjectsUsed": self.subjects_used, "stage": self.stage, "qualityFailed": self.quality_failed,
            "elapsedSeconds": round(self.elapsed_seconds, 1),
            "perStageSeconds": {k: round(v, 1) for k, v in self.per_stage_seconds.items()},
        }
