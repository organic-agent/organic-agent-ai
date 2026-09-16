"""사진 하나를 가리키는 값 타입 — 입력(PhotoRef)과 출력(PhotoAnalysis)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PhotoRef:
    """갤러리 안의 사진 하나. 로컬은 파일, DB 는 photos 행 + 미리보기 S3 키."""

    photo_id: str     # 갤러리 안에서 유일. 로컬은 상대 경로, DB는 photos.id
    path: str | None  # 로컬 파일 경로 (DB 모드에서는 임시 다운로드 경로, download=False 면 None)
    #: EXIF 촬영 시각(datetime)·카메라 바디("make model"). 연사 클러스터의 순서·파티션 키 —
    #: 임베더가 채운 photos.taken_at/camera_make/camera_model. 로컬 모드는 None(파일명 순 폴백).
    taken_at: object | None = None
    camera: str | None = None
    #: DB 모드의 미리보기 S3 키 — download=False 로 목록만 읽은 뒤 `download_previews` 로 자기 몫만 내려받는다(#54).
    preview_key: str | None = None


@dataclass
class PhotoAnalysis:
    """`photo_analysis` 한 행. 컬럼 이름을 그대로 필드로 쓴다. SCORE 는 subjects · sub_scores · model_version 만 채운다."""

    photo_id: str
    subjects: str = "unknown"
    technical_pct: float = 50.0
    aesthetic_pct: float = 50.0
    sub_scores: dict = field(default_factory=dict)   # technical_score, aesthetic_score, sharpness, clip_parent …
    cluster_id: int = -1
    cluster_rank: int = 0
    embed_group_id: int = -1
    model_version: str = ""
    #: 마지막으로 점수를 쓴 시각(DB timestamptz | 로컬 ISO 문자열). force 재계산의 "이번 실행 전 점수" 판정(#54).
    analyzed_at: object | None = None
