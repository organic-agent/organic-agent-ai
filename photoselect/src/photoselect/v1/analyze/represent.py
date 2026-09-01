"""A-7 클러스터 대표 — 연사 안에서 어느 컷을 앞세울 것인가. 결정적 규칙, 모델 없음.

순위 기준(앞이 우선): ① 눈 뜸(얼굴이 있을 때만) ② 기술 품질 ③ 미학 ④ 파일명(안정성).
**같은 연사 안의 상대 비교**라 임계값이 필요 없다 — study/02 step7 [D].
`rank_reason_code`는 왜 1등이 됐는지의 짧은 코드. 이유 문장 템플릿이 읽는다.
"""

from __future__ import annotations

import math

from photoselect.v1.store import PhotoAnalysis


def _key(a: PhotoAnalysis) -> tuple:
    eyes = a.sub_scores.get("eyes_open", float("nan"))
    has_face = a.face_boxes.get("face_count", 0) > 0
    eyes_rank = -(eyes if has_face and not math.isnan(eyes) else 0.0)
    return (eyes_rank, -a.technical_pct, -a.aesthetic_pct, a.photo_id)


def _reason(best: PhotoAnalysis, others: list[PhotoAnalysis]) -> str:
    if not others:
        return "single"
    has_face = best.face_boxes.get("face_count", 0) > 0
    if has_face:
        e = best.sub_scores.get("eyes_open", 0.0)
        oe = [o.sub_scores.get("eyes_open", 0.0) for o in others if o.face_boxes.get("face_count", 0) > 0]
        if oe and e - max(oe) > 0.15:
            return "eyes_open"
    if best.technical_pct - max(o.technical_pct for o in others) > 5:
        return "technical"
    if best.aesthetic_pct - max(o.aesthetic_pct for o in others) > 5:
        return "aesthetic"
    return "tie"


def assign_ranks(rows: list[PhotoAnalysis]) -> None:
    """cluster_rank(0이 대표)와 rank_reason_code를 제자리에서 채운다."""
    by_cluster: dict[int, list[PhotoAnalysis]] = {}
    for r in rows:
        by_cluster.setdefault(r.cluster_id, []).append(r)
    for members in by_cluster.values():
        members.sort(key=_key)
        for rank, m in enumerate(members):
            m.cluster_rank = rank
            m.rank_reason_code = _reason(m, members[:rank] + members[rank + 1:]) if rank == 0 else ""
