"""A-6 근접 중복 클러스터 — 연사(burst)를 한 덩어리로 묶는다.

union-find. 두 사진이 (a) 파일명 순서로 `window` 안에 있고 (b) 임베딩 코사인이
`threshold` 이상이면 같은 클러스터. 연사는 **연속 촬영**이므로 순서 제약이 있어야
"같은 장소에서 다른 순간에 찍은 비슷한 컷"이 잘못 묶이지 않는다.

임계값은 실측으로 정한다 — `similarity_profile()`이 이웃 유사도 분포를 돌려주고
analyze가 그것을 찍어 준다. 합성 가정(01 lab)은 같은 클러스터 0.9 / 다른 클러스터 0.4였다.
"""

from __future__ import annotations

import numpy as np


def _find(parent: list[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def cluster_bursts(emb: np.ndarray, threshold: float, window: int) -> np.ndarray:
    """(N,) cluster_id. 입력 순서 = 파일명 순서여야 한다. emb는 L2 정규화돼 있어야 한다."""
    n = len(emb)
    parent = list(range(n))
    for i in range(n):
        hi = min(n, i + window + 1)
        if hi <= i + 1:
            continue
        sims = emb[i + 1:hi] @ emb[i]
        for k, s in enumerate(sims):
            if s >= threshold:
                a, b = _find(parent, i), _find(parent, i + 1 + k)
                if a != b:
                    parent[b] = a
    roots = [_find(parent, i) for i in range(n)]
    # 등장 순서대로 0,1,2,… 로 다시 번호를 매긴다
    remap: dict[int, int] = {}
    return np.array([remap.setdefault(r, len(remap)) for r in roots], dtype=int)


def partition_order(cameras: list[str | None], taken_ats: list) -> list[list[int]]:
    """연사 클러스터링용 순서 — 카메라 바디별 파티션, 파티션 안은 EXIF 촬영 시각 정렬.

    display_order(업로드/파일명 순)를 촬영 순서로 가정하면 멀티 카메라(메인+세컨드 슈터)에서
    깨진다: 다른 카메라의 동시 촬영 컷이 인접해 교차 병합되거나, 진짜 연사가 창 밖으로
    밀린다. 카메라가 다르면 연사일 수 없으므로 파티션으로 원천 차단한다.

    폴백 규칙: 파티션 안에 taken_at 없는 사진이 하나라도 있으면 그 파티션은 입력 순서
    유지 — 부분 정렬은 시각 있는 사진과 없는 사진의 상대 순서를 뒤섞어 더 위험하다.
    카메라 정보가 없는 사진들은 하나의 "" 파티션이다.
    """
    by_cam: dict[str, list[int]] = {}
    for i, cam in enumerate(cameras):
        by_cam.setdefault(cam or "", []).append(i)
    parts = sorted(by_cam.values(), key=lambda p: p[0])   # 첫 등장 순 — 결과가 입력 순서에 안정
    for part in parts:
        if all(taken_ats[i] is not None for i in part):
            part.sort(key=lambda i: (taken_ats[i], i))    # 동시각은 입력 순서로 안정
    return parts


def cluster_bursts_partitioned(emb: np.ndarray, parts: list[list[int]],
                               threshold: float, window: int) -> np.ndarray:
    """파티션별로 cluster_bursts를 돌리고 전역 유일 id로 합친다. 파티션 경계는 절대 안 넘는다."""
    out = np.full(len(emb), -1, dtype=int)
    base = 0
    for part in parts:
        sub = cluster_bursts(emb[part], threshold, window)
        for local, cid in zip(part, sub):
            out[local] = base + int(cid)
        base += int(sub.max()) + 1 if len(sub) else 0
    return out


def similarity_profile(emb: np.ndarray, window: int) -> dict[str, float]:
    """이웃(순서상 window 내) 쌍의 코사인 분포. 임계값을 고르는 근거."""
    n = len(emb)
    if n < 2:
        return {}
    vals = []
    for i in range(n):
        hi = min(n, i + window + 1)
        if hi > i + 1:
            vals.extend((emb[i + 1:hi] @ emb[i]).tolist())
    v = np.array(vals)
    qs = np.percentile(v, [10, 25, 50, 75, 90, 95, 99])
    return {"n_pairs": float(len(v)), "p10": qs[0], "p25": qs[1], "p50": qs[2],
            "p75": qs[3], "p90": qs[4], "p95": qs[5], "p99": qs[6]}
