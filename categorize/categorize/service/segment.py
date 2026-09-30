"""1층 구간 — 컨셉 경계 후보. 스튜디오는 컨셉 하나를 다 찍고 쉬었다가 다음 컨셉으로 넘어간다.

    시간 모드   촬영 시각순으로 펴고 `segment_gap_s` 보다 길게 쉰 곳에서 자른다. 테스트 컷 같은 아주 작은 구간
               (`segment_min_share` 미만)만 **공백이 짧은 쪽** 이웃에 붙인다 — 큰 공백을 넘어 붙으면 다른 컨셉이 섞인다.
               그보다 큰 짧은 구간(소품 클로즈업 등)은 그대로 두고 VLM 이 어느 컨셉인지 판단한다.
    이미지 모드 시각이 부족하면(`time_min_coverage` 미만이거나 전부 같은 시각) DINOv3 Ward 로 `visual_units` 개 묶음.
               Ward 가 평균연결보다 훨씬 순수하다(갤러리 18: 40묶음 옮길 사진 1.3% vs 18.9%).

구간은 컨셉보다 잘게 나온다 — 합치는 것은 naming 의 VLM 이다. 근거: docs/experiments/concept-segmentation-2026-09-30.md
(시간만 ARI 0.889, 시간 구간 + VLM 0.938, 시간 없음 DINOv3 Ward + VLM 0.80). 연사는 구간 경계를 넘지 않는다.
시계가 다른 카메라 여러 대는 아직 보정하지 않는다 — 시각을 그대로 믿고 한 줄로 합친다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TIME = "time"
VISUAL = "visual"


@dataclass
class Segmentation:
    mode: str                  # TIME | VISUAL
    units: list[list[int]]     # 행 인덱스 묶음. TIME 이면 시간순(구간 안도 시간순), VISUAL 이면 크기순(안은 중심 가까운 순)


def _seconds(t) -> float | None:
    if t is None:
        return None
    return float(t.timestamp()) if hasattr(t, "timestamp") else float(t)


def _time_units(order: list[int], secs: list[float], gap_s: float, min_size: float) -> list[list[int]]:
    """order = 시각순 행 인덱스. 공백으로 자르고 작은 구간은 공백이 짧은 쪽 이웃에 붙인다."""
    units: list[list[int]] = [[order[0]]]
    gaps: list[float] = []                      # gaps[i] = units[i] 와 units[i+1] 사이 공백
    for prev, cur in zip(order, order[1:]):
        g = secs[cur] - secs[prev]
        if g > gap_s:
            units.append([])
            gaps.append(g)
        units[-1].append(cur)
    while len(units) > 1:
        small = min(range(len(units)), key=lambda i: len(units[i]))
        if len(units[small]) >= min_size:
            break
        left = gaps[small - 1] if small > 0 else float("inf")
        right = gaps[small] if small < len(gaps) else float("inf")
        a = small - 1 if left <= right else small      # a 와 a+1 을 합친다
        units[a:a + 2] = [units[a] + units[a + 1]]
        del gaps[a]
    return units


def _cap_time_units(units: list[list[int]], secs: list[float], cap: int) -> list[list[int]]:
    """구간이 cap 을 넘으면 공백이 가장 짧은 이웃끼리 합친다 — VLM 한 호출의 이미지 상한."""
    units = [list(u) for u in units]
    while len(units) > cap:
        gaps = [secs[units[i + 1][0]] - secs[units[i][-1]] for i in range(len(units) - 1)]
        a = int(np.argmin(gaps))
        units[a:a + 2] = [units[a] + units[a + 1]]
    return units


def _visual_units(E: np.ndarray, rows_idx: list[int], k: int) -> list[list[int]]:
    from scipy.cluster.hierarchy import fcluster, linkage

    if len(rows_idx) <= 1:
        return [list(rows_idx)]
    sub = E[rows_idx]
    labels = fcluster(linkage(sub, method="ward"), min(k, len(rows_idx)), "maxclust")
    units = []
    for lab in np.unique(labels):
        members = [rows_idx[i] for i in np.where(labels == lab)[0]]
        c = E[members].mean(axis=0)
        units.append(sorted(members, key=lambda i: -float(E[i] @ c)))
    units.sort(key=len, reverse=True)
    return units


def _keep_bursts_whole(units: list[list[int]], burst_ids: list[int]) -> list[list[int]]:
    """연사가 두 구간에 걸치면 연사 전체를 사진이 더 많은 쪽으로. 구간 안 순서는 유지한다."""
    unit_of = {i: u for u, members in enumerate(units) for i in members}
    by_burst: dict[int, list[int]] = {}
    for i in unit_of:
        by_burst.setdefault(burst_ids[i], []).append(i)
    move: dict[int, int] = {}
    for members in by_burst.values():
        homes = [unit_of[i] for i in members]
        if len(set(homes)) > 1:
            home = max(set(homes), key=homes.count)
            move.update({i: home for i in members if unit_of[i] != home})
    if not move:
        return units
    out = [[i for i in members if i not in move] for members in units]
    for i, u in move.items():
        out[u].append(i)
    return [u for u in out if u]


def build(taken_ats: list, E: np.ndarray, burst_ids: list[int], knobs) -> Segmentation:
    """행 순서 = E 의 행 순서. taken_ats 는 datetime(또는 초) · None."""
    n = len(taken_ats)
    if n == 0:
        return Segmentation(TIME, [])
    secs = [_seconds(t) for t in taken_ats]
    timed = [i for i in range(n) if secs[i] is not None]
    use_time = len(timed) >= knobs.time_min_coverage * n and len({secs[i] for i in timed}) > 1

    if use_time:
        order = sorted(timed, key=lambda i: (secs[i], i))
        units = _time_units(order, secs, knobs.segment_gap_s, knobs.segment_min_share * n)
        units = _cap_time_units(units, secs, knobs.concept_max_units)
        # 시각 없는 사진 → DINOv3 로 가장 닮은 구간(중심 코사인)
        untimed = [i for i in range(n) if secs[i] is None]
        if untimed:
            C = np.stack([E[u].mean(axis=0) for u in units])
            for i in untimed:
                units[int(np.argmax(C @ E[i]))].append(i)
        seg = Segmentation(TIME, _keep_bursts_whole(units, burst_ids))
    else:
        k = min(knobs.visual_units, knobs.concept_max_units)
        seg = Segmentation(VISUAL, _keep_bursts_whole(_visual_units(E, list(range(n)), k), burst_ids))
    return seg
