"""추천 기본 점수(`sub_scores.reco_base`)와 컷 규격 점수 — score 가 쌓은 얼굴 값(#164)을 순위로 바꾼다. numpy 만 쓴다.

    컷 규격 점수 = −눈 위험/sd + smile_weight × 웃음/sd          (갤러리 안 표준편차로 나눈다)
        눈 위험 = σ(a + b·감김 + c·감김² + d·감김·웃음)           가장 많이 감긴 주인공 얼굴(`eye_blink` · `eye_smile`)
        웃음    = 주인공 얼굴의 웃음 평균(`smile`)
        값이 없는 컷(얼굴 없음 · 못 잼)은 같은 연사의 중앙값, 연사 전체가 없으면 갤러리 중앙값
    reco_base = (흑백 연사 → 화질·미학이 낮은 연사 → 연사 크기 → 컷 규격 점수) 순서의 갤러리 안 백분위. 앞일수록 100.

웃으면서 감은 눈은 위험이 낮고 애매하게 감긴 눈은 높다. 화질·미학은 순위의 근거가 아니라 "낮은 장면을 뒤로 미루는" 거름이다 —
빼지 않는다. 흑백만 있는 폴더에서는 모든 연사에 같은 표시가 붙어 순서가 그대로다.

**얼굴 값이 덜 찬 갤러리에는 쓰지 않는다**(`reco_min_face_coverage`). 그때는 `reco_base` 키가 없고 연사 대표도 예전 규칙이라,
wes 는 지금 식(화질 · 미학 백분위)을 그대로 쓴다. 얼굴 값이 없는 것은 이전에 분석한 갤러리와 Lambda 폴백으로 점수를 낸 사진이다(#166).
눈금을 0~100 백분위로 둔 것은 wes 가 기본 점수를 표준화해 가산점과 더하기 때문이다 — 지금 쓰는 값과 같은 눈금이다.
근거: docs/experiments/reco-rubric-2026-11 (e-reco-v3 · g-gates · g4-wes-folders).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from categorize.config.settings import Knobs
from categorize.domain.analysis import PhotoAnalysis

RECO_BASE = "reco_base"


def _burst_key(row: PhotoAnalysis):
    """연사 번호가 없는 사진(-1)은 한 장이 한 연사다 — 모르는 것끼리 뭉치지 않게(wes `FolderCoverage` 와 같은 규칙)."""
    return row.burst_id if row.burst_id >= 0 else ("solo", row.photo_id)


def _std(values: np.ndarray) -> float:
    sd = float(np.std(values)) if len(values) else 0.0
    return sd if sd > 1e-9 else 1.0


def face_coverage(rows: list[PhotoAnalysis]) -> float:
    """얼굴 값 단계가 돈 사진의 비율. `face_n` 이 0 이면 "얼굴 없음"으로 돈 것이고, 키가 없거나 null 이면 돌지 않은 것이다."""
    if not rows:
        return 0.0
    return sum(1 for r in rows if r.sub_scores.get("face_n") is not None) / len(rows)


def cut_scores(rows: list[PhotoAnalysis], k: Knobs) -> dict[str, float]:
    """사진 id → 컷 규격 점수. 클수록 대표로 좋다."""
    a, b, c, d = k.reco_eye_risk
    risk = np.full(len(rows), np.nan)
    smile = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        blink, eye_smile, mean_smile = (r.sub_scores.get(key) for key in ("eye_blink", "eye_smile", "smile"))
        if blink is None or eye_smile is None or mean_smile is None:
            continue
        risk[i] = 1.0 / (1.0 + math.exp(-(a + b * blink + c * blink * blink + d * blink * eye_smile)))
        smile[i] = mean_smile
    known = ~np.isnan(risk)
    score = np.zeros(len(rows))
    if known.any():
        score[known] = -risk[known] / _std(risk[known]) + k.reco_smile_weight * smile[known] / _std(smile[known])
    neutral = float(np.median(score[known])) if known.any() else 0.0
    members: dict = defaultdict(list)
    for i, r in enumerate(rows):
        members[_burst_key(r)].append(i)
    for idx in members.values():
        have = [score[i] for i in idx if known[i]]
        fill = float(np.median(have)) if have else neutral
        for i in idx:
            if not known[i]:
                score[i] = fill
    return {r.photo_id: float(s) for r, s in zip(rows, score)}


def set_reco_base(rows: list[PhotoAnalysis], k: Knobs) -> dict[str, float] | None:
    """`sub_scores.reco_base` 를 채우고 컷 규격 점수를 돌려준다. 얼굴 값이 덜 찼으면 키를 지우고 None.

    백분위가 이미 매겨진 행에만 부른다(`set_percentiles` 뒤) — 거름이 화질·미학 백분위를 읽는다."""
    if not rows or face_coverage(rows) < k.reco_min_face_coverage:
        for r in rows:
            r.sub_scores.pop(RECO_BASE, None)
        return None
    cut = cut_scores(rows, k)
    members: dict = defaultdict(list)
    for r in rows:
        members[_burst_key(r)].append(r)
    scene: dict = {}
    for key, ms in members.items():
        low = (np.median([m.technical_pct for m in ms]) <= k.reco_low_pct
               or np.median([m.aesthetic_pct for m in ms]) <= k.reco_low_pct)
        bw = sum(1 for m in ms if m.sub_scores.get("is_bw") == 1) > len(ms) / 2
        scene[key] = (bool(bw), bool(low), -len(ms), str(key))
    order = sorted(rows, key=lambda r: scene[_burst_key(r)] + (-cut[r.photo_id], r.photo_id))
    last = max(1, len(order) - 1)
    for rank, r in enumerate(order):
        r.sub_scores[RECO_BASE] = round(100.0 * (len(order) - 1 - rank) / last, 4) if len(order) > 1 else 100.0
    return cut
