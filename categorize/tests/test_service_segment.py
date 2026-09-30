"""1층 구간 — 시간 공백으로 자르기 · 작은 구간 흡수 · 이미지 모드(DINOv3 Ward) · 연사 보존."""

from __future__ import annotations

import datetime as dt

import numpy as np

from categorize.config.settings import Knobs
from categorize.service import segment
from tests.helpers import T0, unit


def _times(*blocks):
    """blocks = (장수, 앞 공백 초) 목록 → 5초 간격 시각."""
    out, t = [], T0
    for n, gap in blocks:
        t += dt.timedelta(seconds=gap)
        for _ in range(n):
            out.append(t)
            t += dt.timedelta(seconds=5)
    return out


def _emb(n, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([unit(rng.normal(size=8)) for _ in range(n)])


def test_time_mode_cuts_at_long_breaks():
    times = _times((10, 0), (8, 600), (12, 400))
    seg = segment.build(times, _emb(30), list(range(30)), Knobs(segment_min_share=0.0))
    assert seg.mode == segment.TIME
    assert [len(u) for u in seg.units] == [10, 8, 12]
    assert seg.units[0] == list(range(10))


def test_small_segment_joins_the_neighbour_across_the_shorter_break():
    """3장짜리 테스트 컷은 폴더 자리를 받지 않는다 — 공백이 짧은 쪽(뒤, 400초)에 붙고 큰 공백(3000초)은 넘지 않는다."""
    times = _times((20, 0), (3, 3000), (20, 400))
    seg = segment.build(times, _emb(43), list(range(43)), Knobs(segment_min_share=0.1))
    assert [len(u) for u in seg.units] == [20, 23]
    assert set(seg.units[1]) == set(range(20, 43))


def test_order_follows_taken_at_not_input_order():
    times = _times((5, 0), (5, 600))
    shuffled = times[5:] + times[:5]                         # 파일명 순서가 시각과 어긋난 갤러리
    seg = segment.build(shuffled, _emb(10), list(range(10)), Knobs(segment_min_share=0.0))
    assert seg.units == [[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]


def test_too_many_segments_merge_across_the_shortest_breaks():
    times = _times((4, 0), (4, 900), (4, 400), (4, 1200))
    seg = segment.build(times, _emb(16), list(range(16)), Knobs(segment_min_share=0.0, concept_max_units=3))
    assert [len(u) for u in seg.units] == [4, 8, 4]         # 400초 공백이 가장 짧다


def test_untimed_photo_joins_the_most_similar_segment():
    a, b = unit(np.eye(8)[0]), unit(np.eye(8)[1])
    E = np.stack([a] * 10 + [b] * 10 + [b])
    times = _times((10, 0), (10, 600)) + [None]
    seg = segment.build(times, E, list(range(21)), Knobs(segment_min_share=0.0, time_min_coverage=0.9))
    assert seg.mode == segment.TIME and 20 in seg.units[1]


def test_visual_mode_when_times_are_missing_or_identical():
    rng = np.random.default_rng(1)
    centers = [unit(rng.normal(size=16)) for _ in range(3)]
    E = np.stack([unit(c + 0.02 * rng.normal(size=16)) for c in centers for _ in range(10)])
    for times in ([None] * 30, [T0] * 30):
        seg = segment.build(times, E, list(range(30)), Knobs(visual_units=3))
        assert seg.mode == segment.VISUAL
        assert sorted(sorted(u) for u in seg.units) == [list(range(0, 10)), list(range(10, 20)), list(range(20, 30))]


def test_a_burst_never_spans_two_segments():
    times = _times((5, 0), (5, 600))
    bursts = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]                  # 연사 1 이 공백을 걸친다(4번 사진)
    seg = segment.build(times, _emb(10), bursts, Knobs(segment_min_share=0.0))
    assert sorted(seg.units[0]) == [0, 1, 2, 3] and sorted(seg.units[1]) == [4, 5, 6, 7, 8, 9]
