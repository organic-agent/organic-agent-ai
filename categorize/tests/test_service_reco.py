"""추천 기본 점수(reco_base)와 컷 규격 점수 — 얼굴 값이 찬 갤러리에서만 쓰고, 아니면 예전 규칙 그대로."""

from categorize.config.settings import Knobs
from categorize.domain.analysis import PhotoAnalysis
from categorize.service import reco
from categorize.service.pipeline import assign_ranks, clear_ranks

K = Knobs()


def _row(pid, burst, blink=0.1, smile=0.3, tech=50.0, aes=50.0, face=True, bw=0, **extra):
    scores = {"sharpness": 100.0, "is_bw": bw, **extra}
    if face:
        scores.update(face_n=2, face_measured=2, eye_blink=blink, eye_smile=smile, smile=smile)
    return PhotoAnalysis(photo_id=pid, technical_pct=tech, aesthetic_pct=aes, sub_scores=scores, burst_id=burst)


def test_ambiguous_closed_eyes_lose_to_open_eyes_and_to_laughing_closed_eyes():
    rows = [_row("open", 0, blink=0.1, smile=0.3), _row("ambiguous", 0, blink=0.6, smile=0.1),
            _row("laughing", 0, blink=0.6, smile=0.9)]
    cut = reco.cut_scores(rows, K)
    assert cut["open"] > cut["ambiguous"]
    assert cut["laughing"] > cut["ambiguous"]


def test_representative_follows_cut_score_even_when_quality_is_lower():
    rows = [_row("sharp-but-blinking", 0, blink=0.6, smile=0.1, tech=90.0), _row("open", 0, blink=0.1, smile=0.4, tech=40.0)]
    assign_ranks(rows, reco.set_reco_base(rows, K))
    best = next(r for r in rows if r.burst_rank == 0)
    assert best.photo_id == "open" and best.sub_scores["rank_reason"] == "face"
    assert rows[1].sub_scores["reco_base"] > rows[0].sub_scores["reco_base"]


def test_gallery_without_face_values_keeps_old_rule_and_writes_no_reco_base():
    rows = [_row("a", 0, tech=90.0, face=False, reco_base=12.0), _row("b", 0, tech=10.0, face=False)]
    cut = reco.set_reco_base(rows, K)
    assign_ranks(rows, cut)
    assert cut is None and all("reco_base" not in r.sub_scores for r in rows)
    best = next(r for r in rows if r.burst_rank == 0)
    assert best.photo_id == "a" and best.sub_scores["rank_reason"] == "technical"


def test_coverage_below_threshold_counts_only_photos_the_face_stage_ran_on():
    rows = [_row(str(i), i) for i in range(8)] + [_row("x", 8, face=False), _row("y", 9, face=False)]
    assert reco.face_coverage(rows) == 0.8 and reco.set_reco_base(rows, K) is None
    rows[-1].sub_scores["face_n"] = 0          # 얼굴 없음으로 돈 사진은 "돈 것"이다
    assert reco.set_reco_base(rows, K) is not None


def test_scene_order_is_filter_then_burst_size_then_cut():
    big = [_row(f"big{i}", 0) for i in range(3)]
    small = [_row("small", 1)]
    low = [_row(f"low{i}", 2, aes=5.0) for i in range(4)]            # 가장 크지만 미학이 낮은 장면
    rows = big + small + low
    reco.set_reco_base(rows, K)
    base = {r.photo_id: r.sub_scores["reco_base"] for r in rows}
    assert min(base[r.photo_id] for r in big) > base["small"] > max(base[r.photo_id] for r in low)
    assert max(base.values()) == 100.0 and min(base.values()) == 0.0


def test_black_and_white_scene_goes_last_but_an_all_bw_set_keeps_its_order():
    rows = [_row("bw0", 0, bw=1), _row("bw1", 0, bw=1), _row("color", 1)]
    reco.set_reco_base(rows, K)
    assert rows[2].sub_scores["reco_base"] == 100.0
    only_bw = [_row("a0", 0, bw=1), _row("a1", 0, bw=1), _row("b", 1, bw=1)]
    reco.set_reco_base(only_bw, K)
    assert min(only_bw[0].sub_scores["reco_base"], only_bw[1].sub_scores["reco_base"]) > only_bw[2].sub_scores["reco_base"]


def test_unmeasured_cut_takes_the_burst_median_and_unknown_bursts_stay_apart():
    rows = [_row("a", 0, blink=0.1), _row("b", 0, blink=0.7, smile=0.0), _row("c", 0, blink=0.2)]
    rows.append(PhotoAnalysis(photo_id="side", technical_pct=50.0, aesthetic_pct=50.0, burst_id=0,
                              sub_scores={"face_n": 1, "face_measured": 0, "eye_blink": None, "eye_smile": None, "smile": None}))
    cut = reco.cut_scores(rows, K)
    assert cut["side"] == sorted(cut[p] for p in "abc")[1]
    solos = [_row("p", -1), _row("q", -1), _row("r", 0), _row("s", 0)]
    reco.set_reco_base(solos, K)
    base = {r.photo_id: r.sub_scores["reco_base"] for r in solos}
    assert min(base["r"], base["s"]) > max(base["p"], base["q"])     # -1 끼리 한 연사로 뭉치면 크기 2 로 동률이 된다


def test_clear_ranks_removes_reco_base():
    rows = [_row("a", 0, reco_base=70.0)]
    clear_ranks(rows)
    assert "reco_base" not in rows[0].sub_scores


def test_cut_score_is_stored_next_to_reco_base_and_removed_together():
    rows = [_row("open", 0, blink=0.1, smile=0.4), _row("ambiguous", 0, blink=0.6, smile=0.1), _row("other", 1)]
    cut = reco.set_reco_base(rows, K)
    assert all(r.sub_scores["reco_cut"] == round(cut[r.photo_id], 4) for r in rows)
    assert rows[0].sub_scores["reco_cut"] > rows[1].sub_scores["reco_cut"]
    stale = [_row("a", 0, face=False, reco_base=12.0, reco_cut=0.3), _row("b", 0, face=False)]
    assert reco.set_reco_base(stale, K) is None and all("reco_cut" not in r.sub_scores for r in stale)
    clear_ranks(rows)
    assert all("reco_cut" not in r.sub_scores and "reco_base" not in r.sub_scores for r in rows)
