"""STEP 4 — MMR과 장면 커버리지: 점수 상위 N장이 왜 실패하는가.

────────────────────────────────────────────────────────────────────────────
이론

1) MMR (Carbonell & Goldstein 1998). 이미 뽑은 집합 S가 있을 때 다음 한 장을
       argmax_{d ∉ S} [ λ·Sim1(d, q) − (1−λ)·max_{s∈S} Sim2(d, s) ]
   로 고른다. 앞항은 '얼마나 좋은가'(우리 경우 score), 뒷항은 '이미 뽑은 것과 얼마나
   비슷한가'에 대한 벌점. λ=1이면 그냥 상위 N장, λ가 낮을수록 다양성 우선.
   그리디이고 한 장 뽑을 때마다 max 유사도만 갱신하면 되니 O(K·N)이다. 20줄이면 끝난다.

2) 왜 필요한가. 웨딩 갤러리는 연사(burst)라 거의 같은 컷이 3~8장씩 붙어 있다.
   점수가 높은 컷 옆에는 점수가 비슷한 쌍둥이가 있다 → 상위 30장이 사실상 5장짜리
   슬라이드쇼가 된다. 고객이 "다 비슷한데요"라고 말하는 순간 설득은 실패한다.

3) 커버리지는 MMR과 다른 문제다. MMR은 '비슷한 것끼리 밀어내기'일 뿐 "서약 컷이
   한 장도 없다"를 막지 못한다. 벡터상 멀기만 하면 통과하기 때문이다. 그래서 설계안은
   **장면 축마다 최소 장수를 먼저 배분**하고 남는 자리를 점수로 채운다(하드 제약).

4) 다양성 지표. ILS(intra-list similarity) = 뽑힌 목록 내부 쌍 평균 코사인 유사도.
   낮을수록 다양하다. 만족도와 ILS는 서로 밀고 당긴다 — λ는 그 다이얼이다.

실행: ../../.venv/bin/python step4_mmr.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os

import numpy as np

import gallery as G
from step1_bt import fit_bt
from step3_lambda import axis_confidence, lambda_of, pref_scores, satisfaction, z

TOP_K = 30
STRENGTH, TEMP = 1.5, 0.7


# ── MMR 본체 (이게 전부다) ──────────────────────────────────────────────────
def mmr_select(score: np.ndarray, emb: np.ndarray, k: int, lam: float,
               candidates: np.ndarray | None = None) -> list[int]:
    """score 높고 서로 안 닮은 k장을 그리디로 고른다.

    score: (N,) 표준화된 점수 (Sim1 자리)
    emb:   (N, D) L2 정규화된 벡터 → emb @ emb.T 가 곧 코사인 유사도
    lam:   1.0이면 점수만, 0.0이면 다양성만
    """
    pool = np.arange(len(score)) if candidates is None else np.asarray(candidates)
    selected: list[int] = []
    max_sim = np.full(len(pool), -np.inf)   # pool 각 원소의 '이미 뽑힌 것과의 최대 유사도'
    alive = np.ones(len(pool), dtype=bool)
    for _ in range(min(k, len(pool))):
        penalty = np.where(np.isfinite(max_sim), max_sim, 0.0)
        mmr = lam * score[pool] - (1 - lam) * penalty
        mmr[~alive] = -np.inf
        pick = int(np.argmax(mmr))
        selected.append(int(pool[pick]))
        alive[pick] = False
        sims = emb[pool] @ emb[pool[pick]]          # 새로 뽑힌 것과의 유사도
        max_sim = np.maximum(max_sim, sims)         # 최대값만 갱신하면 된다
    return selected


def ils(emb: np.ndarray, idx: list[int]) -> float:
    """intra-list similarity — 목록 내부 쌍 평균 코사인. 낮을수록 다양하다."""
    if len(idx) < 2:
        return float("nan")
    S = emb[idx] @ emb[idx].T
    iu = np.triu_indices(len(idx), k=1)
    return float(S[iu].mean())


# ── 장면 커버리지 배분 ──────────────────────────────────────────────────────
def coverage_quota(g: G.Gallery, k: int, min_per_scene: int = 1) -> dict[str, int]:
    """장면 분포에 비례해 k장을 배분하되, 갤러리에 존재하는 장면은 최소 min_per_scene장."""
    scenes = [g.scene(i) for i in range(len(g))]
    present = sorted(set(scenes))
    counts = np.array([scenes.count(s) for s in present], dtype=float)
    quota = {s: min_per_scene for s in present}
    remaining = k - min_per_scene * len(present)
    if remaining > 0:
        share = counts / counts.sum() * remaining
        base = np.floor(share).astype(int)
        for s, b in zip(present, base):
            quota[s] += int(b)
        # 반올림 잔여분은 소수부가 큰 장면부터
        left = remaining - int(base.sum())
        for s in [present[i] for i in np.argsort(-(share - base))][:left]:
            quota[s] += 1
    return quota


def select_with_coverage(g: G.Gallery, score: np.ndarray, k: int, lam_mmr: float,
                         one_per_cluster: bool = True) -> list[int]:
    """장면별 쿼터를 먼저 채우고(각 장면 안에서 MMR), 클러스터당 1장 규칙을 건다."""
    quota = coverage_quota(g, k)
    chosen: list[int] = []
    used_clusters: set[int] = set()
    for scene, q in quota.items():
        if q <= 0:
            continue
        cand = [i for i in range(len(g)) if g.scene(i) == scene]
        if one_per_cluster:
            cand = _dedup_by_cluster(g, cand, score, used_clusters)
        picked = mmr_select(score, g.emb, q, lam_mmr, candidates=np.array(cand)) if cand else []
        chosen.extend(picked)
        used_clusters.update(int(g.cluster[i]) for i in picked)
    # 쿼터 합이 k에 못 미치면 남은 자리는 점수순으로
    if len(chosen) < k:
        rest = [i for i in np.argsort(-score) if i not in set(chosen)]
        if one_per_cluster:
            rest = [i for i in rest if int(g.cluster[i]) not in used_clusters]
        chosen.extend(rest[: k - len(chosen)])
    return chosen[:k]


def _dedup_by_cluster(g: G.Gallery, cand: list[int], score: np.ndarray,
                      used: set[int]) -> list[int]:
    """같은 근접 중복 클러스터에서는 점수 최고 1장만 후보로 남긴다."""
    best: dict[int, int] = {}
    for i in cand:
        c = int(g.cluster[i])
        if c in used:
            continue
        if c not in best or score[i] > score[best[c]]:
            best[c] = i
    return list(best.values())


def report(g: G.Gallery, sat: np.ndarray, idx: list[int], label: str) -> None:
    n_scene = len({g.scene(i) for i in idx})
    n_dupe = len(idx) - len({int(g.cluster[i]) for i in idx})
    print(f"  {label:<26} 만족도 {sat[idx].mean():+.3f} · ILS {ils(g.emb, idx):.3f} "
          f"· 장면 {n_scene:>2}종 · 같은 클러스터 중복 {n_dupe:>2}장")


# 장면 분포 프로파일. 커버리지 실습의 결론이 이 값에 달려 있다.
#   balanced — 기본. 아래 모든 숫자는 이 값 기준.
#   measured — 2026-08-25 VLM 실측 분포(snap 68%, 예식 5개 값 0%). 커버리지가
#              배분할 장면이 거의 없는 상태를 재현한다. 환경변수로 바꿔 실행:
#                  SCENE_PROFILE=measured ../../.venv/bin/python step4_mmr.py
SCENE_PROFILE = os.environ.get("SCENE_PROFILE", "balanced")


def main() -> None:
    g = G.make_gallery(1000, seed=0, scene_profile=SCENE_PROFILE)
    if SCENE_PROFILE != "balanced":
        print(f"※ 장면 분포 프로파일: {SCENE_PROFILE} — 해설 문서의 숫자와 다르다\n")
    w_true = G.make_person(seed=0, strength=STRENGTH)
    sat = satisfaction(g, w_true)

    # 30쌍 정도 받은 상태의 현실적인 점수를 만든다 (step3와 같은 방식)
    pool, _ = G.split_pool(G.design_pair_pool(g), 0.5, seed=7)
    tr = G.sample_pairs(pool, 30, seed=1000)
    c, r = G.answer_pairs(g, w_true, tr, temp=TEMP, seed=2000)
    w_hat = fit_bt(g.X[c] - g.X[r], reg=1.0)
    lam = lambda_of(30)
    score = (1 - lam) * z((g.tech + g.aes) / 2) + lam * z(pref_scores(g, w_hat, axis_confidence(g, w_hat, c, r)))

    print("=" * 92)
    print(f"[A] 상위 {TOP_K}장을 고르는 네 가지 방법 (갤러리 {len(g)}장, 증거 30쌍, λ_score={lam:.2f})")
    print("=" * 92)
    top = list(np.argsort(-score)[:TOP_K])
    report(g, sat, top, "점수 상위 N장")
    for lm in (0.9, 0.7, 0.5):
        report(g, sat, mmr_select(score, g.emb, TOP_K, lm), f"MMR (λ_mmr={lm})")
    report(g, sat, select_with_coverage(g, score, TOP_K, 0.7), "커버리지 + 클러스터 + MMR")
    print("\n  읽는 법:")
    print("   ① 상위 N장에는 같은 클러스터(연사 쌍둥이)가 들어온다. 화면에서 '다 비슷한데요'가 되는 실패.")
    print("   ② 통념과 달리 MMR은 만족도까지 올린다. 쌍둥이가 자리를 차지하면 다른 좋은")
    print("      클러스터가 통째로 밀려나기 때문 — 연사 구조에서는 다양성이 공짜가 아니라 이득이다.")
    print("   ③ 그런데 '평균 만족도'라는 지표 자체가 중복을 벌하지 않는다. 30장이 전부 같은 컷이어도")
    print("      평균은 높을 수 있다. 그래서 ILS·클러스터 중복·장면 종류를 **함께** 본다.")
    print("      지표가 실패 모드를 못 잡으면 지표를 늘리는 것이 먼저다 — step5의 주제.")

    # ── [B] λ_mmr 트레이드오프 곡선 ─────────────────────────────────────
    print("\n" + "=" * 92)
    print("[B] λ_mmr 다이얼 — 관련성과 다양성의 교환")
    print("=" * 92)
    print(f"{'λ_mmr':<8}{'만족도':>10}{'ILS':>10}{'장면 종류':>10}{'클러스터 중복':>14}")
    for lm in (1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.0):
        idx = mmr_select(score, g.emb, TOP_K, lm)
        n_scene = len({g.scene(i) for i in idx})
        n_dupe = len(idx) - len({int(g.cluster[i]) for i in idx})
        print(f"{lm:<8}{sat[idx].mean():>10.3f}{ils(g.emb, idx):>10.3f}{n_scene:>10}{n_dupe:>14}")
    print("  → λ_mmr=0은 '가장 안 닮은 것'만 고른다 = 품질을 완전히 버린다(만족도 붕괴).")
    print("     이 합성 데이터에서는 0.3~0.5가 가장 좋게 나오지만, 위 ③의 이유로 다양성 쪽이")
    print("     과대평가되고 있을 수 있다. Elastic 글의 권장 대역도 0.3~0.9로 넓다.")
    print("     → λ_mmr은 실제 갤러리 결과를 사람 눈으로 보고 정할 값이다. 오프라인 수치로 못 박지 않는다.")

    # ── [C] 장면 분포 비교 ──────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("[C] 장면 분포 — 상위 N장 vs 커버리지 배분")
    print("=" * 92)
    quota = coverage_quota(g, TOP_K)
    cov = select_with_coverage(g, score, TOP_K, 0.7)
    gallery_dist = {s: sum(1 for i in range(len(g)) if g.scene(i) == s) for s in G.AXES["scene"]}
    print(f"{'scene':<10}{'갤러리':>8}{'쿼터':>7}{'상위N':>7}{'커버리지':>9}")
    for s in G.AXES["scene"]:
        if gallery_dist[s] == 0:
            continue
        n_top = sum(1 for i in top if g.scene(i) == s)
        n_cov = sum(1 for i in cov if g.scene(i) == s)
        print(f"{s:<10}{gallery_dist[s]:>8}{quota.get(s, 0):>7}{n_top:>7}{n_cov:>9}")
    print("  → 상위 N장에서 0장이 되는 장면이 생긴다. 그게 하필 '서약'이면 고객은 항의한다.")
    print("     커버리지는 점수로 해결되지 않는 종류의 요구다 — 그래서 하드 제약으로 둔다.")


if __name__ == "__main__":
    main()
