"""STEP 8 — 다중 클래스와 혼동행렬: "scene 74%"가 숨긴 것 (A-4 VLM 태그).

────────────────────────────────────────────────────────────────────────────
이론

2026-08-25 VLM 채점 리포트는 축별 정확도를 **스칼라 하나**로 적었다.

    scene 74%  ·  lighting 84%  ·  expression 90%  ·  framing 92%  ·  subjects 98%

그런데 채점 원본은 사실 **혼동행렬**이다 — 어떤 정답이 어떤 예측으로 갔는지의 표.
리포트가 스칼라로 압축하면서 버린 정보가 있고, 그중 일부는 **설계 결정에 필요한 것**이다.

1) 다중 클래스에서 '정확도'는 클래스 분포에 지배된다. `snap`이 68%인 갤러리에서
   "전부 snap이라고 답하는" 태거의 scene 정확도는 68%다. 우리 12B 모델이 74%다.
   **그 차이가 6%p뿐이라는 것을 스칼라만 보면 알 수 없다.**

2) 평균 내는 방식이 두 가지다.
   · **micro 평균** — 전체 맞힌 수 / 전체. 큰 클래스가 지배한다 (= 그냥 정확도)
   · **macro 평균** — 클래스별 지표를 낸 뒤 단순 평균. 작은 클래스도 한 표
   `ring`(16장)과 `snap`(271장)을 같은 무게로 볼 것인가가 이 선택이다.

3) 클래스별 재현율과 정밀도는 서로 다른 실패를 가리킨다.
   · 재현율 낮음 = **그 장면을 놓친다** → 커버리지가 그 장면을 못 채운다
   · 정밀도 낮음 = **아닌 것을 그 장면이라 부른다** → 이유 문장이 거짓말을 한다
   우리 시스템에서 두 실패의 비용이 다르다.

4) 지원 수(support)가 적은 클래스의 지표는 못 믿는다. 50장 표본에서 `ring`이 1장이면
   그 클래스의 재현율은 0 아니면 1이다 (02 step3 [A]의 소표본 문제 그대로).

실행: ../../.venv/bin/python step8_confusion.py
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import numpy as np

import abatch as A
import pairsim as P              # noqa: F401  (경로 삽입)

import gallery as G              # noqa: E402

# 2026-08-25 실측 혼동 (spike-report.md 발견 1·2). 형식: 정답 → {예측: 건수}
MEASURED_SCENE = {"snap": {"walk": 5, "prep": 3, "detail": 2, "unknown": 2, "kiss": 1}}
MEASURED_LIGHTING = {"indoor": {"natural": 8}}


def show_matrix(M: np.ndarray, labels: list[str], keep: list[str] | None = None) -> None:
    """혼동행렬을 찍는다. keep을 주면 그 클래스만(지원 수 0인 클래스를 숨긴다)."""
    idx = [labels.index(l) for l in (keep or labels)]
    names = [labels[i] for i in idx]
    head = "정답\\예측".ljust(12) + "".join(f"{n[:7]:>8}" for n in names)
    print("  " + head)
    for i in idx:
        row = f"{labels[i][:10]:<12}" + "".join(
            f"{M[i, j]:>8}" if i != j else f"{('[' + str(M[i, j]) + ']'):>8}" for j in idx)
        print("  " + row)
    print("  ([대각선] = 맞힌 것)")


def main() -> None:
    g = A.tagged_gallery(1000, seed=0, tag_error=1.0)
    true_tags, obs_tags = g.tags_true, g.tags

    # ── [A] 스칼라 정확도가 숨기는 것 ───────────────────────────────────
    print("=" * 92)
    print("[A] 축별 정확도 스칼라 — 그리고 그 옆에 있어야 할 숫자")
    print("=" * 92)
    print("  **같은 태거를 두 갤러리에 돌린다.** 오차 모델은 조건부 확률이라 완전히 동일하다.")
    print(f"  {'':<12}" + f"{'balanced (snap 27%)':^32}" + f"{'measured (snap 68%)':^32}")
    print(f"  {'축':<12}" + f"{'정확도':>10}{'최빈값 기준':>11}{'macro F1':>11}"
          + f"{'정확도':>10}{'최빈값 기준':>11}{'macro F1':>11}")
    g2 = A.tagged_gallery(1000, seed=0, tag_error=1.0, scene_profile="measured")
    for ax, vals in G.AXES.items():
        row = f"  {ax:<12}"
        for gg in (g, g2):
            t = [d[ax] for d in gg.tags_true]
            pr = [d[ax] for d in gg.tags]
            acc = float(np.mean([a == b for a, b in zip(t, pr)]))
            major = max(t.count(v) for v in vals) / len(t)     # '전부 최빈값' 태거의 정확도
            M = A.confusion_matrix(t, pr, vals)
            pc = A.per_class(M)
            macro_f1 = float(np.nanmean(pc["f1"][pc["support"] > 0]))
            row += f"{acc:>10.3f}{major:>11.3f}{macro_f1:>11.3f}"
        print(row)
    print("  → **`scene` 정확도가 갤러리에 따라 0.90에서 0.76으로 바뀐다.** 태거는 똑같은데.")
    print("     오답이 `snap`에서만 나오므로 snap이 많은 갤러리일수록 전체 정확도가 떨어진다.")
    print("     08-25 실측 '74%'는 **snap 68% 표본에서 잰 값**이다. 다른 갤러리에 그대로")
    print("     인용하면 틀린다 — 리포트에 **표본의 클래스 분포를 함께 적어야 하는 이유**다.")
    print("  → 그리고 두 갤러리 모두 '전부 최빈값' 태거의 기준선이 0.27~0.68이다.")
    print("     **기준선 없는 정확도는 읽을 수 없다.** 01 step5 [A]의 '0.5와 구분되느냐'가")
    print("     다중 클래스에서는 '최빈 클래스 비율과 구분되느냐'가 된다.")
    print("  → macro F1은 정확도와 다르게 움직인다 — 클래스마다 한 표씩이라 분포에 덜 휘둘린다.")
    print("     `scene`의 macro F1은 두 갤러리에서 비슷하다. **분포에 안 휘둘리는 지표가 필요하면 macro다.**")

    # ── [B] 혼동행렬을 펼친다 ──────────────────────────────────────────
    print("\n" + "=" * 92)
    print("[B] `scene` 혼동행렬 — 오답이 어디로 가는가")
    print("=" * 92)
    t = [d["scene"] for d in true_tags]
    p = [d["scene"] for d in obs_tags]
    labels = G.AXES["scene"]
    M = A.confusion_matrix(t, p, labels)
    present = [l for l in labels if M[labels.index(l)].sum() > 0]
    show_matrix(M, labels, keep=present)
    pc = A.per_class(M)
    print()
    print(f"  {'클래스':<10}{'지원 수':>8}{'재현율':>9}{'정밀도':>9}{'F1':>8}   주된 오답처")
    for i, l in enumerate(labels):
        if pc["support"][i] == 0:
            continue
        off = [(labels[j], M[i, j]) for j in range(len(labels)) if j != i and M[i, j] > 0]
        worst = max(off, key=lambda x: x[1])[0] if off else "—"
        print(f"  {l:<10}{int(pc['support'][i]):>8}{pc['recall'][i]:>9.3f}"
              f"{pc['precision'][i]:>9.3f}{pc['f1'][i]:>8.3f}   → {worst}")
    print("  → **재현율과 정밀도가 다른 실패를 가리킨다.**")
    print("       재현율 낮음 = 그 장면을 놓친다 → **커버리지가 그 장면을 못 채운다**")
    print("       정밀도 낮음 = 아닌 것을 그 장면이라 부른다 → **이유 문장이 거짓말을 한다**")
    print("     '서약 장면이라 골랐어요'인데 서약이 아니면 고객은 바로 알아챈다.")
    print("     우리 비용 구조에서는 **정밀도가 더 비싸다** — 그런데 스칼라 정확도는 둘을 섞는다.")

    # ── [C] micro vs macro ─────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("[C] micro와 macro — 작은 클래스에 몇 표를 줄 것인가")
    print("=" * 92)
    sup = pc["support"]
    valid = sup > 0
    micro = float(np.diag(M).sum() / M.sum())
    macro_r = float(np.nanmean(pc["recall"][valid]))
    weighted_r = float(np.nansum(pc["recall"][valid] * sup[valid]) / sup[valid].sum())
    print(f"  micro 평균(=정확도)        {micro:.3f}   큰 클래스가 지배한다")
    print(f"  macro 평균(클래스별 단순평균) {macro_r:.3f}   작은 클래스도 한 표")
    print(f"  지원 수 가중 평균           {weighted_r:.3f}")
    print()
    big = labels[int(np.argmax(sup))]
    small = [labels[i] for i in np.argsort(sup) if sup[i] > 0][:3]
    print(f"  최대 클래스 `{big}` {int(sup.max())}장  vs  최소 클래스 {small} "
          f"{[int(sup[labels.index(s)]) for s in small]}장")
    print("  → 우리에게 맞는 것은 **macro**다. 이유는 커버리지 때문이다 —")
    print("     `ring`을 못 잡으면 반지 사진이 초안에서 통째로 빠진다. 16장짜리 클래스지만")
    print("     **고객에게는 `snap` 271장과 같은 무게의 항목**이다.")
    print("  → 08-25 리포트의 축별 정확도는 전부 micro였다. macro를 함께 적어야 한다.")

    # ── [D] 실측 혼동을 넣으면 ─────────────────────────────────────────
    print("\n" + "=" * 92)
    print("[D] 2026-08-25 실측 채점 50장을 같은 방식으로 읽으면")
    print("=" * 92)
    print("  실측 오답 (spike-report 발견 1·2) — 정답 → 예측")
    print("    scene    : snap → walk×5, prep×3, detail×2, unknown×2, kiss×1   (13건 전부 정답이 snap)")
    print("    lighting : indoor → natural ×8                                  (8건 전부 같은 방향)")
    print()
    print("  스칼라로 적으면          : scene 74%, lighting 84%")
    print("  혼동행렬로 읽으면 나오는 것:")
    print("    · scene의 오답은 **전부 한 행에서 나간다**(정답 snap). 즉 다른 장면의 재현율은")
    print("      이 표본에서 100%이고, 무너진 것은 **snap의 재현율**뿐이다 (34장 중 21장 = 0.62)")
    print("    · 반대로 walk·prep·detail·kiss의 **정밀도**가 무너진다 — 아닌 것을 그렇게 부른다")
    print("    · lighting은 오답이 한 칸(indoor→natural)에 몰려 있다 = **한 규칙으로 고칠 수 있다**")
    print("  → 이 세 문장은 **스칼라에서는 절대 나오지 않는다.** 그리고 셋 다 행동을 바꾼다:")
    print("      ① snap 재현율이 문제 → 프롬프트에 '애매하면 snap' 규칙 (실제로 그렇게 했다)")
    print("      ② walk·prep 정밀도가 문제 → 그 값들을 이유 문장에 쓰면 안 된다")
    print("      ③ lighting은 한 칸 → 축 정의를 고칠 필요 없이 프롬프트로 해결")
    print()
    print("  ※ 그런데 이 표본으로 클래스별 지표를 보고하면 안 된다.")
    print("     50장 중 `kiss` 1~2장, `bouquet` 1장 수준이다. 재현율이 0 아니면 1이 나온다.")
    print("     02 step3 [A]에서 본 소표본 문제 그대로 — **지원 수를 반드시 함께 적는다.**")

    # ── [E] 클래스 병합 판단 ───────────────────────────────────────────
    print("\n" + "=" * 92)
    print("[E] 어휘를 고칠 것인가 — 혼동행렬로 판단한다")
    print("=" * 92)
    print("  스파이크 리포트가 열어 둔 선택지: \"`walk`를 없애고 `snap`에 합칠지\"")
    print("  혼동행렬은 이 질문에 답할 수 있다 — **두 클래스가 서로만 헷갈리는가?**")
    print()
    pairs = [("walk", "snap"), ("prep", "snap"), ("detail", "snap")]
    print(f"  {'클래스 쌍':<18}{'A→B':>7}{'B→A':>7}{'상호 혼동 비중':>14}   판단")
    for a, b in pairs:
        ia, ib = labels.index(a), labels.index(b)
        ab, ba = int(M[ia, ib]), int(M[ib, ia])
        off_a = int(M[ia].sum() - M[ia, ia])
        off_b = int(M[ib].sum() - M[ib, ib])
        share = (ab + ba) / max(off_a + off_b, 1)
        verdict = "병합 검토" if share > 0.5 else "다른 원인도 있다"
        print(f"  {f'{a} ↔ {b}':<18}{ab:>7}{ba:>7}{share:>14.2f}   {verdict}")
    print("  → 상호 혼동 비중이 높으면 = 두 클래스가 **서로만** 헷갈린다 = 어휘가 겹친다는 신호.")
    print("     낮으면 다른 클래스와도 섞이고 있다는 뜻이라 병합해도 해결이 안 된다.")
    print("  → **어휘 설계 변경은 이 표를 근거로 한다.** 정확도 스칼라로는 판단할 수 없다.")
    print("  ※ 다만 위 숫자는 합성 오차 모델의 결과다. 실제 판단은 **본식 갤러리를 확보해**")
    print("     같은 표를 뽑은 뒤에 한다 — 지금 데이터에는 예식 장면이 아예 없다(08-25 발견 3).")


if __name__ == "__main__":
    main()
