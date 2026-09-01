"""VLM 프롬프트 1차 vs 2차 — 같은 50장, 같은 정답표로 축별 정확도 비교 (STATUS.md 1.1).

정답표는 1차 채점(`out/vlm_grade.csv`)에서 만든다: verdict=ok면 1차 예측이 정답, ng면 `correct`가
정답. 그래서 **2차는 사람 재채점 없이** 채점된다 — 단, 캡션은 자유 텍스트라 제외.

    .venv/bin/python vlm_compare.py --v1 out/vlm_tags.csv --v2 out/vlm_tags_v2.csv --grade out/vlm_grade.csv

주의: 같은 정답표로 프롬프트를 고쳐 가며 재는 것은 **그 50장에 과적합**할 수 있다(study/02 step4).
2차안이 좋게 나와도 확정은 새 표본으로 한다.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

AXES = ["scene", "framing", "lighting", "expression", "subjects"]


def load(path: Path) -> dict[str, dict]:
    return {r["photo_id"]: r for r in csv.DictReader(path.open(encoding="utf-8")) if not r.get("error")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--grade", type=Path, required=True)
    a = ap.parse_args()
    v1, v2 = load(a.v1), load(a.v2)

    # 정답표: (photo_id, axis) → 값
    truth: dict[tuple[str, str], str] = {}
    for g in csv.DictReader(a.grade.open(encoding="utf-8")):
        pid, ax = g["photo_id"], g["axis"]
        if ax not in AXES or pid not in v1:
            continue
        if g["verdict"] == "ok":
            truth[(pid, ax)] = v1[pid][ax]
        elif g.get("correct"):
            truth[(pid, ax)] = g["correct"]
        # ng인데 정정값 없음(오클릭 1건) → 정답 미상, 제외

    common = [p for p in v1 if p in v2]
    print(f"표본 {len(common)}장 · 정답표 {len(truth)}칸 (캡션 제외)\n")
    print(f"{'축':<12}{'1차':>7}{'2차':>7}{'변화':>8}   2차 오답 방향")
    for ax in AXES:
        n = c1 = c2 = 0
        conf2: collections.Counter = collections.Counter()
        for pid in common:
            t = truth.get((pid, ax))
            if t is None:
                continue
            n += 1
            c1 += v1[pid][ax] == t
            ok2 = v2[pid][ax] == t
            c2 += ok2
            if not ok2:
                conf2[f"{t}→{v2[pid][ax]}"] += 1
        d = (c2 - c1) / n if n else 0
        worst = ", ".join(f"{k}×{v}" for k, v in conf2.most_common(3))
        print(f"{ax:<12}{c1 / n:>7.0%}{c2 / n:>7.0%}{d:>+8.0%}   {worst}")

    # 2차가 1차와 다르게 답했는데 정답표상 '틀림'인 것 — 정답표 자체가 의심스러운 칸 (사람이 볼 것)
    flips = [(pid, ax, v1[pid][ax], v2[pid][ax], truth[(pid, ax)])
             for pid in common for ax in AXES
             if (pid, ax) in truth and v1[pid][ax] == truth[(pid, ax)] and v2[pid][ax] != truth[(pid, ax)]]
    print(f"\n1차는 맞았는데 2차가 틀린 칸 {len(flips)}개 — 프롬프트 변경의 부작용 후보:")
    for pid, ax, p1, p2, t in flips[:12]:
        print(f"  {ax:<11} {p1} → {p2}   {pid.split('/')[-1][:28]}")
    print("\n판단 기준(spike-report): lighting 84→90%+, scene 74→85%+ 이면 프롬프트로 해결.")


if __name__ == "__main__":
    main()
