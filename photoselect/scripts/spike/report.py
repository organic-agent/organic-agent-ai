"""리포트 — 골든셋 라벨 대비 러너 점수의 분리도와 속도 요약.

selected 라벨이 채워진 그룹만 AUC·recall@K를 계산한다. 라벨이 아직 없으면
점수 분포·처리 시간 요약만 낸다 (그것만으로도 15분 예산 판정에는 충분).
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

import metrics

SCORE_COLUMNS = [
    "arniqa_technical_score",
    "laion_aesthetic_aesthetic_score",
    "faces_eyes_open",
    "faces_smile",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("out/report.md"))
    args = parser.parse_args()

    with args.manifest.open(newline="") as f:
        labels = {r["photo_id"]: r for r in csv.DictReader(f)}
    with args.scores.open(newline="") as f:
        scored = list(csv.DictReader(f))

    lines = ["# 스파이크 리포트", ""]

    # 처리 시간 요약
    lines += ["## 처리 시간 (사진당)", ""]
    time_cols = sorted({k for r in scored for k in r if k.endswith("_sec")})
    for col in time_cols:
        secs = np.array([float(r[col]) for r in scored if r.get(col)])
        if len(secs):
            lines.append(
                f"- `{col[:-4]}`: 평균 {secs.mean():.3f}s · p95 {np.percentile(secs, 95):.3f}s"
                f" · 3,000장 직렬 {secs.mean() * 3000 / 60:.1f}분 (n={len(secs)})"
            )
    lines.append("")

    # 그룹별 라벨 평가
    groups = sorted({labels[r["photo_id"]]["group"] for r in scored if r["photo_id"] in labels})
    for group in groups:
        rows = [
            r
            for r in scored
            if r["photo_id"] in labels and labels[r["photo_id"]]["group"] == group
        ]
        labeled = [r for r in rows if labels[r["photo_id"]]["selected"] in ("0", "1")]
        lines.append(f"## {group} — {len(rows)}장 (라벨 {len(labeled)}장)")
        lines.append("")
        if not labeled:
            lines.append("라벨 없음 — selected 채운 뒤 재실행")
            lines.append("")
            continue
        y = np.array([int(labels[r["photo_id"]]["selected"]) for r in labeled])
        lines.append(f"작가 셀렉 {int(y.sum())}장 / {len(y)}장")
        lines.append("")
        lines.append("| 점수 축 | AUC | recall@K |")
        lines.append("|---|---|---|")
        for col in SCORE_COLUMNS:
            vals = [r.get(col, "") for r in labeled]
            if not any(vals):
                continue
            s = np.array([float(v) if v else np.nan for v in vals])
            mask = ~np.isnan(s)
            if mask.sum() < 10:
                continue
            lines.append(
                f"| `{col}` | {metrics.auc(s[mask], y[mask]):.3f}"
                f" | {metrics.recall_at_k(s[mask], y[mask]):.3f} |"
            )
        lines.append("")

    # 점수 축 간 겹침 (전체)
    lines.append("## 점수 축 간 SRCC (축이 서로 얼마나 겹치나)")
    lines.append("")
    for a, b in combinations(SCORE_COLUMNS, 2):
        pairs = [
            (float(r[a]), float(r[b]))
            for r in scored
            if r.get(a) and r.get(b)
        ]
        if len(pairs) < 10:
            continue
        arr = np.array(pairs)
        mask = ~np.isnan(arr).any(axis=1)
        if mask.sum() < 10:
            continue
        lines.append(f"- `{a}` × `{b}`: {metrics.srcc(arr[mask, 0], arr[mask, 1]):.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n→ {args.out}")


if __name__ == "__main__":
    main()
