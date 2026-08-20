"""점수 실행 — 매니페스트의 사진마다 선택한 러너를 돌려 scores.csv에 적재.

- 사진 × 러너 단위 처리 시간(초)을 함께 기록한다 (Lambda 15분 예산 판정 재료).
- 기존 scores.csv가 있으면 이미 점수가 있는 (photo_id, 러너) 조합은 건너뛴다
  — 중단 후 재실행이 곧 재개 (서비스의 멱등 설계와 같은 모양).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import manifest as manifest_mod
import runners


def load_existing(out: Path) -> dict[str, dict[str, str]]:
    if not out.exists():
        return {}
    with out.open(newline="") as f:
        return {row["photo_id"]: row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runners", required=True, help="쉼표 구분 (예: faces,arniqa)")
    parser.add_argument("--out", type=Path, default=Path("out/scores.csv"))
    parser.add_argument("--limit", type=int, default=0, help="0이면 전체")
    parser.add_argument("--group", default="", help="이 그룹만 (예: dataset2)")
    args = parser.parse_args()

    rows = manifest_mod.load(args.manifest)
    if args.group:
        rows = [r for r in rows if r["group"] == args.group]
    if args.limit:
        rows = rows[: args.limit]

    names = [n.strip() for n in args.runners.split(",") if n.strip()]
    active = {name: runners.create(name) for name in names}
    print(f"러너 {names} × 사진 {len(rows)}장")

    existing = load_existing(args.out)
    results: dict[str, dict[str, str]] = dict(existing)
    fieldnames = ["photo_id"]

    started = time.monotonic()
    done = 0
    for row in rows:
        pid = row["photo_id"]
        record = results.setdefault(pid, {"photo_id": pid})
        for name, runner in active.items():
            time_key = f"{name}_sec"
            if record.get(time_key):  # 이미 처리 — 재개
                continue
            t0 = time.monotonic()
            try:
                signals = runner.score(row["path"])
            except Exception as e:  # 깨진 파일 등 — 기록하고 계속
                print(f"  실패 {pid} ({name}): {e}")
                signals = {}
            elapsed = time.monotonic() - t0
            for k, v in signals.items():
                record[f"{name}_{k}"] = f"{v:.6f}"
            record[time_key] = f"{elapsed:.4f}"
            record[f"{name}_version"] = runner.version
        done += 1
        if done % 25 == 0:
            rate = done / (time.monotonic() - started)
            print(f"  {done}/{len(rows)} ({rate:.2f}장/초)")

    for record in results.values():
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results.values())

    # 러너별 평균 처리 시간 요약
    print(f"\n완료 → {args.out}")
    for name in names:
        secs = [
            float(r[f"{name}_sec"])
            for r in results.values()
            if r.get(f"{name}_sec")
        ]
        if secs:
            avg = sum(secs) / len(secs)
            print(f"  {name}: 평균 {avg:.3f}s/장 (n={len(secs)}) → 3,000장 직렬 {avg * 3000 / 60:.1f}분")


if __name__ == "__main__":
    main()
