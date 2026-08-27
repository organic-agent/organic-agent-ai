"""골든셋 매니페스트 — 데이터셋 스캔과 로드.

매니페스트는 CSV 한 장이 전부다:
    photo_id,path,group,selected
selected는 스캔 시점엔 빈칸이고, 작가 최종 셀렉 라벨을 수동으로 1로 채운다.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".heic"}


def build(root: Path, out: Path) -> None:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = path.relative_to(root)
        # Live Photo 번들(*.livephoto/) 내부는 본 HEIC의 부속물 — 건너뛴다
        if any(p.endswith(".livephoto") for p in rel.parent.parts):
            continue
        # 그룹 = 상위 2단계까지 (갤러리 단위 — dataset1/류지혜고객님, dataset2)
        group = "/".join(rel.parts[:-1][:2]) or "root"
        rows.append(
            {
                "photo_id": str(rel),
                "path": str(path.resolve()),
                "group": group,
                "selected": "",
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["photo_id", "path", "group", "selected"])
        writer.writeheader()
        writer.writerows(rows)
    groups = {}
    for r in rows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
    print(f"{len(rows)} photos → {out}")
    for g, n in sorted(groups.items()):
        print(f"  {g}: {n}")


def load(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="데이터셋 스캔 → 매니페스트 CSV")
    b.add_argument("--root", type=Path, required=True)
    b.add_argument("--out", type=Path, default=Path("out/manifest.csv"))
    args = parser.parse_args()

    if args.cmd == "build":
        if not args.root.is_dir():
            sys.exit(f"데이터셋 루트가 없다: {args.root}")
        build(args.root, args.out)


if __name__ == "__main__":
    main()
