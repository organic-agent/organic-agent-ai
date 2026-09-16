"""로컬 실행 진입점. Lambda 와 같은 job.run_* 을 부른다.

    python -m preference export --gallery-id 8 [--golden golden/파일명_정리.xlsx]   # DB → out/preference/8/ (터널 한 번)
    python -m preference sanity --gallery-id 8 --golden golden/파일명_정리.xlsx      # 요구사항 (1). --local 이면 npz 캐시
    python -m preference train  [--gallery-ids 8,9,10] [--local]                     # 홀드아웃 + 게이트 + 저장

DB 모드 접속은 DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE. --local 은 export 로 받아둔 npz 만 쓴다.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from preference.config.settings import Settings
from preference.repository.db_store import DbStore
from preference.repository.golden import load_golden
from preference.repository.local_store import LocalStore
from preference.service import job


def _golden_ids(gd, golden_path: Path) -> list[str]:
    """golden 파일명 → photo_id. 파일명이 photos.original_file_name 과 맞아야 한다."""
    gallery_id = gd.gallery_id
    by_name = {n: p for n, p in zip(gd.file_names, gd.photo_ids)}
    items = load_golden(golden_path)
    missing = [it.file_name for it in items if it.file_name not in by_name]
    if missing:
        print(f"경고: golden {len(missing)}개 파일명이 갤러리 {gallery_id} 에 없다: {missing[:5]}", file=sys.stderr)
    return [by_name[it.file_name] for it in items if it.file_name in by_name]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="preference")
    ap.add_argument("cmd", choices=["export", "sanity", "train"])
    ap.add_argument("--gallery-id", help="photos.gallery_id")
    ap.add_argument("--gallery-ids", help="쉼표 구분 — train. 비우면 CLOSED 갤러리 전부")
    ap.add_argument("--golden", type=Path, help="선택 사진 xlsx (DB 의 photo_selection_items 대신)")
    ap.add_argument("--local", action="store_true", help="DB 없이 out/preference/ 의 npz 캐시로")
    ap.add_argument("--out", type=Path, help="결과 JSON 을 이 파일에도 쓴다")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")
    settings = Settings.from_env()
    local = LocalStore(settings.out_root)

    if args.cmd == "export":
        if not args.gallery_id:
            sys.exit("--gallery-id 가 필요하다")
        db = DbStore(settings)
        gd = db.read_gallery(args.gallery_id)
        p = local.write_gallery(gd)
        selected = _golden_ids(gd, args.golden) if args.golden else db.read_selected(args.gallery_id)
        local.write_selected(args.gallery_id, selected)
        result = {"gallery_id": args.gallery_id, "n_photos": gd.n, "n_selected": len(selected), "path": str(p)}

    elif args.cmd == "sanity":
        if not args.gallery_id:
            sys.exit("--gallery-id 가 필요하다")
        store = local if args.local else DbStore(settings)
        if args.golden:
            selected = _golden_ids(store.read_gallery(args.gallery_id), args.golden)
        else:
            selected = store.read_selected(args.gallery_id)
        if not selected:
            sys.exit("선택 사진이 없다 — --golden 을 주거나 photo_selection_items 가 있어야 한다")
        result = job.run_sanity(store, settings, args.gallery_id, selected)

    else:
        store = local if args.local else DbStore(settings)
        ids = [g.strip() for g in args.gallery_ids.split(",")] if args.gallery_ids else None
        result = job.run_train(store, settings, ids, fallback=local)

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
