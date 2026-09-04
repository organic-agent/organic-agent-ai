"""로컬 실행 진입점. Lambda 와 같은 job.run() 을 부른다.

    python -m categorize --gallery-id 12 --job-id J            # DB 모드, 잡 — naming 까지 (Bedrock 필수)
    python -m categorize --gallery-id 12 [--llm]                # DB 모드, 잡 없이 그룹화 (+naming 은 저장 안 함)
    python -m categorize --local "dataset1/데이터셋1" [--llm]    # 로컬 데이터셋 (score --local 뒤에) → out/v3/

DB 모드 접속은 DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE, 대표 사진은 S3_BUCKET.
--job-id 가 있으면 --llm 은 자동이다(잡의 산출물이 naming 이라서). Bedrock 은 AWS 자격증명 + InvokeModel.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from categorize import job, llm as llm_mod


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="categorize")
    ap.add_argument("--gallery-id", type=int, help="photos.gallery_id (DB 모드)")
    ap.add_argument("--job-id", type=int, help="ai_analysis_jobs.id — RUNNING→DONE/FAILED 를 여기서 기록")
    ap.add_argument("--llm", action="store_true", help="naming 까지 Bedrock 으로 (--job-id 면 자동)")
    ap.add_argument("--limit", type=int, help="앞에서 N장만 (빠른 확인용)")
    ap.add_argument("--local", metavar="GALLERY", help="로컬 데이터셋 갤러리 이름 (DB 없이 out/ 에서 읽고 쓴다)")
    ap.add_argument("--all-formats", action="store_true", help="HEIC 포함 (기본은 JPG만, 로컬)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")
    from categorize.config import Settings
    settings = Settings.from_env()
    want_llm = args.llm or args.job_id is not None
    llm = llm_mod.bedrock_client(settings) if want_llm else None

    if args.local:
        from categorize import pipeline
        from categorize.gallery import load_local
        from categorize.store import LocalStore
        st = LocalStore(settings.out_root, dataset_root=settings.dataset_root)
        refs = load_local(settings.dataset_root, args.local, limit=args.limit, jpg_only=not args.all_formats)
        result = pipeline.run(st, args.local, refs, settings, llm, job_id=None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.gallery_id is None:
        sys.exit("--gallery-id 또는 --local 이 필요하다")
    result = job.run(gallery_id=args.gallery_id, settings=settings, job_id=args.job_id, llm=llm, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
