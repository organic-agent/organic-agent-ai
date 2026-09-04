"""로컬 CLI. 워커와 같은 run()을 부른다. 이 패키지는 폴더화 배치 하나다 — 잡은 둘(#26).

    python -m photoselect score      --db --gallery 12 [--force] [--limit N]   # 사진별 점수 (torch)
    python -m photoselect categorize --db --gallery 12 [--job-id J] [--llm]   # 그룹 + naming (torch 없음)
    python -m photoselect analyze    --db --gallery 12 [--llm]                # = score → categorize (wes FULL)
    python -m photoselect naming     --db --gallery 12 --job-id J             # = categorize (wes NAMING)
    python -m photoselect worker --llm                                        # 웹 버튼(잡) 폴링 처리

    python -m photoselect score --list                                        # 로컬 데이터셋 갤러리 목록
    python -m photoselect score --gallery "dataset1/…" [--limit 50] [--force]

DB 모드(--db): 갤러리는 photos.gallery_id 숫자. 접속은 DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/
DB_SSLMODE, 미리보기는 S3_BUCKET (wes scripts/local-worker.sh 참조). score 는 torch 가 있는 venv.
폴더별 추천·비교샷은 wes 로 이관됐다(#25).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from photoselect import gallery as gal, jobs, llm as llm_mod, store as store_mod, worker
from photoselect.config import Settings


def _common(p, *, llm_help: str) -> None:
    p.add_argument("--gallery", help="로컬은 갤러리 이름, --db 면 gallery_id 숫자")
    p.add_argument("--db", action="store_true", help="wes DB를 읽고 쓴다")
    p.add_argument("--job-id", type=int, help="ai_analysis_jobs.id — 있으면 RUNNING→DONE/FAILED 를 여기서 기록")
    p.add_argument("--limit", type=int, help="앞에서 N장만 (빠른 확인용)")
    p.add_argument("--force", action="store_true", help="이미 점수가 있는 사진도 다시 (score)")
    p.add_argument("--all-formats", action="store_true", help="HEIC 포함 (기본은 JPG만, 로컬)")
    p.add_argument("--llm", action="store_true", help=llm_help)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="photoselect")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="SCORE — 사진별 점수·CLIP 벡터·피사체 (torch)")
    _common(sc, llm_help="(score 에서는 무시)")
    sc.add_argument("--list", action="store_true", help="로컬 데이터셋 갤러리 이름과 장수 목록")

    ct = sub.add_parser("categorize", help="CATEGORIZE — 백분위·연사·임베딩 그룹 + naming (torch 없음)")
    _common(ct, llm_help="naming 까지 Bedrock 으로. --job-id 가 있는 잡은 필수")

    an = sub.add_parser("analyze", help="= score → categorize (wes FULL 잡과 같은 순서)")
    _common(an, llm_help="naming 까지 Bedrock 으로. --job-id 가 있는 잡은 필수")

    nm = sub.add_parser("naming", help="= categorize (wes NAMING 잡과 같음, Bedrock 필수)")
    _common(nm, llm_help="(naming 은 항상 Bedrock)")

    w = sub.add_parser("worker", help="잡 폴링 워커 — PENDING 분석 잡을 집어 폴더화 실행 (--db 고정)")
    w.add_argument("--llm", action="store_true", help="naming 을 Bedrock 으로 (폴더화 잡은 필수)")
    w.add_argument("--poll", type=float, default=2.0, help="빈 큐일 때 대기 초 (기본 2)")
    w.add_argument("--once", action="store_true", help="쌓인 잡만 처리하고 종료")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")
    settings = Settings.from_env()

    if args.cmd == "worker":
        llm = llm_mod.bedrock_client(settings) if args.llm else None
        worker.loop(settings, llm=llm, poll_seconds=args.poll, once=args.once)
        return

    if args.cmd == "score" and args.list:
        for g, n in gal.list_galleries(settings.dataset_root):
            print(f"{n:>6}  {g}")
        return
    if not args.gallery:
        sys.exit("--gallery 가 필요하다 (로컬 목록은 score --list)")

    steps = {"score": ("score",), "categorize": ("categorize",),
             "analyze": ("score", "categorize"), "naming": ("categorize",)}[args.cmd]
    want_llm = args.llm or args.cmd == "naming"
    llm = llm_mod.bedrock_client(settings) if want_llm else None

    if args.db:
        mode = "FULL" if steps == ("score", "categorize") else ("NAMING" if steps == ("categorize",) else None)
        st = store_mod.DbStore(settings)
        if args.job_id is not None and not jobs.claim(st.conn, jobs.ANALYSIS, args.job_id):
            sys.exit(f"잡 {args.job_id} 은 PENDING 이 아니다 (없거나 다른 워커가 집었다)")
        if mode is None:   # score 단독 — 잡 계약 밖. 워커 배선 없이 점수만 적재한다
            from photoselect import score
            from photoselect.storage import PreviewStorage
            refs = gal.load_db(st.conn, PreviewStorage(settings.s3_bucket), int(args.gallery),
                               settings.work_dir, limit=args.limit)
            result = score.run(st, args.gallery, refs, settings, force=args.force)
        else:
            result = worker.run_analysis_job(st, args.job_id, int(args.gallery), mode, settings,
                                             llm=llm, force=args.force, limit=args.limit)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # 로컬 데이터셋 모드
    from photoselect import categorize, score
    st = store_mod.LocalStore(settings.out_root, dataset_root=settings.dataset_root)
    refs = gal.load_local(settings.dataset_root, args.gallery, limit=args.limit, jpg_only=not args.all_formats)
    result: dict = {"gallery": args.gallery, "pipeline": "v3", "mode": args.cmd}
    if "score" in steps:
        result["score"] = score.run(st, args.gallery, refs, settings, force=args.force)
    if "categorize" in steps:
        result["categorize"] = categorize.run(st, args.gallery, refs, settings, llm, job_id=None)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
