"""로컬 CLI. 워커와 같은 run()을 부른다. 이 패키지는 폴더화 배치 하나다 (foldering).

    python -m photoselect_v1 analyze --db --gallery 12 [--llm]        # 폴더화 FULL (--llm 이면 naming까지)
    python -m photoselect_v1 naming  --db --gallery 12 --job-id J     # naming만 다시
    python -m photoselect_v1 worker --llm                             # 웹 버튼(잡) 폴링 처리

    python -m photoselect_v1 analyze --list                           # 로컬 데이터셋 갤러리 목록
    python -m photoselect_v1 analyze --gallery "dataset1/…" [--limit 50] [--force]

DB 모드(--db): 갤러리는 photos.gallery_id 숫자. 접속은 DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/
DB_SSLMODE, 미리보기는 S3_BUCKET (wes scripts/local-worker.sh 참조). 실행 환경: torch 가 있는 venv.
폴더별 추천·비교샷은 wes 로 이관됐다(#25).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from photoselect_v1 import gallery as gal, jobs, llm as llm_mod, store as store_mod, worker
from photoselect_v1.config import Settings


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="photoselect_v1")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="폴더화 FULL — 사진별 분석 + 임베딩 그룹 (--llm 이면 naming까지)")
    a.add_argument("--gallery")
    a.add_argument("--db", action="store_true", help="wes DB를 읽고 쓴다 (--gallery 는 gallery_id 숫자)")
    a.add_argument("--job-id", type=int, help="ai_analysis_jobs.id — 있으면 RUNNING→DONE/FAILED 를 여기서 기록")
    a.add_argument("--list", action="store_true", help="갤러리 이름과 장수 목록")
    a.add_argument("--limit", type=int, help="앞에서 N장만 (빠른 확인용)")
    a.add_argument("--force", action="store_true", help="이미 분석된 사진도 다시")
    a.add_argument("--all-formats", action="store_true", help="HEIC 포함 (기본은 JPG만)")
    a.add_argument("--llm", action="store_true",
                   help="분석 뒤 naming(Bedrock)까지 이어 돈다. --job-id 가 있는 잡은 필수")

    nm = sub.add_parser("naming", help="임베딩 그룹에 (큰 분류, 컨셉) 이름·배정 (Bedrock 필요)")
    nm.add_argument("--gallery", required=True, help="로컬은 갤러리 이름, --db 면 gallery_id 숫자")
    nm.add_argument("--db", action="store_true", help="wes DB를 읽고 쓴다 (ai_concept_assignments 는 --job-id 필수)")
    nm.add_argument("--job-id", type=int, help="ai_analysis_jobs.id (mode=NAMING) — 상태 전이와 배정의 FK")

    w = sub.add_parser("worker", help="잡 폴링 워커 — PENDING 분석 잡을 집어 폴더화 실행 (--db 고정)")
    w.add_argument("--llm", action="store_true", help="naming 을 Bedrock 으로 (폴더화 잡은 필수)")
    w.add_argument("--poll", type=float, default=2.0, help="빈 큐일 때 대기 초 (기본 2)")
    w.add_argument("--once", action="store_true", help="쌓인 잡만 처리하고 종료")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")
    settings = Settings.from_env()
    st = store_mod.LocalStore(settings.out_root, dataset_root=settings.dataset_root)

    if args.cmd == "analyze":
        from photoselect_v1.foldering import analyze, naming

        if args.list:
            for g, n in gal.list_galleries(settings.dataset_root):
                print(f"{n:>6}  {g}")
            return
        if not args.gallery:
            sys.exit("--gallery 또는 --list")
        if args.db:
            _run_db_analyze(args, settings)
            return
        refs = gal.load_local(settings.dataset_root, args.gallery, limit=args.limit,
                              jpg_only=not args.all_formats)
        result = analyze.run(st, args.gallery, refs, settings, force=args.force)
        if args.llm:
            result["naming"] = naming.run(st, args.gallery, settings,
                                          llm_mod.bedrock_client(settings), job_id=None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "naming":
        from photoselect_v1.foldering import naming

        llm = llm_mod.bedrock_client(settings)
        if args.db:
            dbst = store_mod.DbStore(settings)
            if args.job_id is not None and not jobs.claim(dbst.conn, jobs.ANALYSIS, args.job_id):
                sys.exit(f"잡 {args.job_id} 은 PENDING 이 아니다 (없거나 다른 워커가 집었다)")
            result = worker.run_analysis_job(dbst, args.job_id, int(args.gallery), "NAMING",
                                             settings, llm=llm)
        else:
            result = naming.run(st, args.gallery, settings, llm, job_id=None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "worker":
        llm = llm_mod.bedrock_client(settings) if args.llm else None
        worker.loop(settings, llm=llm, poll_seconds=args.poll, once=args.once)
        return


def _run_db_analyze(args, settings: Settings) -> None:
    """DB 모드 폴더화. 잡 id가 있으면 그 행의 상태를 여기서 옮긴다 — 워커(worker.py)와 같은 함수."""
    st = store_mod.DbStore(settings)
    if args.job_id is not None and not jobs.claim(st.conn, jobs.ANALYSIS, args.job_id):
        sys.exit(f"잡 {args.job_id} 은 PENDING 이 아니다 (없거나 다른 워커가 집었다)")
    llm = llm_mod.bedrock_client(settings) if args.llm else None
    result = worker.run_analysis_job(st, args.job_id, int(args.gallery), "FULL", settings,
                                     llm=llm, force=args.force, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
