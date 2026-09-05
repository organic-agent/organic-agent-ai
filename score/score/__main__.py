"""로컬 실행 진입점. Lambda 와 같은 job.run() 을 부른다.

    python -m score --gallery-id 12 [--job-id J] [--force] [--limit N]     # DB 모드 (wes 공유 Postgres)
    python -m score --local "dataset1/데이터셋1" [--limit 50] [--force]     # 로컬 데이터셋 → out/v3/
    python -m score --list                                                   # 로컬 데이터셋 갤러리 목록
    python -m score worker [--poll 2] [--once]                               # 로컬 폴링 워커(아래)

DB 모드 접속은 DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE, 미리보기는 S3_BUCKET
(wes scripts/local-ai.sh 참조). 잡(--job-id)은 끝나면 categorize 를 깨우므로 CATEGORIZE_COMMAND 또는
CATEGORIZE_FUNCTION_NAME 이 있어야 한다.

worker: wes 에 분석 Lambda invoker 가 생기기 전까지의 로컬 대용 — `ai_analysis_jobs` PENDING 을 집어
FULL 은 여기서 점수를 낸 뒤 categorize 를 서브프로세스로, NAMING 은 categorize 만 띄운다. 운영에는 없다.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from score import job


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="score")
    ap.add_argument("cmd", nargs="?", choices=["worker"], help="worker: 로컬 폴링 워커")
    ap.add_argument("--gallery-id", type=int, help="photos.gallery_id (DB 모드)")
    ap.add_argument("--job-id", type=int, help="ai_analysis_jobs.id — 있으면 RUNNING 기록 + 끝나면 categorize 체인")
    ap.add_argument("--force", action="store_true", help="이미 점수가 있는 사진도 다시")
    ap.add_argument("--limit", type=int, help="앞에서 N장만 (빠른 확인용)")
    ap.add_argument("--shards", type=int, default=1, help="갤러리를 N 샤드로 나눠 이 프로세스에서 순서대로 (Lambda 는 동시에)")
    ap.add_argument("--local", metavar="GALLERY", help="로컬 데이터셋 갤러리 이름 (DB 없이 out/ 에 쓴다)")
    ap.add_argument("--list", action="store_true", help="로컬 데이터셋 갤러리 이름과 장수 목록")
    ap.add_argument("--all-formats", action="store_true", help="HEIC 포함 (기본은 JPG만, 로컬)")
    ap.add_argument("--poll", type=float, default=2.0, help="worker: 빈 큐일 때 대기 초")
    ap.add_argument("--once", action="store_true", help="worker: 쌓인 잡만 처리하고 종료")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")
    from score.config import Settings
    settings = Settings.from_env()

    if args.cmd == "worker":
        from score import worker
        worker.loop(settings, poll_seconds=args.poll, once=args.once)
        return

    if args.list:
        from score.gallery import list_galleries
        for g, n in list_galleries(settings.dataset_root):
            print(f"{n:>6}  {g}")
        return

    if args.local:
        from score import pipeline
        from score.gallery import load_local
        from score.store import LocalStore
        st = LocalStore(settings.out_root, dataset_root=settings.dataset_root)
        refs = load_local(settings.dataset_root, args.local, limit=args.limit, jpg_only=not args.all_formats)
        result = pipeline.run(st, args.local, refs, settings, force=args.force)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.gallery_id is None:
        sys.exit("--gallery-id 또는 --local 이 필요하다 (--list 로 로컬 목록)")
    if args.shards > 1:
        started_at = job.now_iso() if args.force else None
        results = [job.run(gallery_id=args.gallery_id, force=args.force, settings=settings, job_id=args.job_id,
                           limit=args.limit, shard=job.Shard(i, args.shards), run_started_at=started_at)
                   for i in range(args.shards)]
        result = {"gallery": str(args.gallery_id), "mode": "score", "shards": results,
                  "processed": sum(r.get("processed", 0) for r in results),
                  "stopped": any(r.get("stopped") for r in results),
                  "skipped": job.ALREADY_RUNNING if all(job.was_skipped(r) for r in results) else 0}
    else:
        result = job.run(gallery_id=args.gallery_id, force=args.force, settings=settings,
                         job_id=args.job_id, limit=args.limit)
    # 처리 0장이라도 전부 이미 점수가 있는 재실행이면 끝난 것이다 — categorize 를 이어 부른다. 건너뛴 사진 수로
    # 판단하면 안 된다(job.was_skipped 참고).
    if args.job_id is not None and not job.was_skipped(result) and not result.get("stopped"):
        from score import chain
        result["chained"] = chain.invoke_categorize(settings, args.gallery_id, args.job_id)
        if not result["chained"]:
            from score import db, jobs
            connection = db.connect(settings)
            try:
                jobs.fail(connection, args.job_id, "categorize 서브프로세스 시작 실패")
            finally:
                connection.close()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
