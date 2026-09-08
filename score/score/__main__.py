"""로컬 실행 진입점. Lambda 와 같은 job.run() 을 부른다.

    python -m score --gallery-id 12 [--force] [--limit N]                  # DB 모드 (wes 공유 Postgres) — 갤러리 전체
    python -m score --local "dataset1/데이터셋1" [--limit 50] [--force]     # 로컬 데이터셋 → out/v3/
    python -m score --list                                                   # 로컬 데이터셋 갤러리 목록
    python -m score --gallery-id 12 --photo-ids 1,2,3                      # 그 목록만 (운영 Lambda 와 같은 경로)
    python -m score worker --gpu [--once] [--no-idle-stop]                 # GPU 집기 워커 (운영 인스턴스의 기본 CMD)

DB 모드 접속은 DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE, 미리보기는 S3_BUCKET
(wes scripts/local-ai.sh 참조). 잡 테이블·categorize 체인은 wes 가 소유한다 — 이 CLI 는 점수만 쓴다(#98).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from score import job


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="score")
    ap.add_argument("cmd", nargs="?", choices=["worker", "train"],
                    help="worker --gpu: GPU 집기 워커(#75). train: SageMaker 벤치마크 진입점")
    ap.add_argument("--gpu", action="store_true", help="worker: GPU 집기 루프 (photo_analysis SKIP LOCKED 32장씩, 유휴면 자기 정지)")
    ap.add_argument("--no-idle-stop", action="store_true", help="worker --gpu: 유휴여도 인스턴스를 정지하지 않는다(로컬)")
    ap.add_argument("--photo-ids", metavar="ID,ID,…", help="v2 폴백(#75): 이 사진 id 목록만 점수. 잡·체인 없음")
    ap.add_argument("--gallery-id", type=int, help="photos.gallery_id (DB 모드)")
    ap.add_argument("--force", action="store_true", help="이미 점수가 있는 사진도 다시")
    ap.add_argument("--limit", type=int, help="앞에서 N장만 (빠른 확인용)")
    ap.add_argument("--local", metavar="GALLERY", help="로컬 데이터셋 갤러리 이름 (DB 없이 out/ 에 쓴다)")
    ap.add_argument("--list", action="store_true", help="로컬 데이터셋 갤러리 이름과 장수 목록")
    ap.add_argument("--all-formats", action="store_true", help="HEIC 포함 (기본은 JPG만, 로컬)")
    ap.add_argument("--once", action="store_true", help="worker --gpu: 배치 하나만 처리하고 종료")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")
    from score.config import Settings
    settings = Settings.from_env()

    if args.cmd == "train":
        from score import sagemaker
        sys.exit(sagemaker.main())

    if args.cmd == "worker":
        if not args.gpu:
            sys.exit("worker 는 --gpu 로만 쓴다 (잡 폴링 워커는 wes 가 EVENT 로 부르면서 없어졌다, #98)")
        from score import gpu_worker
        summary = gpu_worker.loop(settings, once=args.once, stop_on_idle=not args.no_idle_stop)
        print(json.dumps(summary, ensure_ascii=False))
        if summary.get("aborted"):
            sys.exit(1)
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
    if args.photo_ids:
        ids = [int(x) for x in args.photo_ids.split(",") if x.strip()]
        print(json.dumps(job.run(gallery_id=args.gallery_id, settings=settings, photo_ids=ids), ensure_ascii=False, indent=2))
        return
    result = job.run(gallery_id=args.gallery_id, force=args.force, settings=settings, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
