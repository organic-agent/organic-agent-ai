"""로컬 실행 진입점. Lambda와 같은 job.run()을 부른다.

컨테이너를 빌드해 ECR에 밀고 배포하는 사이클을 돌기 전에, 실제 S3와 실제 RDS를 상대로 로직을
검증하는 용도다. 자세한 절차는 README의 "로컬 실행" 절에 있다.
"""

from __future__ import annotations

import argparse
import json
import logging

from embedder import job


def main() -> None:
    parser = argparse.ArgumentParser(prog="embedder")
    parser.add_argument("--gallery-id", type=int, required=True)
    parser.add_argument(
        "--force",
        action="store_true",
        help="이미 임베딩이 있는 사진까지 다시 계산한다. 모델·전처리를 바꿨을 때만.",
    )
    parser.add_argument(
        "--shards", type=int, default=1,
        help="갤러리를 N개 샤드로 나눠 한 프로세스에서 순차로 돈다(#56). Lambda 의 동시 샤드와 같은 분배·잠금 키.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
    )

    if args.shards <= 1:
        result = job.run(gallery_id=args.gallery_id, force=args.force)
    else:
        # force 의 시작 시각을 샤드가 공유해야 뒤 샤드가 앞 샤드의 벡터를 "이번 실행 것"으로 본다.
        run_started_at = job.now_iso() if args.force else None
        result = [
            job.run(gallery_id=args.gallery_id, force=args.force, shard=job.Shard(i, args.shards),
                    run_started_at=run_started_at)
            for i in range(args.shards)
        ]
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
