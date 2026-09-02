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
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
    )

    result = job.run(gallery_id=args.gallery_id, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
