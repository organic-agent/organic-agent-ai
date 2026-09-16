"""로컬 실행 진입점. Lambda와 같은 job.run()을 부른다.

컨테이너를 빌드해 ECR에 밀고 배포하는 사이클을 돌기 전에, 실제 S3와 실제 RDS를 상대로 로직을
검증하는 용도다. 자세한 절차는 README의 "로컬 실행" 절에 있다.

    python -m embedder --gallery-id 1                    # 갤러리에서 벡터 없는 사진 전체 (wes scripts/local-ai.sh)
    python -m embedder --gallery-id 1 --photo-ids 1,2,3  # 그 목록만 (운영 Lambda 와 같은 경로)

재계산은 플래그가 아니라 `photo_analysis` 행 삭제다(관리자 재처리) — v2 에 --force 는 없다(#100).
"""

from __future__ import annotations

import argparse
import json
import logging

from embedder.service import job


def main() -> None:
    parser = argparse.ArgumentParser(prog="embedder")
    parser.add_argument("--gallery-id", type=int, required=True)
    parser.add_argument(
        "--photo-ids", metavar="ID,ID,…",
        help="이 사진 id 목록만 임베딩한다(#73, 운영 Lambda 와 같은 경로). 없으면 갤러리에서 벡터 없는 사진 전체.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
    )

    ids = [int(x) for x in args.photo_ids.split(",") if x.strip()] if args.photo_ids else None
    result = job.run(gallery_id=args.gallery_id, photo_ids=ids)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
