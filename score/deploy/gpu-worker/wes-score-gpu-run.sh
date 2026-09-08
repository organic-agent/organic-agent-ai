#!/bin/bash
# wes-score GPU 워커 — systemd ExecStart(#87). env 생성 → ECR 로그인 → 이동 태그 :gpu pull → 컨테이너를 포그라운드로 exec.
#
# 코드 배포 = 태그 이동(AI CI 가 main 마다 gpu·gpu-<sha> 를 push). 같은 다이제스트면 pull 은 2초 안팎. AMI 재빌드는 드라이버 때만.
# 콜드 예산(NEXT.md §1-B): 부팅 16s + pull 확인 2s + 러너 로드 9s + 워밍 <1s → Start → 첫 배치 ≤ 40s.
set -euo pipefail

ENV_FILE="${WES_SCORE_ENV_FILE:-/run/wes-score.env}"
/usr/local/bin/wes-score-env.sh
# shellcheck disable=SC1090
set -a; . "$ENV_FILE"; set +a

aws ecr get-login-password --region "$AWS_DEFAULT_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY" >/dev/null
t0=$(date +%s)
docker pull --quiet "$ECR_IMAGE"
echo "pull ${ECR_IMAGE} $(( $(date +%s) - t0 ))s digest=$(docker image inspect --format '{{index .RepoDigests 0}}' "$ECR_IMAGE" 2>/dev/null || echo ?)"

# 이전 기동이 남긴 컨테이너가 있으면 치운다(이름 충돌 방지).
docker rm -f wes-score-worker >/dev/null 2>&1 || true
# --rm: 끝나면 삭제. 이미지 CMD 가 `worker --gpu`(Dockerfile.gpu). 컨테이너는 IMDSv2(hop limit 2)로 역할 자격증명·자기 인스턴스 id 를 얻는다.
exec docker run --rm --name wes-score-worker --gpus all --env-file "$ENV_FILE" \
  --log-driver journald --log-opt tag=wes-score-worker \
  "$ECR_IMAGE"
