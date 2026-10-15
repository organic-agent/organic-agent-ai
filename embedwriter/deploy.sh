#!/usr/bin/env bash
# embedwriter/ 를 이미지로 만들어 ECR에 올리고, Lambda가 그 이미지를 집게 한다.
#
# 사용법: embedwriter/deploy.sh                      — 어느 디렉토리에서 실행해도 된다
#         NAME_PREFIX=wes-dev embedwriter/deploy.sh   — dev 스택(<prefix>-embedwriter)
#
# embedder/deploy.sh 와 같은 이유로 terraform apply 나 :latest 푸시만으로는 코드가 안 나간다
# (lifecycle ignore_changes = [image_uri], Lambda 는 갱신 시점의 다이제스트를 고정) — update-function-code 를 반드시 부른다.
# 주소는 terraform output 이 아니라 AWS API로 조회한다.
set -euo pipefail

REGION="ap-northeast-2"
NAME_PREFIX="${NAME_PREFIX:-wes}"
REPO_NAME="${NAME_PREFIX}-embedwriter"
FUNCTION_NAME="${NAME_PREFIX}-embedwriter"
TAG="latest"

# 스크립트가 있는 곳이 빌드 컨텍스트다. COPY embedwriter/ 가 이 디렉토리를 기준으로 잡힌다.
cd "$(dirname "${BASH_SOURCE[0]}")"

for tool in aws docker jq; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "$tool 가 필요하다." >&2
    exit 1
  fi
done

REPO=$(aws ecr describe-repositories --region "$REGION" \
  --repository-names "$REPO_NAME" \
  --query 'repositories[0].repositoryUri' --output text)

if [ -z "$REPO" ] || [ "$REPO" = "None" ]; then
  echo "ECR 저장소 $REPO_NAME 을 찾지 못했다. 인프라가 먼저 apply 되어야 한다." >&2
  exit 1
fi

echo "저장소: $REPO"
echo "함수:   $FUNCTION_NAME"
echo

aws ecr get-login-password --region "$REGION" \
  | docker login --username AWS --password-stdin "${REPO%%/*}"

METADATA=$(mktemp -t embedwriter-build.XXXXXX)
trap 'rm -f "$METADATA"' EXIT

# 세 플래그가 전부 필요하다. 이유는 embedder/README.md 의 "빌드와 배포" 참고.
#   --platform linux/amd64           맥 기본은 arm64인데 함수는 x86_64로 만들어진다
#   --provenance=false --sbom=false  attestation 이 붙으면 manifest list 가 되어 Lambda 가 거절한다
docker buildx build \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  --metadata-file "$METADATA" \
  -t "$REPO:$TAG" --push .

DIGEST=$(jq -r '.["containerimage.digest"]' "$METADATA")
if [ -z "$DIGEST" ] || [ "$DIGEST" = "null" ]; then
  echo "푸시한 이미지의 다이제스트를 읽지 못했다." >&2
  exit 1
fi

echo
echo "푸시됨: $DIGEST"

# 태그가 아니라 태그를 지금 막 가리키게 된 다이제스트로 갱신한다.
aws lambda update-function-code --region "$REGION" \
  --function-name "$FUNCTION_NAME" \
  --image-uri "$REPO@$DIGEST" \
  --no-cli-pager --output text --query 'LastUpdateStatus'

# 갱신은 비동기다. InProgress 인 동안 들어온 메시지는 옛 코드가 처리한다.
echo "갱신을 기다리는 중..."
aws lambda wait function-updated --region "$REGION" --function-name "$FUNCTION_NAME"

CODE_SHA=$(aws lambda get-function-configuration --region "$REGION" \
  --function-name "$FUNCTION_NAME" --query 'CodeSha256' --output text)

# 컨테이너 이미지 함수의 CodeSha256 은 매니페스트 다이제스트와 같은 값이다.
if [ "$CODE_SHA" != "${DIGEST#sha256:}" ]; then
  echo "배포된 코드가 방금 푸시한 이미지와 다르다." >&2
  echo "  기대: ${DIGEST#sha256:}" >&2
  echo "  실제: $CODE_SHA" >&2
  exit 1
fi

echo
echo "완료. $FUNCTION_NAME 이 ${DIGEST#sha256:} 을 실행한다."
echo
echo "  * 메시지 계약(v1)을 바꿨다면 writer 를 먼저 배포한다 — 모르는 버전은 실패로 돌아가 DLQ 로 간다."
echo "  * 스키마가 함께 바뀌었다면 앱(wes)이 먼저 배포되어 있어야 한다. 이 함수는 photos 를 직접 UPDATE 한다."
