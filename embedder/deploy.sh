#!/usr/bin/env bash
# embedder/ 를 이미지로 만들어 ECR에 올리고, Lambda가 그 이미지를 집게 한다.
#
# 사용법: embedder/deploy.sh        — 어느 디렉토리에서 실행해도 된다
#
# 이 함수는 앱과 배포 경로가 다르다. cd.yml 은 embedder/ 를 건드리지 않으므로 main 에
# 머지해도 운영 함수는 그대로다. 따로 안 나가는 것이 둘 더 있다:
#
#   - terraform apply 로도 코드는 안 나간다. modules/embedding 에
#     lifecycle { ignore_changes = [image_uri] } 가 걸려 있다.
#   - ECR에 :latest 를 푸시하는 것만으로도 안 나간다. Lambda는 갱신 시점의 다이제스트를
#     고정해 두므로 update-function-code 를 반드시 불러야 한다.
#
# 주소를 terraform output 이 아니라 AWS API로 조회한다. 인프라 저장소가 어디에 있든,
# 심지어 없어도 동작하게 하기 위해서다 (scripts/db-tunnel.sh 와 같은 이유).
set -euo pipefail

REGION="ap-northeast-2"
REPO_NAME="wes-embedder"
FUNCTION_NAME="wes-embedder"
TAG="latest"

# 스크립트가 있는 곳이 빌드 컨텍스트다. Dockerfile 이 embedder/ 안에 있고
# COPY embedder/ 가 이 디렉토리를 기준으로 잡힌다.
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

METADATA=$(mktemp -t embedder-build.XXXXXX)
trap 'rm -f "$METADATA"' EXIT

# 가중치를 굽는 단계가 Hugging Face 토큰을 요구한다(DINOv3는 게이트 모델). HF_TOKEN 이
# 있으면 그것을, 없으면 `hf auth login` 이 남긴 파일을 빌드 시크릿으로 넘긴다. 시크릿은
# 이미지 레이어에 남지 않는다.
HF_TOKEN_FILE="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [ -n "${HF_TOKEN:-}" ]; then
  HF_SECRET=(--secret "id=hf_token,env=HF_TOKEN")
elif [ -r "$HF_TOKEN_FILE" ]; then
  HF_SECRET=(--secret "id=hf_token,src=$HF_TOKEN_FILE")
else
  echo "Hugging Face 토큰이 없다. HF_TOKEN 을 내보내거나 'hf auth login' 을 먼저 실행할 것." >&2
  echo "토큰의 계정이 huggingface.co/${EMBED_MODEL_ID:-facebook/dinov3-vitb16-pretrain-lvd1689m} 라이선스를 승인했어야 한다." >&2
  exit 1
fi

# 세 플래그가 전부 필요하다. 이유는 README 의 "빌드와 배포" 참고.
#   --platform linux/amd64        맥 기본은 arm64인데 함수는 x86_64로 만들어져 있다
#   --provenance=false --sbom=false  attestation 이 붙으면 manifest list 가 되어 Lambda 가 거절한다
docker buildx build \
  --platform linux/amd64 \
  --provenance=false --sbom=false \
  --metadata-file "$METADATA" \
  "${HF_SECRET[@]}" \
  -t "$REPO:$TAG" --push .

DIGEST=$(jq -r '.["containerimage.digest"]' "$METADATA")
if [ -z "$DIGEST" ] || [ "$DIGEST" = "null" ]; then
  echo "푸시한 이미지의 다이제스트를 읽지 못했다." >&2
  exit 1
fi

echo
echo "푸시됨: $DIGEST"

# 태그가 아니라 태그를 지금 막 가리키게 된 다이제스트로 갱신한다. :latest 로 넘겨도
# Lambda 가 알아서 현재 다이제스트를 고정하지만, 그 사이에 다른 사람이 같은 태그를
# 덮어쓰면 의도하지 않은 이미지가 걸린다.
aws lambda update-function-code --region "$REGION" \
  --function-name "$FUNCTION_NAME" \
  --image-uri "$REPO@$DIGEST" \
  --no-cli-pager --output text --query 'LastUpdateStatus'

# 갱신은 비동기다. InProgress 인 동안 호출하면 옛 코드가 돈다.
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
echo "  * 스키마가 함께 바뀌었다면 앱이 먼저 배포되어 있어야 한다. Lambda 는 photos 를"
echo "    직접 UPDATE 하므로, 없는 컬럼을 쓰면 같은 UPDATE 에 든 임베딩 벡터까지 날아간다."
echo "  * 실행 결과의 failed 가 대상 수와 같으면 임베딩이 아니라 IAM(s3:PutObject) 문제다."
echo "    미리보기 PUT 이 벡터보다 먼저라 그 사진들은 벡터도 없고, 권한을 고친 뒤 다시 부르면"
echo "    fetch_targets 가 자연히 다시 집는다 (force 불필요)."
echo "  * 타임아웃 앞에서 멈추면 자기 자신을 EVENT 재호출한다. 실행 역할에 lambda:InvokeFunction"
echo "    (자기 함수)과, NAT 없는 서브넷이므로 Lambda 인터페이스 VPC 엔드포인트가 있어야 한다."
echo "    없으면 reinvoked=false 로 끝나고 앱이 다시 불러야 이어진다."
