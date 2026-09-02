# Lambda 컨테이너 이미지. zip 패키지는 압축 해제 250MB 한도라 torch 하나로 넘는다.
FROM public.ecr.aws/lambda/python:3.12

# CPU 전용 휠. 기본 PyPI에서 받으면 CUDA 런타임이 함께 들어와 이미지가 GB 단위로 불어나는데,
# Lambda에는 GPU가 없어서 전부 죽은 무게다. transformers가 이 인덱스에 없으므로 분리한다.
#
# torchvision도 필요하다. 최신 transformers의 AutoImageProcessor는 fast 이미지 프로세서를
# 기본으로 쓰고 그게 torchvision 백엔드를 요구해서, 없으면 모델을 받는 단계에서 ImportError로
# 죽는다("requires the Torchvision library"). torch와 같은 인덱스로 받아 버전을 맞춘다.
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# RDS의 CA 번들. IAM 인증은 sslmode=verify-full로 붙으므로 이 파일이 없으면 접속 자체가
# 실패한다. 런타임에 받아올 수는 없다 -- 이 함수가 붙는 서브넷에는 인터넷 경로가 없다.
RUN mkdir -p /opt/rds-ca \
 && curl -fsSL -o /opt/rds-ca/global-bundle.pem \
      https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

# 가중치를 빌드 타임에 굽는다. 실행 시점의 Lambda는 NAT 없는 프라이빗 서브넷에 있어
# huggingface.co에 나갈 수 없다. 여기서 받아두지 않으면 첫 호출이 네트워크 타임아웃으로 죽는다.
#
# DINOv3는 게이트 모델이라 라이선스를 승인한 계정의 토큰이 있어야 받아진다. 토큰은 ENV/ARG가
# 아니라 빌드 시크릿으로 넘긴다 -- ARG는 이미지 히스토리에 남는다. deploy.sh 가
# `--secret id=hf_token`으로 넣어 주고, 이 RUN 한 줄 밖에서는 파일이 존재하지 않는다.
ENV HF_HOME=/opt/hf
ARG EMBED_MODEL_ID=facebook/dinov3-vitb16-pretrain-lvd1689m
RUN --mount=type=secret,id=hf_token,required=true \
    HF_TOKEN="$(cat /run/secrets/hf_token)" \
    python -c "from transformers import AutoImageProcessor, AutoModel; \
m = '${EMBED_MODEL_ID}'; \
AutoImageProcessor.from_pretrained(m); AutoModel.from_pretrained(m)"

# 런타임 파일시스템은 /tmp 말고는 읽기 전용이다. 캐시를 다시 쓰려 하지 않도록 오프라인으로 못박는다.
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

COPY embedder/ ${LAMBDA_TASK_ROOT}/embedder/

CMD ["embedder.handler.handler"]
