# score Lambda 컨테이너 이미지. zip 은 압축 해제 250MB 한도라 torch 하나로 넘는다. embedder/Dockerfile 과 같은 뼈대.
FROM public.ecr.aws/lambda/python:3.12

# CPU 전용 휠. 기본 PyPI 에서 받으면 CUDA 런타임이 함께 들어와 이미지가 GB 단위로 불어난다.
# requirements.txt 의 torch 핀은 로컬(맥)용이라 여기서는 걸러 낸다.
RUN pip install --no-cache-dir torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN grep -v -E '^(torch|torchvision)==' requirements.txt > /tmp/req.txt && pip install --no-cache-dir -r /tmp/req.txt

# RDS 의 CA 번들. 런타임에 받아올 수 없다 -- 이 함수가 붙는 서브넷에는 인터넷 경로가 없다.
RUN mkdir -p /opt/rds-ca \
 && curl -fsSL -o /opt/rds-ca/global-bundle.pem \
      https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

COPY score/ ${LAMBDA_TASK_ROOT}/score/

# 가중치를 빌드 타임에 굽는다. NAT 없는 프라이빗 서브넷이라 런타임 다운로드는 전부 실패한다.
#   · CLIP ViT-L/14 (openai)  → open_clip 캐시 ($HOME/.cache/clip)
#   · LAION 미학 MLP           → score/weights/ (infrastructure/runners/common.py fetch_weight)
#   · ARNIQA (hub 코드 + spaq 회귀기) → $TORCH_HOME/hub — arniqa.HUB_REPO 의 커밋 SHA 와 같은 ref
# 런타임 파일시스템은 /tmp 말고 읽기 전용이라 캐시 위치를 /opt 아래로 못박고 같은 값을 ENV 로 남긴다.
ENV HOME=/opt/home \
    TORCH_HOME=/opt/torch
RUN mkdir -p /opt/home /opt/torch \
 && cd ${LAMBDA_TASK_ROOT} \
 && python -c "from score.infrastructure.runners import LaionRunner, ArniqaRunner; LaionRunner(); ArniqaRunner()"

CMD ["score.controller.handler.handler"]
