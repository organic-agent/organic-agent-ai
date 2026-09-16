# categorize Lambda 컨테이너 이미지. torch 가 없어 zip 으로도 갈 수 있지만, embedder·score 와 같은 배포 경로(ECR)를 쓴다.
FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# RDS 의 CA 번들. 런타임에 받아올 수 없다 -- 이 함수가 붙는 서브넷에는 인터넷 경로가 없다.
# Bedrock 도 같은 이유로 bedrock-runtime 인터페이스 VPC 엔드포인트가 있어야 닿는다.
RUN mkdir -p /opt/rds-ca \
 && curl -fsSL -o /opt/rds-ca/global-bundle.pem \
      https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

COPY categorize/ ${LAMBDA_TASK_ROOT}/categorize/

CMD ["categorize.controller.handler.handler"]
