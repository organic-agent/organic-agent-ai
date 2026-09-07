"""EC2 stop/start 로 score GPU 벤치마크 2단계(#68) — "켜면 스스로 잡을 돌리고 끝나면 스스로 정지" 모양의 실측. 운영 경로가 아니다.

    python scripts/ec2_benchmark.py setup                          # 인스턴스 역할·프로파일 wes-gpu-benchmark (한 번)
    python scripts/ec2_benchmark.py launch --instance g6.xlarge    # DLAMI(AL2023) 인스턴스 생성 → 첫 부팅에 이미지 pull → 스스로 정지
    python scripts/ec2_benchmark.py run --gallery-id 8 --force     # 정지 인스턴스에 태그로 갤러리를 넘기고 StartInstances → 정지될 때까지 타임라인
    python scripts/ec2_benchmark.py logs <run-label>               # S3 에 올라온 로그·타이밍 다시 보기
    python scripts/ec2_benchmark.py terminate                      # 벤치마크 끝 — 인스턴스 삭제

인스턴스는 퍼블릭 서브넷(ECR pull·SSM 용 공인 IP) + Lambda 와 같은 SG(RDS 인바운드가 이미 허용) 에 둔다. 매 부팅마다 systemd 서비스가
인스턴스 태그(GalleryId · RunLabel · Force · AutoStop)를 IMDS 로 읽어 `wes-score:gpu` 컨테이너를 돌리고, 로그·타이밍을 S3 `gpu-benchmark/ec2/<label>/`
에 올린 뒤 `shutdown -h` 한다(InstanceInitiatedShutdownBehavior=stop 이라 정지). DB·S3 접속값은 매 부팅에 운영 Lambda 설정에서 읽는다.

타임라인: StartInstances 호출(t0) → running(t1) → OS 부팅 완료 → 컨테이너 시작 → 잡 끝 → shutdown → stopped(t5). 과금은 t1~stopping.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone

import boto3

REGION = "ap-northeast-2"
LAMBDA_NAME = "wes-score"
ECR_REPO = "wes-score"
IMAGE_TAG = "gpu"
ROLE_NAME = "wes-gpu-benchmark"
NAME_TAG = "wes-gpu-benchmark"
AMI_PARAM = "/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id"
PUBLIC_SUBNET = "subnet-056c2f66b49d6a751"     # wes-public-ap-northeast-2a
S3_PREFIX = "gpu-benchmark/ec2"

RUN_SCRIPT = r'''#!/bin/bash
# wes-score GPU 벤치마크 — 매 부팅마다 systemd 가 부른다(#68). 태그로 무엇을 할지 받는다.
set -u
REGION=ap-northeast-2
T_SERVICE=$(date +%s.%N)
BTIME=$(awk '/btime/ {print $2}' /proc/stat)
TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 600")
md() { curl -sf -H "X-aws-ec2-metadata-token: $TOKEN" "http://169.254.169.254/latest/meta-data/$1"; }
IID=$(md instance-id); ITYPE=$(md instance-type)
GID=$(md tags/instance/GalleryId || true); LABEL=$(md tags/instance/RunLabel || echo boot-$(date +%m%d-%H%M%S))
FORCE=$(md tags/instance/Force || echo 0); AUTOSTOP=$(md tags/instance/AutoStop || echo true); LIMIT=$(md tags/instance/Limit || true)
BUCKET=$(aws lambda get-function-configuration --region $REGION --function-name wes-score --query 'Environment.Variables.S3_BUCKET' --output text)
OUT=/var/log/wes-score-runs/$LABEL; mkdir -p $OUT
echo "label=$LABEL gallery=$GID force=$FORCE autostop=$AUTOSTOP instance=$IID $ITYPE btime=$BTIME service_start=$T_SERVICE" | tee $OUT/timing.txt

REPO=$(aws ecr describe-repositories --region $REGION --repository-names wes-score --query 'repositories[0].repositoryUri' --output text)
if ! docker image inspect $REPO:gpu >/dev/null 2>&1 || [ "${PULL:-0}" = 1 ]; then
  aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin ${REPO%%/*}
  T_PULL0=$(date +%s.%N); docker pull $REPO:gpu; echo "pull_seconds=$(python3 -c "print(round($(date +%s.%N) - $T_PULL0, 1))")" | tee -a $OUT/timing.txt
fi

if [ -n "$GID" ]; then
  aws lambda get-function-configuration --region $REGION --function-name wes-score --query 'Environment.Variables' --output json \
    | python3 -c 'import sys,json; [print(f"{k}={v}") for k,v in json.load(sys.stdin).items() if k!="CATEGORIZE_FUNCTION_NAME"]' > /run/wes-score.env
  chmod 600 /run/wes-score.env
  T_DOCKER0=$(date +%s.%N); echo "docker_start=$T_DOCKER0" | tee -a $OUT/timing.txt
  docker run --rm --gpus all --env-file /run/wes-score.env \
    -e GALLERY_ID=$GID -e FORCE=$FORCE ${LIMIT:+-e LIMIT=$LIMIT} -e SHARD_PHOTOS=0 \
    -e CLIP_BATCH=32 -e ARNIQA_BATCH=8 -e SCORE_FP16=1 -e SCORE_DECODE_WORKERS=4 -e SCORE_DOWNLOAD_WORKERS=16 \
    -e SM_OUTPUT_DATA_DIR=/out -v $OUT:/out $REPO:gpu train > $OUT/container.log 2>&1
  RC=$?; T_DOCKER1=$(date +%s.%N)
  echo "docker_end=$T_DOCKER1 rc=$RC docker_seconds=$(python3 -c "print(round($T_DOCKER1 - $T_DOCKER0, 1))")" | tee -a $OUT/timing.txt
  rm -f /run/wes-score.env
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader >> $OUT/timing.txt 2>&1
echo "service_end=$(date +%s.%N)" >> $OUT/timing.txt
aws s3 cp --region $REGION --recursive $OUT s3://$BUCKET/gpu-benchmark/ec2/$LABEL/ --only-show-errors
# OS halt(shutdown -h) 든 API stop 이든 g6 는 stopped 까지 5~6분(GPU 인스턴스 특성, AWS 문서). 과금은 stopping 진입 시 끝난다.
# OnDone=terminate 면(AMI 에서 매번 새로 띄우는 모양) 삭제 — terminate 는 다음 launch 를 막지 않는다.
ONDONE=$(md tags/instance/OnDone || echo stop)
if [ "$AUTOSTOP" = "true" ]; then
  if [ "$ONDONE" = "terminate" ]; then aws ec2 terminate-instances --region $REGION --instance-ids $IID >/dev/null || shutdown -h now
  else aws ec2 stop-instances --region $REGION --instance-ids $IID >/dev/null || shutdown -h now; fi
fi
'''

USER_DATA = f'''#!/bin/bash
# 첫 부팅: 실행 스크립트·systemd 서비스 설치. 이후 부팅은 서비스가 알아서 (#68)
set -eu
cat > /usr/local/bin/wes-score-gpu-run.sh <<'RUNEOF'
{RUN_SCRIPT}
RUNEOF
chmod 755 /usr/local/bin/wes-score-gpu-run.sh
cat > /etc/systemd/system/wes-score-gpu.service <<'UNITEOF'
[Unit]
Description=wes-score GPU benchmark run (#68)
After=network-online.target docker.service
Wants=network-online.target
Requires=docker.service

[Service]
Type=oneshot
ExecStart=/usr/local/bin/wes-score-gpu-run.sh
TimeoutStartSec=3600

[Install]
WantedBy=multi-user.target
UNITEOF
# DLAMI 부팅 지연 제거(실측 66s → ?): dkms 가 매 부팅 56s 동안 NVIDIA 모듈을 재빌드 검사한다 — 드라이버는 이미 설치돼 있다. update-motd 30s 도 불필요.
systemctl mask dkms.service update-motd.service
systemctl daemon-reload
systemctl enable wes-score-gpu.service
# 첫 부팅에서도 한 번 돈다 — 이미지 pull 뒤 (GalleryId 태그가 없으면) 정지
systemctl start --no-block wes-score-gpu.service
'''


def _s():
    return boto3.session.Session(region_name=REGION)


def _lambda_cfg(s):
    return s.client("lambda").get_function_configuration(FunctionName=LAMBDA_NAME)


def _find_instance(s) -> dict | None:
    r = s.client("ec2").describe_instances(Filters=[{"Name": "tag:Name", "Values": [NAME_TAG]},
                                                    {"Name": "instance-state-name", "Values": ["pending", "running", "stopping", "stopped"]}])
    inst = [i for res in r["Reservations"] for i in res["Instances"]]
    return inst[0] if inst else None


def cmd_setup(args) -> None:
    s = _s()
    iam = s.client("iam")
    account = s.client("sts").get_caller_identity()["Account"]
    bucket = _lambda_cfg(s)["Environment"]["Variables"]["S3_BUCKET"]
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "ec2.amazonaws.com"}, "Action": "sts:AssumeRole"}]}
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"], "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"]},
        {"Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": "*"},
        {"Effect": "Allow", "Action": ["ecr:DescribeRepositories", "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
         "Resource": f"arn:aws:ecr:{REGION}:{account}:repository/{ECR_REPO}"},
        {"Effect": "Allow", "Action": ["lambda:GetFunctionConfiguration"], "Resource": f"arn:aws:lambda:{REGION}:{account}:function:{LAMBDA_NAME}"},
        # 자기 자신만 정지 — Name 태그로 묶는다
        {"Effect": "Allow", "Action": ["ec2:StopInstances", "ec2:TerminateInstances"], "Resource": f"arn:aws:ec2:{REGION}:{account}:instance/*",
         "Condition": {"StringEquals": {"aws:ResourceTag/Name": NAME_TAG}}},
    ]}
    try:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                        Description="score GPU benchmark (#68) EC2 instance role - delete after benchmark")
        print(f"역할 생성: {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"역할 있음: {ROLE_NAME}")
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="benchmark", PolicyDocument=json.dumps(policy))
    iam.attach_role_policy(RoleName=ROLE_NAME, PolicyArn="arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore")
    try:
        iam.create_instance_profile(InstanceProfileName=ROLE_NAME)
        iam.add_role_to_instance_profile(InstanceProfileName=ROLE_NAME, RoleName=ROLE_NAME)
        print(f"인스턴스 프로파일 생성: {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"인스턴스 프로파일 있음: {ROLE_NAME}")
    print("반영에 10초쯤 걸린다.")


def cmd_launch(args) -> None:
    s = _s()
    if _find_instance(s):
        sys.exit("이미 있다 — terminate 먼저")
    ec2 = s.client("ec2")
    ami = s.client("ssm").get_parameter(Name=AMI_PARAM)["Parameter"]["Value"]
    sg = _lambda_cfg(s)["VpcConfig"]["SecurityGroupIds"]
    r = ec2.run_instances(
        ImageId=ami, InstanceType=args.instance, MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": ROLE_NAME},
        UserData=USER_DATA,
        InstanceInitiatedShutdownBehavior="stop",
        MetadataOptions={"HttpTokens": "required", "InstanceMetadataTags": "enabled"},
        BlockDeviceMappings=[{"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 100, "VolumeType": "gp3", "DeleteOnTermination": True}}],
        NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": PUBLIC_SUBNET, "Groups": sg, "AssociatePublicIpAddress": True}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": [{"Key": "Name", "Value": NAME_TAG}, {"Key": "purpose", "Value": "gpu-benchmark-68"},
                                                                  {"Key": "AutoStop", "Value": "true"}]}],
    )
    iid = r["Instances"][0]["InstanceId"]
    print(f"생성: {iid} ({args.instance}, AMI {ami}). 첫 부팅에서 이미지를 받고 스스로 정지한다 — 정지될 때까지 기다린다.")
    _wait_state(ec2, iid, "stopped", timeout=1200)
    print("정지됨. run 으로 측정 시작.")


def _wait_state(ec2, iid: str, target: str, timeout: int = 900, poll: float = 5.0) -> float:
    t0 = time.time()
    last = None
    while time.time() - t0 < timeout:
        st = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
        if st != last:
            print(f"  {datetime.now().strftime('%H:%M:%S')}  {st}")
            last = st
        if st == target:
            return time.time()
        time.sleep(poll)
    raise TimeoutError(f"{iid}: {target} 안 됨 ({timeout}s)")


def cmd_run(args) -> None:
    s = _s()
    ec2 = s.client("ec2")
    inst = _find_instance(s)
    if not inst or inst["State"]["Name"] != "stopped":
        sys.exit("정지된 벤치마크 인스턴스가 없다 (launch 먼저, 또는 정지 대기)")
    iid = inst["InstanceId"]
    label = f"g{args.gallery_id}-{args.label or inst['InstanceType']}-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}"
    tags = [{"Key": "GalleryId", "Value": str(args.gallery_id)}, {"Key": "RunLabel", "Value": label},
            {"Key": "Force", "Value": "1" if args.force else "0"}, {"Key": "AutoStop", "Value": "true"}]
    if args.limit:
        tags.append({"Key": "Limit", "Value": str(args.limit)})
    ec2.create_tags(Resources=[iid], Tags=tags)
    t0 = time.time()
    ec2.start_instances(InstanceIds=[iid])
    print(f"StartInstances {iid} → {label}  ({datetime.now().strftime('%H:%M:%S')})")
    t_running = _wait_state(ec2, iid, "running", timeout=300, poll=2)
    t_stopped = _wait_state(ec2, iid, "stopped", timeout=args.max_seconds, poll=5)
    ec2.delete_tags(Resources=[iid], Tags=[{"Key": "GalleryId"}, {"Key": "Limit"}])
    print(f"\n== {label}: start→running {t_running - t0:.0f}s · start→stopped {t_stopped - t0:.0f}s")
    report(s, label, t0=t0, t_running=t_running, t_stopped=t_stopped)


def report(s, label: str, t0: float | None = None, t_running: float | None = None, t_stopped: float | None = None) -> None:
    bucket = _lambda_cfg(s)["Environment"]["Variables"]["S3_BUCKET"]
    s3 = s.client("s3")
    key = f"{S3_PREFIX}/{label}/"
    try:
        timing = s3.get_object(Bucket=bucket, Key=key + "timing.txt")["Body"].read().decode()
    except s3.exceptions.NoSuchKey:
        print("  S3 에 타이밍이 없다 — 컨테이너가 안 돌았거나 아직 업로드 전")
        return
    kv = {}
    for line in timing.splitlines():
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                kv[k] = v
    print("  -- 타이밍 --")
    btime = float(kv.get("btime", 0))
    svc = float(kv.get("service_start", 0))
    d0, d1 = float(kv.get("docker_start", 0)), float(kv.get("docker_end", 0))
    if t0:
        print(f"  StartInstances → running        {t_running - t0:6.0f}s")
        print(f"  StartInstances → 커널 부팅        {btime - t0:6.0f}s")
        print(f"  StartInstances → 서비스 시작      {svc - t0:6.0f}s")
        print(f"  StartInstances → 컨테이너 시작    {d0 - t0:6.0f}s   ← 콜드 기동")
        print(f"  컨테이너 (다운로드·로드·점수)      {d1 - d0:6.0f}s")
        print(f"  컨테이너 끝 → stopped/terminate   {t_stopped - d1:6.0f}s")
        print(f"  StartInstances → stopped         {t_stopped - t0:6.0f}s   ← 제출→끝")
        print(f"  과금 초(running → stopped 근사)   {t_stopped - t_running:6.0f}s")
    print("  " + timing.replace("\n", "\n  "))
    try:
        log = s3.get_object(Bucket=bucket, Key=key + "container.log")["Body"].read().decode()
        for ln in log.splitlines():
            if any(k in ln for k in ("러너 로드", "다운로드", "BENCHMARK_RESULT", "완료:", "실패", "Traceback", "Error")):
                print("  " + ln[:600])
        with open(f"ec2-{label}.log", "w") as f:
            f.write(timing + "\n" + log)
        print(f"  전체 로그: ec2-{label}.log")
    except s3.exceptions.NoSuchKey:
        print("  container.log 없음")


AMI_NAME = "wes-score-gpu-benchmark"


def cmd_bake(args) -> None:
    """정지된 벤치마크 인스턴스(이미지 pull · 서비스 · mask 완료)를 AMI 로 굽는다 — run-fresh 가 여기서 띄운다."""
    s = _s()
    ec2 = s.client("ec2")
    inst = _find_instance(s)
    if not inst or inst["State"]["Name"] != "stopped":
        sys.exit("정지된 벤치마크 인스턴스가 필요하다")
    name = f"{AMI_NAME}-{datetime.now(timezone.utc).strftime('%m%d-%H%M')}"
    r = ec2.create_image(InstanceId=inst["InstanceId"], Name=name, Description="score GPU benchmark (#68) - image pre-pulled, dkms masked",
                         TagSpecifications=[{"ResourceType": "image", "Tags": [{"Key": "Name", "Value": AMI_NAME}, {"Key": "purpose", "Value": "gpu-benchmark-68"}]}])
    ami = r["ImageId"]
    print(f"AMI 생성 중: {ami} ({name})")
    t0 = time.time()
    ec2.get_waiter("image_available").wait(ImageIds=[ami], WaiterConfig={"Delay": 15, "MaxAttempts": 80})
    print(f"AMI 사용 가능 ({time.time() - t0:.0f}s): {ami}")


def _latest_ami(s) -> str:
    imgs = s.client("ec2").describe_images(Owners=["self"], Filters=[{"Name": "tag:Name", "Values": [AMI_NAME]}, {"Name": "state", "Values": ["available"]}])["Images"]
    if not imgs:
        sys.exit("AMI 가 없다 — bake 먼저")
    return sorted(imgs, key=lambda i: i["CreationDate"])[-1]["ImageId"]


def cmd_run_fresh(args) -> None:
    """AMI 에서 새 인스턴스를 띄워 잡을 돌리고 terminate — 정지 대기가 없는 모양. 타임라인은 RunInstances 호출부터."""
    s = _s()
    ec2 = s.client("ec2")
    ami = _latest_ami(s)
    sg = _lambda_cfg(s)["VpcConfig"]["SecurityGroupIds"]
    label = f"g{args.gallery_id}-{args.label or 'fresh'}-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}"
    tags = [{"Key": "Name", "Value": NAME_TAG + "-fresh"}, {"Key": "purpose", "Value": "gpu-benchmark-68"},
            {"Key": "GalleryId", "Value": str(args.gallery_id)}, {"Key": "RunLabel", "Value": label},
            {"Key": "Force", "Value": "1" if args.force else "0"}, {"Key": "AutoStop", "Value": "true"}, {"Key": "OnDone", "Value": "terminate"}]
    if args.limit:
        tags.append({"Key": "Limit", "Value": str(args.limit)})
    t0 = time.time()
    r = ec2.run_instances(
        ImageId=ami, InstanceType=args.instance, MinCount=1, MaxCount=1,
        IamInstanceProfile={"Name": ROLE_NAME},
        InstanceInitiatedShutdownBehavior="terminate",
        MetadataOptions={"HttpTokens": "required", "InstanceMetadataTags": "enabled"},
        NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": PUBLIC_SUBNET, "Groups": sg, "AssociatePublicIpAddress": True}],
        TagSpecifications=[{"ResourceType": "instance", "Tags": tags}],
    )
    iid = r["Instances"][0]["InstanceId"]
    print(f"RunInstances {iid} ({args.instance}, {ami}) → {label}  ({datetime.now().strftime('%H:%M:%S')})")
    t_running = _wait_state(ec2, iid, "running", timeout=300, poll=2)
    # 끝나면 스스로 terminate → shutting-down
    t_end = None
    last = None
    while time.time() - t0 < args.max_seconds:
        st = ec2.describe_instances(InstanceIds=[iid])["Reservations"][0]["Instances"][0]["State"]["Name"]
        if st != last:
            print(f"  {datetime.now().strftime('%H:%M:%S')}  {st}")
            last = st
        if st in ("shutting-down", "terminated", "stopping", "stopped"):
            t_end = time.time()
            if st in ("stopping", "stopped"):      # AMI 의 스크립트가 stop 을 불렀으면 여기서 지운다 — 타임라인엔 영향 없음
                ec2.terminate_instances(InstanceIds=[iid])
            break
        time.sleep(5)
    print(f"\n== {label}: launch→running {t_running - t0:.0f}s · launch→terminate 호출 {(t_end or time.time()) - t0:.0f}s")
    report(s, label, t0=t0, t_running=t_running, t_stopped=t_end or time.time())


def cmd_logs(args) -> None:
    report(_s(), args.label)


def cmd_terminate(args) -> None:
    s = _s()
    inst = _find_instance(s)
    if not inst:
        print("인스턴스 없음")
        return
    s.client("ec2").terminate_instances(InstanceIds=[inst["InstanceId"]])
    print(f"삭제: {inst['InstanceId']}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="ec2_benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    l = sub.add_parser("launch")
    l.add_argument("--instance", default="g6.xlarge")
    r = sub.add_parser("run")
    r.add_argument("--gallery-id", type=int, required=True)
    r.add_argument("--force", action="store_true")
    r.add_argument("--limit", type=int)
    r.add_argument("--label")
    r.add_argument("--max-seconds", type=int, default=1800)
    sub.add_parser("bake")
    rf = sub.add_parser("run-fresh")
    rf.add_argument("--gallery-id", type=int, required=True)
    rf.add_argument("--force", action="store_true")
    rf.add_argument("--limit", type=int)
    rf.add_argument("--instance", default="g6.xlarge")
    rf.add_argument("--label")
    rf.add_argument("--max-seconds", type=int, default=1800)
    lg = sub.add_parser("logs")
    lg.add_argument("label")
    sub.add_parser("terminate")
    args = ap.parse_args(argv)
    {"setup": cmd_setup, "launch": cmd_launch, "run": cmd_run, "bake": cmd_bake, "run-fresh": cmd_run_fresh,
     "logs": cmd_logs, "terminate": cmd_terminate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
