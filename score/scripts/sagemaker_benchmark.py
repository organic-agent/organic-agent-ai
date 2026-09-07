"""SageMaker training job 으로 score GPU 벤치마크를 제출하고 결과를 모은다(#68). 운영 경로가 아니다.

    python scripts/sagemaker_benchmark.py setup                      # 실행 역할 wes-sagemaker-benchmark 만들기 (한 번)
    python scripts/sagemaker_benchmark.py run --gallery-id 1 --force  # 잡 제출 + 대기 + 로그 요약
    python scripts/sagemaker_benchmark.py run --gallery-id 7 --force --instance ml.g4dn.xlarge --no-fp16
    python scripts/sagemaker_benchmark.py logs <job-name>            # 끝난 잡의 로그·타임라인만 다시

DB·S3 접속값과 VPC 는 운영 `wes-score` Lambda 설정에서 그대로 복사한다(lambda:GetFunctionConfiguration) — 같은 RDS·같은
버킷·같은 서브넷/SG 라 SageMaker ENI 가 Lambda 와 똑같이 RDS 에 닿는다. 이미지는 ECR `wes-score:gpu`(build-gpu-image.yml).
비밀번호가 잡 정의(Environment)에 실린다 — DescribeTrainingJob 권한이 있는 사람에게 보인다. 벤치마크 기간만 쓰고 잡을 지운다.

타임라인: CreationTime → TrainingStartTime(컨테이너 시작 = 콜드 기동) → TrainingEndTime. BillableTimeInSeconds 가 과금 초.
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
ROLE_NAME = "wes-sagemaker-benchmark"
LOG_GROUP = "/aws/sagemaker/TrainingJobs"
KEEP_LINES = ("러너 로드", "/장)", "다운로드", "BENCHMARK_RESULT", "완료:", "실패", "Error", "error", "Traceback",
              "SageMaker 벤치마크")


def _session():
    return boto3.session.Session(region_name=REGION)


def lambda_config(s) -> dict:
    return s.client("lambda").get_function_configuration(FunctionName=LAMBDA_NAME)


def image_uri(s) -> str:
    repo = s.client("ecr").describe_repositories(repositoryNames=[ECR_REPO])["repositories"][0]["repositoryUri"]
    return f"{repo}:{IMAGE_TAG}"


def role_arn(s) -> str:
    return s.client("iam").get_role(RoleName=ROLE_NAME)["Role"]["Arn"]


def cmd_setup(args) -> None:
    """SageMaker 실행 역할. 권한은 벤치마크에 필요한 만큼만 — 미리보기 버킷 읽기·결과 쓰기, ECR pull, 로그, VPC ENI."""
    s = _session()
    iam = s.client("iam")
    cfg = lambda_config(s)
    bucket = cfg["Environment"]["Variables"]["S3_BUCKET"]
    account = s.client("sts").get_caller_identity()["Account"]
    trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "sagemaker.amazonaws.com"},
                                                     "Action": "sts:AssumeRole"}]}
    policy = {"Version": "2012-10-17", "Statement": [
        {"Effect": "Allow", "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
         "Resource": [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/*"]},
        {"Effect": "Allow", "Action": ["ecr:GetAuthorizationToken"], "Resource": "*"},
        {"Effect": "Allow", "Action": ["ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage"],
         "Resource": f"arn:aws:ecr:{REGION}:{account}:repository/{ECR_REPO}"},
        {"Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:CreateLogStream", "logs:PutLogEvents", "logs:DescribeLogStreams"],
         "Resource": f"arn:aws:logs:{REGION}:{account}:log-group:{LOG_GROUP}*"},
        {"Effect": "Allow", "Action": ["cloudwatch:PutMetricData"], "Resource": "*"},
        {"Effect": "Allow", "Action": ["ec2:CreateNetworkInterface", "ec2:CreateNetworkInterfacePermission", "ec2:DeleteNetworkInterface",
                                       "ec2:DeleteNetworkInterfacePermission", "ec2:DescribeNetworkInterfaces", "ec2:DescribeVpcs",
                                       "ec2:DescribeDhcpOptions", "ec2:DescribeSubnets", "ec2:DescribeSecurityGroups"],
         "Resource": "*"},
    ]}
    try:
        iam.create_role(RoleName=ROLE_NAME, AssumeRolePolicyDocument=json.dumps(trust),
                        Description="score GPU benchmark (#68) SageMaker training execution role - delete after benchmark")
        print(f"역할 생성: {ROLE_NAME}")
    except iam.exceptions.EntityAlreadyExistsException:
        print(f"역할 있음: {ROLE_NAME}")
    iam.put_role_policy(RoleName=ROLE_NAME, PolicyName="benchmark", PolicyDocument=json.dumps(policy))
    print("정책 반영. 역할 전파에 10초쯤 걸린다.")


def cmd_run(args) -> None:
    s = _session()
    cfg = lambda_config(s)
    env = dict(cfg["Environment"]["Variables"])
    env.pop("CATEGORIZE_FUNCTION_NAME", None)          # 체인 없음 — categorize 는 GPU 와 무관
    env.update({
        "GALLERY_ID": str(args.gallery_id), "FORCE": "1" if args.force else "0",
        "SHARD_PHOTOS": "0",                            # 한 프로세스가 전부
        "CLIP_BATCH": str(args.clip_batch), "ARNIQA_BATCH": str(args.arniqa_batch),
        "SCORE_FP16": "0" if args.no_fp16 else "1", "SCORE_DECODE_WORKERS": str(args.decode_workers),
        "SCORE_DOWNLOAD_WORKERS": str(args.download_workers), "SCORE_DEVICE": args.device,
    })
    if args.limit:
        env["LIMIT"] = str(args.limit)
    vpc = cfg["VpcConfig"]
    bucket = env["S3_BUCKET"]
    label = args.label or f"{args.instance.split('.')[-2]}-{'fp32' if args.no_fp16 else 'fp16'}"
    name = f"wes-score-gpu-g{args.gallery_id}-{label}-{datetime.now(timezone.utc).strftime('%m%d-%H%M%S')}"
    sm = s.client("sagemaker")
    sm.create_training_job(
        TrainingJobName=name,
        AlgorithmSpecification={"TrainingImage": image_uri(s), "TrainingInputMode": "File"},
        RoleArn=role_arn(s),
        OutputDataConfig={"S3OutputPath": f"s3://{bucket}/sagemaker-benchmark/"},
        ResourceConfig={"InstanceType": args.instance, "InstanceCount": 1, "VolumeSizeInGB": 30},
        VpcConfig={"SecurityGroupIds": vpc["SecurityGroupIds"], "Subnets": vpc["SubnetIds"]},
        StoppingCondition={"MaxRuntimeInSeconds": args.max_seconds},
        Environment=env,
        Tags=[{"Key": "purpose", "Value": "gpu-benchmark-68"}],
    )
    print(f"제출: {name}  ({args.instance}, 갤러리 {args.gallery_id}, force={args.force}, fp16={not args.no_fp16})")
    if args.no_wait:
        return
    wait_and_report(s, name)


def wait_and_report(s, name: str) -> None:
    sm = s.client("sagemaker")
    last = None
    while True:
        d = sm.describe_training_job(TrainingJobName=name)
        status, secondary = d["TrainingJobStatus"], d.get("SecondaryStatus")
        if (status, secondary) != last:
            print(f"  {datetime.now().strftime('%H:%M:%S')}  {status} / {secondary}")
            last = (status, secondary)
        if status in ("Completed", "Failed", "Stopped"):
            break
        time.sleep(15)
    report(s, d)


def report(s, d: dict) -> None:
    name = d["TrainingJobName"]
    created, start, end = d["CreationTime"], d.get("TrainingStartTime"), d.get("TrainingEndTime")
    print(f"\n== {name}: {d['TrainingJobStatus']}  {d.get('FailureReason', '')}")
    print(f"  인스턴스        {d['ResourceConfig']['InstanceType']}")
    print(f"  제출 → 컨테이너  {(start - created).total_seconds():.0f}s  (콜드 기동)" if start else "  컨테이너 시작 안 됨")
    if start and end:
        print(f"  컨테이너 → 끝    {(end - start).total_seconds():.0f}s")
        print(f"  제출 → 끝        {(end - created).total_seconds():.0f}s")
    print(f"  과금 초          {d.get('BillableTimeInSeconds', '?')}")
    lines = fetch_logs(s, name)
    print("  -- 로그 --")
    for ln in lines:
        if any(k in ln for k in KEEP_LINES):
            print("  " + ln[:400])
    path = f"sagemaker-{name}.log"
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  전체 로그: {path}")


def fetch_logs(s, name: str) -> list[str]:
    logs = s.client("logs")
    try:
        streams = logs.describe_log_streams(logGroupName=LOG_GROUP, logStreamNamePrefix=name)["logStreams"]
    except logs.exceptions.ResourceNotFoundException:
        return []
    out: list[str] = []
    for st in streams:
        token = None
        while True:
            kw = {"logGroupName": LOG_GROUP, "logStreamName": st["logStreamName"], "startFromHead": True}
            if token:
                kw["nextToken"] = token
            r = logs.get_log_events(**kw)
            out.extend(e["message"].rstrip() for e in r["events"])
            if r.get("nextForwardToken") == token or not r["events"]:
                break
            token = r["nextForwardToken"]
    return out


def cmd_logs(args) -> None:
    s = _session()
    report(s, s.client("sagemaker").describe_training_job(TrainingJobName=args.job_name))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sagemaker_benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    r = sub.add_parser("run")
    r.add_argument("--gallery-id", type=int, required=True)
    r.add_argument("--force", action="store_true")
    r.add_argument("--limit", type=int)
    r.add_argument("--instance", default="ml.g4dn.xlarge")
    r.add_argument("--device", default="auto", help="auto | cuda | cpu (같은 인스턴스에서 CPU 만 재 볼 때)")
    r.add_argument("--no-fp16", action="store_true")
    r.add_argument("--clip-batch", type=int, default=32)
    r.add_argument("--arniqa-batch", type=int, default=8)
    r.add_argument("--decode-workers", type=int, default=4)
    r.add_argument("--download-workers", type=int, default=16)
    r.add_argument("--max-seconds", type=int, default=3600)
    r.add_argument("--label", help="잡 이름에 붙일 짧은 표식")
    r.add_argument("--no-wait", action="store_true")
    lg = sub.add_parser("logs")
    lg.add_argument("job_name")
    args = ap.parse_args(argv)
    {"setup": cmd_setup, "run": cmd_run, "logs": cmd_logs}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
