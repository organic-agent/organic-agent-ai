"""ap-northeast-2 Bedrock 모델 가용성 확인 (plan.md §6 최우선 확인 사항).

- 온디맨드 Anthropic 모델 목록
- apac 크로스 리전 추론 프로필 목록 (미제공 모델의 대안 — apac.anthropic.…)

AWS 자격 증명 필요. 결과는 tech-stack.md §5 확정에 쓴다.
"""

from __future__ import annotations

import sys

import boto3

REGION = "ap-northeast-2"


def main() -> None:
    client = boto3.client("bedrock", region_name=REGION)

    print(f"# {REGION} Anthropic 온디맨드 모델")
    try:
        models = client.list_foundation_models(byProvider="Anthropic")["modelSummaries"]
    except Exception as e:
        sys.exit(f"조회 실패 (자격 증명·권한 확인): {e}")
    for m in sorted(models, key=lambda m: m["modelId"]):
        on_demand = "ON_DEMAND" in m.get("inferenceTypesSupported", [])
        mark = "o" if on_demand else "x (프로필 필요)"
        print(f"  [{mark}] {m['modelId']}")

    print("\n# apac 크로스 리전 추론 프로필 (anthropic)")
    try:
        profiles = client.list_inference_profiles()["inferenceProfileSummaries"]
        for p in sorted(profiles, key=lambda p: p["inferenceProfileId"]):
            if "anthropic" in p["inferenceProfileId"]:
                print(f"  {p['inferenceProfileId']} ({p['status']})")
    except Exception as e:
        print(f"  프로필 조회 실패: {e}")


if __name__ == "__main__":
    main()
