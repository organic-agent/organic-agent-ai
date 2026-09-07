"""연산 장치 — Lambda 는 CPU, SageMaker/EC2 GPU 는 cuda, 로컬 맥은 mps(#68).

러너는 여기서 고른 device 로 가중치를 옮기고, `autocast()` 안에서 forward 한다. cuda 가 아니면 autocast 는 아무것도
안 한다 — CPU 경로(Lambda)의 수치는 이 모듈이 생기기 전과 같다.
"""

from __future__ import annotations

import contextlib
import os

import torch


def pick_device(requested: str | None = None) -> str:
    """"auto"(기본) 면 cuda → mps → cpu. 명시하면 그대로 — 없는 장치를 고르면 torch 가 러너 로드에서 죽는다(일찍 죽는 편이 낫다)."""
    name = (requested or os.environ.get("SCORE_DEVICE") or "auto").lower()
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast(device: str, fp16: bool):
    """cuda + fp16 일 때만 half autocast. 결과 텐서는 러너가 `.float()` 로 되돌린다."""
    if fp16 and device == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def describe(device: str) -> str:
    """로그용 — 어떤 GPU 인지 남긴다."""
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        return f"cuda:{props.name} {props.total_memory / 2**30:.0f}GB"
    return device
