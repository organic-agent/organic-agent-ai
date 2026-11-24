"""학습 스크립트(peft)가 남긴 LoRA 가중치를 score 가 읽는 모양으로 싼다(#170).

    .venv/bin/python scripts/pack_face_lora.py <학습 산출물.pt> <버전> [--alpha 16]
        → score/assets/face_values_lora.pt   {"version", "rank", "alpha", "lora", "head"}

peft 의 키(`base_model.model.….layer.3.attention.q_proj.lora_A.default.weight`)는 transformers 버전에 따라 앞머리가 달라진다.
`layer.3.attention.q_proj.lora_A` 로 줄여 저장하면 러너(`runners/face_lora.py`)가 백본의 `layer.3.attention.q_proj.weight` 에 더한다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

OUT = Path(__file__).resolve().parents[1] / "score" / "assets" / "face_values_lora.pt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source")
    ap.add_argument("version")
    ap.add_argument("--alpha", type=int, default=None, help="LoRA alpha. 없으면 2 × rank(학습 스크립트의 값)")
    args = ap.parse_args()
    saved = torch.load(args.source, map_location="cpu", weights_only=True)
    lora = {}
    for name, tensor in saved["lora"].items():
        m = re.search(r"(layer\.\d+\.attention\.[qv]_proj)\.(lora_[AB])\.", name)
        if m is None:
            raise SystemExit(f"모르는 LoRA 키: {name}")
        lora[f"{m.group(1)}.{m.group(2)}"] = tensor.float().contiguous()
    rank = int(saved["rank"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"version": args.version, "rank": rank, "alpha": args.alpha or 2 * rank, "lora": lora,
                "head": {k: v.float().contiguous() for k, v in saved["head"].items()}}, OUT)
    print(f"{OUT} · {args.version} · LoRA 텐서 {len(lora)}개 · {OUT.stat().st_size / 1e6:.1f}MB")


if __name__ == "__main__":
    main()
