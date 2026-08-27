"""ARNIQA 회귀기 비교 — STATUS.md 1.1 '실사 회귀기 비교'.

ARNIQA는 같은 백본 위에 회귀기를 8종 제공한다. hub 기본값 kadid10k는 **합성 왜곡**(원본 81장에
인위적 왜곡 25종) 데이터셋이고, 우리 사진은 실사다. 실사 왜곡 데이터셋(koniq10k·spaq·clive·flive)
회귀기가 우리 갤러리에서 더 변별력 있는지 잰다.

무엇을 재나
  ① 원점수 분포 — 대역이 좁으면 백분위가 잡음의 순위가 된다 (study/02 step9 [C])
  ② 회귀기 간 순위 상관 — 서로 같은 것을 재면 어느 것을 써도 같고, 다르면 골라야 한다
  ③ 흐림 반응 — 같은 사진을 흐리게 했을 때 얼마나 떨어지나 (study/00 step2). 잡음 바닥은 좌우 반전
  ④ LAION 미학과의 상관 — 기술 품질이 미학과 독립적인가

실행 (약 5분, 첫 실행은 회귀기 다운로드)
    cd organic-agent-ai
    photoselect/scripts/spike/.venv/bin/python photoselect/scripts/arniqa_regressors.py \\
        --gallery "dataset1/류지혜고객님 (2)" --n 60
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from photoselect import gallery as gal  # noqa: E402
from photoselect.analyze.runners.arniqa import ArniqaRunner  # noqa: E402
from photoselect.config import Settings  # noqa: E402

REGRESSORS = ["kadid10k", "koniq10k", "spaq", "clive"]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--regressors", default=",".join(REGRESSORS))
    args = ap.parse_args()
    regs = args.regressors.split(",")

    settings = Settings.from_env()
    refs = gal.load_local(settings.dataset_root, args.gallery)
    rng = np.random.default_rng(0)
    refs = [refs[i] for i in sorted(rng.choice(len(refs), size=min(args.n, len(refs)), replace=False))]
    print(f"갤러리 {args.gallery} · 표본 {len(refs)}장 · 회귀기 {regs}\n")

    # LAION 점수는 analysis.jsonl에서 (이미 있으면)
    d = settings.out_root / args.gallery.replace("/", "__")
    laion = {}
    if (d / "analysis.jsonl").exists():
        for line in (d / "analysis.jsonl").open(encoding="utf-8"):
            r = json.loads(line)
            laion[r["photo_id"]] = r["sub_scores"].get("aesthetic_score")

    tmp = Path(tempfile.mkdtemp(prefix="arniqa-"))
    scores: dict[str, np.ndarray] = {}
    blur_drop: dict[str, list[float]] = {}
    flip_noise: dict[str, list[float]] = {}
    for reg in regs:
        print(f"[{reg}] 로딩 ...", end="", flush=True)
        r = ArniqaRunner(regressor_dataset=reg)
        print(" ok")
        s = []
        for i, ref in enumerate(refs):
            s.append(r.score(ref.path)["technical_score"])
            if i < 12:   # 앞 12장으로 흐림·반전 반응
                img = ImageOps.exif_transpose(Image.open(ref.path)).convert("RGB")
                pb, pf = tmp / "b.jpg", tmp / "f.jpg"
                img.filter(ImageFilter.GaussianBlur(5.0)).save(pb, quality=95)
                ImageOps.mirror(img).save(pf, quality=95)
                blur_drop.setdefault(reg, []).append(s[-1] - r.score(str(pb))["technical_score"])
                flip_noise.setdefault(reg, []).append(abs(s[-1] - r.score(str(pf))["technical_score"]))
        scores[reg] = np.array(s)

    print("\n[A] 원점수 분포 — 대역이 넓을수록 백분위가 의미 있다")
    print(f"  {'회귀기':<10}{'min':>8}{'p25':>8}{'p50':>8}{'p75':>8}{'max':>8}{'IQR':>8}{'std':>8}")
    for reg, s in scores.items():
        q = np.percentile(s, [0, 25, 50, 75, 100])
        print(f"  {reg:<10}" + "".join(f"{v:>8.3f}" for v in q) + f"{q[3] - q[1]:>8.3f}{s.std():>8.3f}")

    print("\n[B] 회귀기 간 순위 상관 (Spearman) — 1이면 같은 것을 잰다")
    print("  " + f"{'':<10}" + "".join(f"{r:>10}" for r in regs))
    for a in regs:
        print("  " + f"{a:<10}" + "".join(f"{spearman(scores[a], scores[b]):>10.2f}" for b in regs))

    print("\n[C] 흐림 반응 vs 잡음 바닥 (앞 12장) — 배수가 클수록 흐림을 확실히 잡는다")
    print(f"  {'회귀기':<10}{'흐림 시 하락':>12}{'반전 잡음':>10}{'배수':>8}")
    for reg in regs:
        bd, fn = np.mean(blur_drop[reg]), np.mean(flip_noise[reg])
        print(f"  {reg:<10}{bd:>12.3f}{fn:>10.3f}{(bd / fn if fn > 1e-9 else float('inf')):>8.1f}")

    if laion:
        print("\n[D] LAION 미학과의 순위 상관 — 낮을수록 독립적인 축")
        la = np.array([laion.get(ref.photo_id, np.nan) for ref in refs])
        ok = ~np.isnan(la)
        for reg in regs:
            print(f"  {reg:<10}{spearman(scores[reg][ok], la[ok]):>8.2f}")

    print("\n읽는 법: [A] IQR·std가 크고 [C] 배수가 크며 [D] 상관이 낮은 회귀기가 우리 용도에 맞다.")
    print("       [B]에서 서로 0.9 이상이면 어느 것을 써도 순위는 같다 — 그때는 대역이 넓은 쪽.")


if __name__ == "__main__":
    main()
