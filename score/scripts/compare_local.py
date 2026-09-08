"""CPU 경로 회귀 검사 — 두 코드 버전이 로컬 데이터셋 몇 장에 같은 점수를 내는지(비트 동일) 비교한다(#68·#89).

GPU 최적화(fp16·배치·프리페치)를 넣을 때마다 "Lambda(CPU) 경로의 수치는 변하지 않았다"를 이걸로 확인했다.
DB 없이 `python -m score --local` 을 두 번 돌리고 `analysis.jsonl` + CLIP 벡터를 비교한다.

    # 1) 기준 코드(main)와 지금 작업 트리를 각각 16장 점수 → 비교. 기준은 임시 git worktree 로 체크아웃한다.
    python scripts/compare_local.py run "dataset1/데이터셋1" --out /tmp/cmp/head
    python scripts/compare_local.py run "dataset1/데이터셋1" --out /tmp/cmp/main --rev main
    python scripts/compare_local.py compare /tmp/cmp/main /tmp/cmp/head          # 0 차이면 exit 0

    # 2) 한 번에: --rev 와 --out 둘을 주면 run 두 번 + compare
    python scripts/compare_local.py check "dataset1/데이터셋1" --rev main --limit 16

손잡이는 CPU Lambda 기본값(SCORE_DEVICE=cpu, fp16 무시, CLIP 8 · ARNIQA 1 배치, 디코드 스레드 0)으로 고정한다 — 비교 대상은
"같은 손잡이에서 코드가 같은 답을 내는가"다. 허용 오차 기본 0(비트 동일). MPS·GPU 를 볼 때는 `--tol 1e-5` 정도.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

SCORE_ROOT = Path(__file__).resolve().parents[1]           # …/score
REPO_ROOT = SCORE_ROOT.parent
NUMERIC = ("technical_score", "aesthetic_score", "sharpness", "subjects_margin")

#: CPU Lambda 와 같은 손잡이. 환경에 남아 있을 GPU 값이 섞이지 않게 전부 명시한다.
CPU_ENV = {"SCORE_DEVICE": "cpu", "SCORE_FP16": "0", "CLIP_BATCH": "8", "ARNIQA_BATCH": "1", "SCORE_DECODE_WORKERS": "0",
           "ARNIQA_LONG_EDGE": "1024"}


def _gallery_dir(out: Path, gallery: str) -> Path:
    return out / "v3" / gallery.replace("/", "__")


def run(gallery: str, out: Path, limit: int, rev: str | None, dataset_root: Path | None) -> None:
    """`python -m score --local` 을 out 아래에 쓴다. rev 를 주면 그 커밋을 임시 worktree 로 꺼내 그 코드로 돈다(같은 venv)."""
    out = out.resolve()
    if out.exists():
        import shutil
        shutil.rmtree(out)
    env = dict(os.environ, **CPU_ENV, SCORE_OUT=str(out))
    # worktree 에서 돌면 config 의 상대 기본값(../../dataset)이 임시 경로를 가리키므로 항상 명시한다.
    env["SCORE_DATASET"] = str(Path(dataset_root).resolve() if dataset_root else REPO_ROOT.parent / "dataset")
    code_root = SCORE_ROOT
    worktree: Path | None = None
    try:
        if rev:
            worktree = Path(tempfile.mkdtemp(prefix="score-cmp-"))
            subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "add", "--detach", str(worktree), rev],
                           check=True, capture_output=True)
            code_root = worktree / "score"
            # 가중치 캐시는 공유(다시 받지 않게). weights/ 가 코드 옆에 있으면 그것을 쓰는 러너를 위해 심볼릭 링크.
            if (SCORE_ROOT / "weights").exists() and not (code_root / "weights").exists():
                os.symlink(SCORE_ROOT / "weights", code_root / "weights")
        # PYTHONPATH 가 editable 설치보다 앞이라 worktree 의 score 패키지가 import 된다.
        env["PYTHONPATH"] = str(code_root) + os.pathsep + env.get("PYTHONPATH", "")
        label = rev or "working tree"
        print(f"[run] {label} → {out}  ({gallery}, {limit}장, CPU 손잡이)")
        subprocess.run([sys.executable, "-m", "score", "--local", gallery, "--limit", str(limit), "--force"],
                       cwd=str(code_root), env=env, check=True)
    finally:
        if worktree is not None:
            subprocess.run(["git", "-C", str(REPO_ROOT), "worktree", "remove", "--force", str(worktree)],
                           capture_output=True)
    print(f"[run] 결과: {_gallery_dir(out, gallery)}")


def _load(out: Path, gallery: str) -> tuple[dict[str, dict], dict[str, np.ndarray]]:
    d = _gallery_dir(out, gallery)
    if not (d / "analysis.jsonl").exists():
        raise SystemExit(f"결과가 없다: {d}/analysis.jsonl (run 먼저)")
    rows = {}
    with (d / "analysis.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                rows[r["photo_id"]] = r
    clips: dict[str, np.ndarray] = {}
    if (d / "clip_embeddings.npy").exists():
        ids = json.loads((d / "clip_embeddings_ids.json").read_text(encoding="utf-8"))
        clips = dict(zip(ids, np.load(d / "clip_embeddings.npy")))
    return rows, clips


def compare(a: Path, b: Path, gallery: str, tol: float) -> bool:
    """두 결과의 수치·라벨·CLIP 벡터를 비교해 표를 찍고, 전부 tol 안이면 True."""
    ra, ca = _load(a, gallery)
    rb, cb = _load(b, gallery)
    ids = sorted(set(ra) & set(rb))
    only = (set(ra) ^ set(rb))
    print(f"[compare] 공통 {len(ids)}장" + (f", 한쪽에만 {sorted(only)}" if only else ""))
    ok = not only and bool(ids)
    for key in NUMERIC:
        pairs = [(ra[i]["sub_scores"].get(key), rb[i]["sub_scores"].get(key)) for i in ids]
        pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        diff = max(abs(float(x) - float(y)) for x, y in pairs)
        same = diff <= tol
        ok &= same
        print(f"  {key:16s} n={len(pairs):3d} max|Δ|={diff:.3e} {'OK' if same else 'DIFF'}")
    for key in ("subjects", "clip_parent"):
        va = [ra[i].get(key) if key == "subjects" else ra[i]["sub_scores"].get(key) for i in ids]
        vb = [rb[i].get(key) if key == "subjects" else rb[i]["sub_scores"].get(key) for i in ids]
        mism = sum(1 for x, y in zip(va, vb) if x != y)
        ok &= mism == 0
        print(f"  {key:16s} n={len(ids):3d} 불일치={mism} {'OK' if mism == 0 else 'DIFF'}")
    cids = [i for i in ids if i in ca and i in cb]
    if cids:
        diff = max(float(np.max(np.abs(ca[i].astype(np.float64) - cb[i].astype(np.float64)))) for i in cids)
        cos = min(float(np.dot(ca[i], cb[i]) / (np.linalg.norm(ca[i]) * np.linalg.norm(cb[i]) + 1e-12)) for i in cids)
        same = diff <= tol
        ok &= same
        print(f"  {'clip_embedding':16s} n={len(cids):3d} max|Δ|={diff:.3e} min cos={cos:.6f} {'OK' if same else 'DIFF'}")
    mv = {ra[i]["model_version"] for i in ids} | {rb[i]["model_version"] for i in ids}
    print(f"  model_version    {sorted(mv)}")
    print(f"[compare] {'비트 동일' if ok and tol == 0 else ('통과' if ok else '차이 있음')} (tol={tol:g})")
    return ok


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="compare_local", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="한 버전으로 점수 → --out")
    r.add_argument("gallery")
    r.add_argument("--out", required=True, type=Path)
    r.add_argument("--limit", type=int, default=16)
    r.add_argument("--rev", help="git 커밋/브랜치 — 임시 worktree 로 그 코드를 돌린다. 없으면 작업 트리")
    r.add_argument("--dataset", type=Path, help="데이터셋 루트(기본 ../dataset)")
    c = sub.add_parser("compare", help="두 결과 디렉토리 비교")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    c.add_argument("--gallery", required=True)
    c.add_argument("--tol", type=float, default=0.0)
    k = sub.add_parser("check", help="run(기준 rev) + run(작업 트리) + compare")
    k.add_argument("gallery")
    k.add_argument("--rev", default="main")
    k.add_argument("--limit", type=int, default=16)
    k.add_argument("--tol", type=float, default=0.0)
    k.add_argument("--out", type=Path, help="결과를 남길 곳(기본 임시 디렉토리)")
    k.add_argument("--dataset", type=Path)
    args = ap.parse_args(argv)

    if args.cmd == "run":
        run(args.gallery, args.out, args.limit, args.rev, args.dataset)
        return 0
    if args.cmd == "compare":
        return 0 if compare(args.a, args.b, args.gallery, args.tol) else 1
    base = args.out or Path(tempfile.mkdtemp(prefix="score-cmp-"))
    run(args.gallery, base / "base", args.limit, args.rev, args.dataset)
    run(args.gallery, base / "head", args.limit, None, args.dataset)
    return 0 if compare(base / "base", base / "head", args.gallery, args.tol) else 1


if __name__ == "__main__":
    sys.exit(main())
