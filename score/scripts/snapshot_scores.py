"""갤러리 점수 스냅샷과 비교 — GPU fp16 점수가 CPU 점수와 같은 모델인지 확인한다(#68).

    python scripts/snapshot_scores.py dump 7 cpu-g7.json        # DB_* 환경변수(터널) 로 photo_analysis 를 내려받는다
    python scripts/snapshot_scores.py compare cpu-g7.json gpu-g7.json

비교: technical·aesthetic·sharpness 의 Spearman ρ · 최대 절대차, subjects·clip_parent 일치율, CLIP 벡터 코사인 최소값,
technical+aesthetic 상위 50 집합 겹침. 판정 기준(계획 §4.1): ρ ≥ 0.99.
"""

from __future__ import annotations

import json
import sys

import numpy as np


def dump(gallery_id: int, path: str) -> None:
    from score import db
    from score.config import Settings

    conn = db.connect(Settings.from_env())
    with conn.cursor() as cur:
        cur.execute(
            "SELECT a.photo_id, a.subjects, a.sub_scores, a.model_version, a.analyzed_at, a.clip_embedding::text "
            "FROM photo_analysis a JOIN photos p ON p.id = a.photo_id "
            "WHERE p.gallery_id = %s AND p.deleted_at IS NULL AND a.model_version IS NOT NULL ORDER BY a.photo_id",
            (gallery_id,),
        )
        rows = cur.fetchall()
    conn.close()
    out = {str(r[0]): {"subjects": r[1], "sub_scores": r[2], "model_version": r[3],
                       "analyzed_at": r[4].isoformat() if r[4] else None,
                       "clip": json.loads(r[5]) if r[5] else None} for r in rows}
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"{len(out)}장 → {path}")


def compare(a_path: str, b_path: str) -> None:
    from scipy.stats import spearmanr

    a, b = json.load(open(a_path)), json.load(open(b_path))
    ids = [i for i in a if i in b and a[i]["sub_scores"] and b[i]["sub_scores"]]
    print(f"공통 {len(ids)}장 (A {len(a)} · B {len(b)})")
    for key in ("technical_score", "aesthetic_score", "sharpness"):
        x = np.array([a[i]["sub_scores"].get(key, np.nan) for i in ids], dtype=float)
        y = np.array([b[i]["sub_scores"].get(key, np.nan) for i in ids], dtype=float)
        ok = ~(np.isnan(x) | np.isnan(y))
        rho = spearmanr(x[ok], y[ok]).correlation
        print(f"  {key:16s} spearman={rho:.4f}  maxabs={np.abs(x[ok] - y[ok]).max():.2e}  mean|d|={np.abs(x[ok] - y[ok]).mean():.2e}")
    same_s = sum(a[i]["subjects"] == b[i]["subjects"] for i in ids)
    same_p = sum(a[i]["sub_scores"].get("clip_parent") == b[i]["sub_scores"].get("clip_parent") for i in ids)
    print(f"  subjects 일치 {same_s}/{len(ids)} ({100 * same_s / len(ids):.1f}%)  clip_parent 일치 {same_p}/{len(ids)} ({100 * same_p / len(ids):.1f}%)")
    with_clip = [i for i in ids if a[i]["clip"] and b[i]["clip"]]
    if with_clip:
        cos = [float(np.dot(a[i]["clip"], b[i]["clip"])) for i in with_clip]
        print(f"  CLIP 코사인 min={min(cos):.5f} mean={np.mean(cos):.5f} ({len(with_clip)}장)")
    for key in ("technical_score", "aesthetic_score"):
        top_a = set(sorted(ids, key=lambda i: -a[i]["sub_scores"].get(key, -1e9))[:50])
        top_b = set(sorted(ids, key=lambda i: -b[i]["sub_scores"].get(key, -1e9))[:50])
        print(f"  {key} 상위 50 겹침 {len(top_a & top_b)}/50")


def main(argv: list[str]) -> None:
    if argv[:1] == ["dump"] and len(argv) == 3:
        dump(int(argv[1]), argv[2])
    elif argv[:1] == ["compare"] and len(argv) == 3:
        compare(argv[1], argv[2])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
