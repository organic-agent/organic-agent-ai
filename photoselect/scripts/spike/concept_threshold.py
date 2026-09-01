"""컨셉 그룹 임계 탐색 — 임베더 임베딩(DINOv3; 2026-08-29 실측은 DINOv2)을 평균연결 계층 클러스터로 묶어 임계별 그룹 수·크기를 본다.

사용: docker exec ... psql ... > emb.tsv (id, display_order, scene, embedding, caption) 후
      python scripts/spike/concept_threshold.py emb.tsv
docs/plan-v2-slim.md §3.2 의 실측 근거.
"""
import collections, json, sys
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

rows = [l.rstrip("\n").split("\t") for l in open(sys.argv[1])]
E = np.array([json.loads(r[3]) for r in rows], dtype=np.float32)
E /= np.linalg.norm(E, axis=1, keepdims=True)
caps, scenes = [r[4] for r in rows], [r[2] for r in rows]
Z = linkage(E, method="average", metric="cosine")
for t in (0.10, 0.15, 0.20, 0.25, 0.30):
    lab = fcluster(Z, t=t, criterion="distance")
    sizes = collections.Counter(lab)
    big = [c for c, _ in sizes.most_common(6)]
    print(f"\n== cosine-dist<={t}: {len(sizes)} groups, top sizes {[sizes[c] for c in big]}, "
          f"singletons {sum(1 for n in sizes.values() if n == 1)}")
    for c in big[:5]:
        idx = [i for i, l in enumerate(lab) if l == c]
        cc = collections.Counter(caps[i] for i in idx).most_common(1)[0][0]
        sc = collections.Counter(scenes[i] for i in idx).most_common(2)
        print(f"  size {len(idx):3d} order {min(idx)}-{max(idx)} scenes {sc} | {cc}")
