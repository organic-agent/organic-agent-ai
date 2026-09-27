"""특징 x 의 모양 — 스칼라 11 + 임베딩 1536. 순서가 wes 와의 계약이다(`FEATURE_SPEC = pref-v1`).

값을 만드는 쪽은 `service/features.py`. 여기는 이름·차원·컨테이너만 둔다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FEATURE_SPEC = "pref-v1"

SUBJECTS = ("bride", "groom", "couple", "group")
# [GLOSSARY-1 2026-09-27] cluster_size_rel → burst_size_rel, log_group_share → log_embed_group_share. 이름만 바꿨고 순서는 그대로라
# FEATURE_SPEC(pref-v1)·저장된 w_scalar 와 호환된다 — 이름은 DB에 저장되지 않는다.
SCALAR_NAMES = (
    "technical_pct", "aesthetic_pct", "sharpness_pct",
    "subj_bride", "subj_groom", "subj_couple", "subj_group",
    "log_embed_group_share", "timeline_pos", "burst_size_rel", "is_burst",
)
N_SCALAR = len(SCALAR_NAMES)
N_EMB = 768 * 2


@dataclass
class Features:
    scalar: np.ndarray   # (n, 11)
    emb: np.ndarray      # (n, 1536), 갤러리 평균 제거
    emb_mean: np.ndarray  # (1536,) — wes 가 같은 값을 빼야 하므로 함께 둔다(추론 시 갤러리에서 다시 계산)
