"""LocalStore 왕복 · 컬럼 경계 — CATEGORIZE 의 필드만 덮고 score 의 것은 그대로."""

from __future__ import annotations

import numpy as np

from categorize.config.settings import MODEL_VERSION
from categorize.domain.analysis import ConceptAssignment, PhotoAnalysis
from tests.helpers import world


def test_local_store_roundtrip(tmp_path):
    store, rows, E, C, _ = world(tmp_path)
    got = store.read_analysis("g")
    assert [r.photo_id for r in got] == [r.photo_id for r in rows]
    assert all(r.embed_group_id >= 0 for r in got)
    ids, E2 = store.read_embeddings("g")
    cids, C2 = store.read_clip_embeddings("g")
    assert ids == cids == [r.photo_id for r in rows]
    np.testing.assert_allclose(E2, E, atol=1e-6)
    np.testing.assert_allclose(C2, C, atol=1e-6)

    a = [ConceptAssignment(embed_group_id=0, parent_name="야외 자연", concept_name="해변",
                           confidence=0.9, assigned_by="vlm")]
    store.write_assignments("g", None, a)
    assert store.read_assignments("g") == a


def test_write_groups_keeps_score_columns(tmp_path):
    store, rows, *_ = world(tmp_path)
    grouped = [PhotoAnalysis(photo_id=r.photo_id, subjects="unknown", technical_pct=1.0,
                             cluster_id=99, embed_group_id=7, model_version="") for r in rows]
    store.write_groups("g", grouped)
    after = {r.photo_id: r for r in store.read_analysis("g")}
    # subjects · model_version 은 score 의 것 — 건드리지 않는다
    assert all(after[r.photo_id].subjects == "couple" and after[r.photo_id].model_version == MODEL_VERSION for r in rows)
    assert all(after[r.photo_id].embed_group_id == 7 and after[r.photo_id].cluster_id == 99 for r in rows)
