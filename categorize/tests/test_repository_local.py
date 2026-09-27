"""LocalStore 왕복 · 컬럼 경계 — CATEGORIZE 의 필드만 덮고 score 의 것은 그대로."""

from __future__ import annotations

import json

import numpy as np

from categorize.config.settings import PIPELINE_VERSION
from categorize.domain.analysis import ConceptAssignment, PhotoAnalysis
from categorize.repository.local import LocalStore
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

    a = [ConceptAssignment(embed_group_id=0, concept_name="야외 자연", detail_name="해변",
                           confidence=0.9, assigned_by="vlm", clip_concept_name="야외 자연")]
    store.write_assignments("g", None, a)
    assert store.read_assignments("g") == a
    # [GLOSSARY-2 2026-09-27] 캐시 파일의 키 = 필드 이름 = DB 컬럼 이름(wes V23)
    saved = json.loads((tmp_path / "out" / "v3" / "g" / "assignments.jsonl").read_text(encoding="utf-8"))
    assert (saved["concept_name"], saved["detail_name"], saved["clip_concept_name"]) == ("야외 자연", "해변", "야외 자연")
    assert "proposed_concept_name" in saved


def test_local_store_reads_cache_written_with_old_names(tmp_path):
    """V23 전에 쓴 out/ 파일 — 옛 배정 키(parent_name 이 1층, concept_name 이 2층)와 옛 분석 키(cluster_id·model_version·clip_parent)."""
    store = LocalStore(tmp_path / "out")
    d = tmp_path / "out" / "v3" / "g"
    d.mkdir(parents=True)
    (d / "assignments.jsonl").write_text(json.dumps({
        "job_id": None, "embed_group_id": 1, "parent_name": "야외 자연", "concept_name": "해변", "confidence": 0.9,
        "assigned_by": "vlm", "proposed_parent": None, "clip_parent": "야외 자연", "needs_review": False,
    }, ensure_ascii=False) + "\n", encoding="utf-8")
    (d / "analysis.jsonl").write_text(json.dumps({
        "photo_id": "a", "cluster_id": 3, "cluster_rank": 1, "model_version": "v", "sub_scores": {"clip_parent": "한옥·전통"},
    }, ensure_ascii=False) + "\n", encoding="utf-8")

    (a,) = store.read_assignments("g")
    assert (a.concept_name, a.detail_name, a.clip_concept_name) == ("야외 자연", "해변", "야외 자연")
    (r,) = store.read_analysis("g")
    assert (r.burst_id, r.burst_rank, r.pipeline_version) == (3, 1, "v")
    assert r.sub_scores == {"clip_concept_name": "한옥·전통"}


def test_write_groups_keeps_score_columns(tmp_path):
    store, rows, *_ = world(tmp_path)
    grouped = [PhotoAnalysis(photo_id=r.photo_id, subjects="unknown", technical_pct=1.0,
                             burst_id=99, embed_group_id=7, pipeline_version="") for r in rows]
    store.write_groups("g", grouped)
    after = {r.photo_id: r for r in store.read_analysis("g")}
    # subjects · pipeline_version 은 score 의 것 — 건드리지 않는다
    assert all(after[r.photo_id].subjects == "couple" and after[r.photo_id].pipeline_version == PIPELINE_VERSION for r in rows)
    assert all(after[r.photo_id].embed_group_id == 7 and after[r.photo_id].burst_id == 99 for r in rows)
