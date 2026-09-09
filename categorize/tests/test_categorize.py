"""categorize 단위 테스트 — 합성 데이터로 그룹화·naming(가짜 LLM)·LocalStore 왕복·handler 를 돈다.
모델 없음, torch 없음(그걸 테스트로 고정한다). DB 는 붙지 않는다."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from categorize import cluster, concept, handler, naming, pipeline
from categorize.config import MODEL_VERSION, MODULE_ROOT, PARENTS, Knobs, Settings
from categorize.gallery import PhotoRef
from categorize.naming import majority
from categorize.pipeline import assign_ranks, concat_space, percentile
from categorize.store import ConceptAssignment, LocalStore, PhotoAnalysis


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def _world(tmp_path, n_groups=3, per_group=10, seed=0, clip_parent="실내 스튜디오"):
    """그룹마다 중심 벡터 + 노이즈. E(임베더)·C(CLIP)는 같은 그룹 구조를 공유한다.
    clip_parent 는 score 가 사진마다 저장하는 부모 검증 라벨(sub_scores.clip_parent)이다."""
    rng = np.random.default_rng(seed)
    dim = 32
    centers = [_unit(rng.normal(size=dim)) for _ in range(n_groups)]
    rows, E, C = [], [], []
    for g, c in enumerate(centers):
        for j in range(per_group):
            E.append(_unit(c + 0.08 * rng.normal(size=dim)))
            C.append(_unit(c + 0.08 * rng.normal(size=dim)))
            pid = f"g{g}-{j:02d}"
            rows.append(PhotoAnalysis(
                photo_id=pid, subjects="couple",
                sub_scores={"technical_score": rng.uniform(0.3, 0.8),
                            "aesthetic_score": rng.uniform(5, 6.5),
                            "sharpness": rng.uniform(50, 500),
                            "clip_parent": clip_parent},
                model_version=MODEL_VERSION))
    E, C = np.stack(E), np.stack(C)
    for r, t in zip(rows, percentile([r.sub_scores["technical_score"] for r in rows])):
        r.technical_pct = t
    X = concat_space(E, C)
    gids, _ = concept.concept_groups(X, 0.4, min_groups=2, max_share=0.6)
    for i, (r, g) in enumerate(zip(rows, gids)):
        r.embed_group_id = int(g)
        r.cluster_id = i          # 연사 없음 — 전부 단독 클러스터
    assign_ranks(rows)

    # 대표 사진용 실제 JPEG (naming 이 jpeg_bytes 를 부른다)
    img_root = tmp_path / "dataset"
    img_root.mkdir(exist_ok=True)
    for r in rows:
        Image.new("RGB", (32, 32), (200, 180, 160)).save(img_root / f"{r.photo_id}.jpg")
        r.photo_id = r.photo_id + ".jpg"    # LocalStore.preview_path 는 dataset_root/photo_id
    ids = [r.photo_id for r in rows]

    store = LocalStore(tmp_path / "out", dataset_root=img_root)
    store.write_analysis("g", rows, (ids, E), (ids, C))
    settings = Settings(out_root=tmp_path / "out", dataset_root=img_root)
    return store, rows, E, C, settings


# ── 순수 로직 ────────────────────────────────────────────────────────────────
def test_concat_space_is_mean_of_cosines():
    rng = np.random.default_rng(1)
    E = np.stack([_unit(rng.normal(size=8)) for _ in range(4)])
    C = np.stack([_unit(rng.normal(size=8)) for _ in range(4)])
    X = concat_space(E, C)
    np.testing.assert_allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-6)
    want = (E @ E.T + C @ C.T) / 2
    np.testing.assert_allclose(X @ X.T, want, atol=1e-5)


def test_assign_ranks_puts_reason_in_sub_scores():
    rows = [PhotoAnalysis(photo_id="a", technical_pct=90, sub_scores={"sharpness": 100.0}, cluster_id=0),
            PhotoAnalysis(photo_id="b", technical_pct=10, sub_scores={"sharpness": 10.0}, cluster_id=0)]
    assign_ranks(rows)
    best = next(r for r in rows if r.cluster_rank == 0)
    assert best.photo_id == "a"
    assert best.sub_scores["rank_reason"] == "technical"


# ── 연사 클러스터 — 카메라 파티션 (review-v3-design.md (1)) ──────────────────
def test_partition_order_splits_cameras_and_sorts_by_taken_at():
    cams = ["A", "B", "A", "B", None]
    times = [3, 1, 1, 2, None]
    parts = cluster.partition_order(cams, times)
    assert parts == [[2, 0], [1, 3], [4]]     # 파티션은 첫 등장 순, 안은 taken_at 순


def test_partition_order_keeps_input_order_when_taken_at_missing():
    assert cluster.partition_order(["A", "A", "A"], [5, None, 1]) == [[0, 1, 2]]


def test_partitioned_bursts_never_cross_cameras():
    rng = np.random.default_rng(2)
    base = _unit(rng.normal(size=16))
    emb = np.stack([_unit(base + 0.01 * rng.normal(size=16)) for _ in range(6)])
    cams = ["A", "B", "A", "B", "A", "B"]
    assert len(set(cluster.cluster_bursts(emb, 0.9, 8))) == 1   # 순서만으론 전부 한 덩어리
    parts = cluster.partition_order(cams, list(range(6)))
    cids = cluster.cluster_bursts_partitioned(emb, parts, 0.9, 8)
    assert (cids >= 0).all()
    a_ids = {cids[i] for i in range(6) if cams[i] == "A"}
    b_ids = {cids[i] for i in range(6) if cams[i] == "B"}
    assert a_ids.isdisjoint(b_ids)
    assert len(a_ids) == 1 and len(b_ids) == 1


# ── 임베딩 그룹 — 대칭 적응 규칙 (review-v3-design.md (10)) ──────────────────
def _two_cluster_emb(n_per=10, noise=0.05, seed=3):
    rng = np.random.default_rng(seed)
    centers = [_unit(rng.normal(size=16)) for _ in range(2)]
    return np.stack([_unit(c + noise * rng.normal(size=16))
                     for c in centers for _ in range(n_per)])


def test_concept_groups_raises_threshold_on_fragmentation():
    emb = _two_cluster_emb()
    labels, used = concept.concept_groups(emb, 0.01, min_groups=2)
    assert used > 0.01
    assert len(set(labels.tolist())) == 2


def test_concept_groups_raise_steps_back_before_overmerge():
    emb = _two_cluster_emb()
    labels, used = concept.concept_groups(emb, 0.01, min_groups=4)
    g = len(set(labels.tolist()))
    assert g >= 4
    assert np.bincount(labels).max() <= len(emb) * 0.5


# ── LocalStore 왕복 · 컬럼 경계 ──────────────────────────────────────────────
def test_local_store_roundtrip(tmp_path):
    store, rows, E, C, _ = _world(tmp_path)
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
    store, rows, *_ = _world(tmp_path)
    grouped = [PhotoAnalysis(photo_id=r.photo_id, subjects="unknown", technical_pct=1.0,
                             cluster_id=99, embed_group_id=7, model_version="") for r in rows]
    store.write_groups("g", grouped)
    after = {r.photo_id: r for r in store.read_analysis("g")}
    # subjects · model_version 은 score 의 것 — 건드리지 않는다
    assert all(after[r.photo_id].subjects == "couple" and after[r.photo_id].model_version == MODEL_VERSION for r in rows)
    assert all(after[r.photo_id].embed_group_id == 7 and after[r.photo_id].cluster_id == 99 for r in rows)


# ── naming (가짜 LLM) ────────────────────────────────────────────────────────
class FakeLlm:
    """청크 vision 호출은 그룹마다 이름을, 통합 호출은 concept 표기를 바꿔 돌려준다."""

    def __init__(self, parent="실내 스튜디오", low_conf_gid=None):
        self.calls: list[tuple[str, bool]] = []
        self.parent = parent
        self.low_conf_gid = low_conf_gid

    def complete_json(self, system, user, schema, max_tokens):
        has_image = not isinstance(user, str) and any(k == "image" for k, _ in user)
        kind = "vision" if has_image else "merge"
        self.calls.append((kind, has_image))
        if kind == "vision":
            gids = [int(t.split("]")[0].split()[1]) for k, t in user
                    if k == "text" and t.startswith("[그룹")]
            return {"groups": [
                {"group_id": g, "parent": self.parent, "proposed_parent": None,
                 "concept": f"세트{g}",
                 "confidence": 0.3 if g == self.low_conf_gid else 0.95}
                for g in gids]}
        out = []
        for ln in user.splitlines():
            gid = int(ln.split(":")[0].split()[1])
            out.append({"group_id": gid, "parent": self.parent, "proposed_parent": None,
                        "concept": f"세트{gid}(통일)"})
        return {"groups": out}


def _with_knobs(settings, **kw):
    return Settings(out_root=settings.out_root, dataset_root=settings.dataset_root, knobs=Knobs(**kw), llm=settings.llm)


def test_naming_all_groups_named_by_vlm(tmp_path):
    store, rows, *_, settings = _world(tmp_path)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    back = store.read_assignments("g")
    n_groups = len({r.embed_group_id for r in rows})
    assert len(back) == n_groups == result["groups"]
    assert all(a.assigned_by == "vlm" for a in back)
    assert all(not a.needs_review for a in back)
    assert result["vlmGroups"] == n_groups and result["nearestGroups"] == 0


def test_naming_nearest_inherits_and_flags_far_groups(tmp_path):
    store, rows, *_, settings = _world(tmp_path, n_groups=4, per_group=8, clip_parent=None)
    settings = _with_knobs(settings, naming_max_groups=2, nearest_tau=0.05)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    back = {a.embed_group_id: a for a in store.read_assignments("g")}
    assert result["vlmGroups"] == 2
    nearest = [a for a in back.values() if a.assigned_by == "nearest"]
    assert nearest
    for a in nearest:
        if a.parent_name == "기타":
            assert a.needs_review
        else:
            assert a.concept_name.startswith("세트")


def test_naming_low_confidence_and_clip_mismatch_need_review(tmp_path):
    store, rows, *_, settings = _world(tmp_path, clip_parent="야외 자연")
    gids = sorted({r.embed_group_id for r in rows})
    naming.run(store, "g", settings, FakeLlm(low_conf_gid=gids[0]), job_id=None)
    back = store.read_assignments("g")
    assert all(a.needs_review for a in back)
    assert all(a.clip_parent == "야외 자연" for a in back)


def test_naming_merge_call_unifies_names_across_chunks(tmp_path):
    store, rows, *_, settings = _world(tmp_path, n_groups=5, per_group=6)
    settings = _with_knobs(settings, naming_chunk=2, naming_spread_extra=9.0)
    llm = FakeLlm()
    result = naming.run(store, "g", settings, llm, job_id=None)
    kinds = [kind for kind, _ in llm.calls]
    assert kinds.count("vision") == 3 and kinds.count("merge") == 1
    assert result["llmCalls"] == 4
    assert all(a.concept_name.endswith("(통일)") for a in store.read_assignments("g"))


def test_naming_coverage_target_limits_vlm_groups(tmp_path):
    store, rows, *_, settings = _world(tmp_path, n_groups=4, per_group=8, clip_parent=None)
    settings = _with_knobs(settings, naming_coverage=0.5, nearest_tau=1.0)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["vlmGroups"] < result["groups"]
    assert result["nearestGroups"] == result["groups"] - result["vlmGroups"]
    assert 0.5 <= result["coverage"] < 1.0


def test_naming_spread_adds_second_rep(tmp_path):
    store, rows, *_, settings = _world(tmp_path)
    settings = _with_knobs(settings, naming_spread_extra=0.0)
    result = naming.run(store, "g", settings, FakeLlm(), job_id=None)
    assert result["extraReps"] == result["groups"]
    assert result["vlmGroups"] == result["groups"]


def test_naming_requires_llm(tmp_path):
    store, *_, settings = _world(tmp_path)
    with pytest.raises(RuntimeError, match="Bedrock"):
        naming.run(store, "g", settings, None)


def test_naming_requires_full_analysis(tmp_path):
    store = LocalStore(tmp_path / "out", dataset_root=tmp_path)
    settings = Settings(out_root=tmp_path / "out", dataset_root=tmp_path)
    with pytest.raises(RuntimeError, match="FULL"):
        naming.run(store, "empty", settings, FakeLlm())


def test_majority_ignores_none_and_breaks_ties_by_first_seen():
    assert majority(["야외 자연", None, "야외 자연", "실내 스튜디오"]) == "야외 자연"
    assert majority([None, None]) is None
    assert majority(["a", "b"]) == "a"


# ── pipeline (그룹 → naming) ─────────────────────────────────────────────────
def test_pipeline_groups_then_names(tmp_path):
    store, rows, E, C, settings = _world(tmp_path)
    refs = [PhotoRef(photo_id=r.photo_id, path=None) for r in rows]   # 파일이 필요 없다
    settings = _with_knobs(settings, group_distance=0.4, group_min_groups=2, group_max_share=0.6)

    result = pipeline.run(store, "g", refs, settings, FakeLlm(), job_id=None)

    assert result["mode"] == "categorize" and result["photos"] == len(rows)
    assert result["embeddingsSource"] == "dinov3"
    assert result["clusters"] == len(rows)
    assert result["naming"]["vlmGroups"] == result["groups"]["groups"]
    back = store.read_analysis("g")
    assert all(r.embed_group_id >= 0 and "sharpness_pct" in r.sub_scores for r in back)
    assert all(a.assigned_by == "vlm" for a in store.read_assignments("g"))


def test_pipeline_falls_back_to_clip_when_no_embedder_vectors(tmp_path):
    store, rows, E, C, settings = _world(tmp_path)
    ids = [r.photo_id for r in rows]
    store.write_analysis("g", rows, ([], np.zeros((0, 0))), (ids, C))
    refs = [PhotoRef(photo_id=r.photo_id, path=None) for r in rows]

    result = pipeline.run(store, "g", refs, settings, None)

    assert result["embeddingsSource"] == "clip"
    assert result["naming"] == "skipped (no --llm)"


# ── 경계 — torch 없음 · score 와의 계약 ──────────────────────────────────────
def test_module_never_imports_torch():
    code = ("import sys; import categorize.pipeline, categorize.naming, categorize.job, categorize.handler; "
            "assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=MODULE_ROOT)


def _literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", None) or ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path}: {name} 없음")


def test_model_version_and_parents_match_score_module():
    other = MODULE_ROOT.parent / "score" / "score" / "config.py"
    assert _literal(other, "MODEL_VERSION") == MODEL_VERSION
    assert _literal(other, "PARENTS") == PARENTS
    assert "기타" in PARENTS and len(PARENTS) == len(set(PARENTS))


# ── handler ──────────────────────────────────────────────────────────────────
def test_handler_passes_gallery_job_and_llm(monkeypatch):
    calls = []
    monkeypatch.setattr(handler.job, "run", lambda **kw: calls.append(kw) or {"ok": True})
    monkeypatch.setattr(handler.llm, "bedrock_client", lambda s: "LLM")

    result = handler.handler({"galleryId": "7", "jobId": "3"}, None)

    assert result == {"ok": True}
    assert calls[0]["gallery_id"] == 7 and calls[0]["job_id"] == 3 and calls[0]["llm"] == "LLM"

    handler.handler({"galleryId": 8}, None)
    assert calls[1]["job_id"] is None


# ── 대상 선별 (wes V15) ───────────────────────────────────────────────────────
def test_load_db_targets_previews_and_ignores_photo_status():
    """#93: wes V15(2026-09-08)가 photos.status 에서 EMBEDDED 를 없앴다 — 그 값을 조건에 걸면 대상이 0장이 된다.
    임베더가 지난 사진은 preview_key 로 알아본다(score 의 load_db 와 같은 규칙)."""
    from categorize.gallery import load_db

    class _Cur:
        def __init__(self, conn):
            self.conn = conn

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return None

        def execute(self, sql, params=None):
            self.conn.executed.append((" ".join(sql.split()), params))

        def fetchall(self):
            return self.conn.rows

    class _Conn:
        def __init__(self, rows):
            self.rows = rows
            self.executed = []

        def cursor(self):
            return _Cur(self)

    conn = _Conn(rows=[(11, "previews/a.jpg", None, "Canon", "R5"), (12, "previews/b.jpg", None, None, None)])
    refs = load_db(conn, None, 7, Path("/tmp"), download=False)

    sql, params = conn.executed[0]
    assert "status" not in sql
    assert "JOIN" not in sql and sql.split("FROM")[1].split()[0] == "photos"
    assert "preview_key IS NOT NULL" in sql and "deleted_at IS NULL" in sql
    assert params == (7,)
    assert [(r.photo_id, r.camera, r.path) for r in refs] == [("11", "Canon R5", None), ("12", None, None)]


# ── 잡 계약 (wes V16) ─────────────────────────────────────────────────────────
class _JobConn:
    def __init__(self):
        self.executed = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        conn = self

        class _C:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return None

            def execute(self_inner, sql, params=None):
                conn.executed.append((" ".join(sql.split()), params))

        return _C()

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


def test_jobs_only_writes_error_column():
    """#95: V16 은 status·started_at·finished_at·result 를 지웠고 photoselect 에 UPDATE (error, updated_at) 만 준다."""
    from categorize import jobs

    assert not hasattr(jobs, "start") and not hasattr(jobs, "finish")   # 옛 계약은 사라졌다
    conn = _JobConn()
    jobs.fail(conn, 3, "RuntimeError: boom")

    sql, params = conn.executed[0]
    assert sql == "UPDATE ai_analysis_jobs SET error = %s, updated_at = now() WHERE id = %s"
    assert params == ("RuntimeError: boom", 3)
    assert conn.rollbacks == 1 and conn.commits == 1
    for banned in ("status", "started_at", "finished_at", "result", "version"):
        assert banned not in sql


def test_job_run_does_not_open_or_close_the_job(monkeypatch, tmp_path):
    """성공 경로는 잡 테이블을 건드리지 않는다 — 여는 것도 닫는 것도 wes."""
    from categorize import job as job_mod

    conn = _JobConn()
    monkeypatch.setattr(job_mod.db, "connect", lambda s: conn)
    monkeypatch.setattr(job_mod, "load_db", lambda *a, **k: [PhotoRef(photo_id="1", path=None)])
    monkeypatch.setattr(job_mod, "DbStore", lambda settings, connection: "STORE")
    monkeypatch.setattr(job_mod.pipeline, "run", lambda *a, **k: {"gallery": "7"})

    result = job_mod.run(7, settings=Settings(out_root=tmp_path, dataset_root=tmp_path, work_dir=tmp_path), job_id=3, llm="LLM")

    assert result["gallery"] == "7" and "elapsedSeconds" in result
    assert conn.executed == []                                          # 잡 테이블 접근 0


def test_job_run_writes_error_when_pipeline_fails(monkeypatch, tmp_path):
    from categorize import job as job_mod

    conn = _JobConn()
    monkeypatch.setattr(job_mod.db, "connect", lambda s: conn)
    monkeypatch.setattr(job_mod, "load_db", lambda *a, **k: [PhotoRef(photo_id="1", path=None)])
    monkeypatch.setattr(job_mod, "DbStore", lambda settings, connection: "STORE")

    def boom(*a, **k):
        raise RuntimeError("no vectors")

    monkeypatch.setattr(job_mod.pipeline, "run", boom)

    with pytest.raises(RuntimeError):
        job_mod.run(7, settings=Settings(out_root=tmp_path, dataset_root=tmp_path, work_dir=tmp_path), job_id=3, llm="LLM")

    sql, params = conn.executed[0]
    assert "SET error = %s" in sql and params == ("RuntimeError: no vectors", 3)


# ── 갤러리 한 번 읽기 · 대표 사진 배치 다운로드 ──────────────────────────────
class _CountingStore:
    """LocalStore 를 감싸 read_gallery · preview_paths 호출을 센다."""

    def __init__(self, inner):
        self.inner = inner
        self.reads = 0
        self.preview_calls: list[list[str]] = []

    def read_gallery(self, gallery):
        self.reads += 1
        return self.inner.read_gallery(gallery)

    def preview_paths(self, gallery, photo_ids):
        self.preview_calls.append(list(photo_ids))
        return self.inner.preview_paths(gallery, photo_ids)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def test_pipeline_reads_gallery_once_and_hands_grouped_to_naming(tmp_path):
    """그룹화가 읽은 행·concat 공간을 naming 이 그대로 받는다 — DB 를 두 번 읽지 않는다."""
    store, rows, *_, settings = _world(tmp_path)
    counting = _CountingStore(store)
    refs = [PhotoRef(photo_id=r.photo_id, path=None) for r in rows]
    settings = _with_knobs(settings, group_distance=0.4, group_min_groups=2, group_max_share=0.6)

    result = pipeline.run(counting, "g", refs, settings, FakeLlm(), job_id=None)

    assert counting.reads == 1
    assert result["naming"]["vlmGroups"] == result["groups"]["groups"]


def test_naming_fetches_all_representatives_in_one_batch(tmp_path):
    """대표 사진 경로는 preview_paths 한 번으로 — 사진마다 SELECT + GET 을 직렬로 하지 않는다."""
    store, rows, *_, settings = _world(tmp_path)
    counting = _CountingStore(store)

    result = naming.run(counting, "g", settings, FakeLlm(), job_id=None)

    assert len(counting.preview_calls) == 1
    assert len(counting.preview_calls[0]) >= result["vlmGroups"]
    assert len(set(counting.preview_calls[0])) == len(counting.preview_calls[0])


class _VecConn:
    def __init__(self, rows):
        self.rows = rows
        self.executed = []

    def cursor(self):
        conn = self

        class _C:
            def __enter__(self_inner):
                return self_inner

            def __exit__(self_inner, *a):
                return None

            def execute(self_inner, sql, params=None):
                conn.executed.append((" ".join(sql.split()), params))

            def fetchall(self_inner):
                return conn.rows

        return _C()


def _db_store(tmp_path, conn, bucket=None):
    from categorize.store import DbStore

    settings = Settings(out_root=tmp_path, dataset_root=tmp_path, work_dir=tmp_path / "work", s3_bucket=bucket)
    return DbStore(settings, connection=conn)


def test_db_store_read_gallery_is_one_query(tmp_path):
    """분석 행 + DINOv3 + CLIP 을 한 SELECT 로. model_version 없는 행은 벡터만 남고 행 목록에서 빠진다."""
    e, c = np.ones(4, dtype=np.float32), np.zeros(4, dtype=np.float32)
    conn = _VecConn(rows=[
        (11, "couple", 50.0, 50.0, {"technical_score": 0.5}, -1, 0, -1, MODEL_VERSION, e, "dinov3", c),
        (12, "unknown", 50.0, 50.0, None, -1, 0, -1, None, e, "dinov3", None),          # 임베딩만, 점수 아직
    ])
    data = _db_store(tmp_path, conn).read_gallery("7")

    assert len(conn.executed) == 1
    sql, params = conn.executed[0]
    assert params == (7,)
    assert "a.embedding" in sql and "a.clip_embedding" in sql and "JOIN photos" in sql
    assert [r.photo_id for r in data.rows] == ["11"]
    assert data.rows[0].sub_scores == {"technical_score": 0.5}
    assert set(data.embeddings) == {"11", "12"} and set(data.clip_embeddings) == {"11"}


def test_db_store_read_gallery_rejects_mixed_embedding_models(tmp_path):
    e = np.ones(4, dtype=np.float32)
    conn = _VecConn(rows=[
        (11, "couple", 50.0, 50.0, {}, -1, 0, -1, MODEL_VERSION, e, "dinov3", e),
        (12, "couple", 50.0, 50.0, {}, -1, 0, -1, MODEL_VERSION, e, "dinov2", e),
    ])
    with pytest.raises(RuntimeError, match="embedding_model"):
        _db_store(tmp_path, conn).read_gallery("7")


def test_db_store_preview_paths_batches_select_and_downloads_concurrently(tmp_path, monkeypatch):
    """preview_key 는 SELECT 한 번(ANY), 다운로드는 스레드풀 — 사진 수만큼 왕복하지 않는다."""
    import threading
    import time as _time

    conn = _VecConn(rows=[(11, "previews/a.jpg"), (12, "previews/b.jpg"), (13, "previews/c.jpg")])
    store = _db_store(tmp_path, conn, bucket="bkt")
    seen_threads: set[int] = set()
    lock = threading.Lock()

    class _FakeStorage:
        def download(self, key, dest):
            with lock:
                seen_threads.add(threading.get_ident())
            _time.sleep(0.05)                      # 겹쳐 돌면 세 건이 0.15초가 아니라 ~0.05초
            if key.endswith("c.jpg"):
                raise OSError("no such key")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"jpeg")
            return dest

    store._storage = _FakeStorage()
    started = _time.monotonic()
    paths = store.preview_paths("7", ["11", "12", "13"])
    elapsed = _time.monotonic() - started

    assert len(conn.executed) == 1
    sql, params = conn.executed[0]
    assert "ANY(%s)" in sql and params == ([11, 12, 13],)
    assert set(paths) == {"11", "12"}                     # 실패한 13 은 빠진다
    assert len(seen_threads) > 1 and elapsed < 0.14

    # 이미 받아 둔 파일은 SELECT 도 다운로드도 없이 돌려준다
    again = store.preview_paths("7", ["11", "12"])
    assert again == paths and len(conn.executed) == 1
