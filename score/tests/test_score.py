"""score 단위 테스트 — 합성 데이터로 재개·배치 쓰기·데드라인 정지·LocalStore 컬럼 경계·handler 규칙을 돈다.
모델 없음(torch 러너는 가짜). DB 는 붙지 않는다."""

from __future__ import annotations

import ast
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from score import handler, job, pipeline
from score.config import MODEL_VERSION, MODULE_ROOT, PARENTS, Knobs, Settings
from score.gallery import PhotoRef
from score.store import LocalStore, PhotoAnalysis


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class _FakeLaion:
    """CLIP 흉내 — 이미지·텍스트 모두 32차원 단위 벡터. torch 없음."""

    def __init__(self, dim=32, seed=7):
        self.rng = np.random.default_rng(seed)
        self.dim = dim
        self.embed_calls = 0
        self.batch_sizes = []

    def embed(self, source):
        self.embed_calls += 1
        return _unit(self.rng.normal(size=self.dim))

    def embed_batch(self, sources):
        self.batch_sizes.append(len(sources))
        return np.stack([self.embed(s) for s in sources])

    def prepare(self, source):
        return source

    def embed_prepared(self, tensors):
        return self.embed_batch(tensors)

    def embed_texts(self, prompts):
        return np.stack([_unit(self.rng.normal(size=self.dim)) for _ in prompts])

    def score_from_embedding(self, emb):
        return 5.5


class _FakeArniqa:
    def __init__(self, long_edge=None):
        self.long_edge = long_edge
        self.seen = []
        self.batch_sizes = []

    def score(self, source):
        self.seen.append(source)
        return {"technical_score": 0.6}

    def score_batch(self, sources):
        self.batch_sizes.append(len(sources))
        return [self.score(s) for s in sources]

    def prepare(self, source):
        return source

    def score_prepared(self, tensors):
        return self.score_batch(tensors)


@pytest.fixture
def fake_runners(monkeypatch):
    """pipeline.run 이 함수 안에서 import 하는 torch 러너·classical 을 가짜로 바꾼다."""
    laion = _FakeLaion()
    runners = types.ModuleType("score.runners")
    runners.LaionRunner = lambda **kw: laion
    laion.arniqa = _FakeArniqa()

    def make_arniqa(**kw):
        laion.arniqa.long_edge = kw.get("long_edge")
        return laion.arniqa

    runners.ArniqaRunner = make_arniqa
    classical = types.ModuleType("score.classical")
    classical.measure = lambda source: {"sharpness": 100.0, "highlight_clip": 0.0, "shadow_clip": 0.0, "mean_luma": 120.0}
    monkeypatch.setitem(sys.modules, "score.runners", runners)
    monkeypatch.setitem(sys.modules, "score.classical", classical)
    return laion


def _world(tmp_path, n=12):
    """사진 n 장 JPEG + 그중 절반은 이미 점수·CLIP 이 있는 LocalStore."""
    img_root = tmp_path / "dataset"
    img_root.mkdir()
    ids = [f"p{i:02d}.jpg" for i in range(n)]
    for pid in ids:
        Image.new("RGB", (32, 32), (200, 180, 160)).save(img_root / pid)
    store = LocalStore(tmp_path / "out", dataset_root=img_root)
    scored = ids[: n // 2]
    rows = [PhotoAnalysis(photo_id=pid, subjects="couple", technical_pct=77.0, cluster_id=3, embed_group_id=5,
                          sub_scores={"technical_score": 0.5, "clip_parent": "실내 스튜디오"},
                          model_version=MODEL_VERSION) for pid in scored]
    store.write_scores("g", rows, (scored, np.stack([_unit(np.ones(32) + i) for i in range(len(scored))])))
    refs = [PhotoRef(photo_id=pid, path=str(img_root / pid)) for pid in ids]
    settings = Settings(out_root=tmp_path / "out", dataset_root=img_root, knobs=Knobs(write_batch=4))
    return store, refs, scored, settings


# ── pipeline ──────────────────────────────────────────────────────────────────
def test_skips_scored_photos_and_stores_clip_parent(tmp_path, fake_runners):
    store, refs, scored, settings = _world(tmp_path)

    result = pipeline.run(store, "g", refs, settings)

    assert result["skipped"] == len(scored) and result["processed"] == len(refs) - len(scored)
    assert fake_runners.embed_calls == len(refs) - len(scored)
    assert result["stopped"] is False and result["remaining"] == 0
    back = {r.photo_id: r for r in store.read_analysis("g")}
    new = back[refs[-1].photo_id]
    assert new.sub_scores["clip_parent"] in PARENTS and new.model_version == MODEL_VERSION
    assert "technical_score" in new.sub_scores and "sharpness" in new.sub_scores
    ids, C = store.read_clip_embeddings("g")
    assert set(ids) == {r.photo_id for r in refs} and C.shape[0] == len(refs)

    # force 면 전부 다시.
    assert pipeline.run(store, "g", refs, settings, force=True)["processed"] == len(refs)


def test_write_scores_keeps_categorize_columns(tmp_path, fake_runners):
    store, refs, scored, settings = _world(tmp_path)
    before = {r.photo_id: (r.technical_pct, r.cluster_id, r.embed_group_id) for r in store.read_analysis("g")}
    assert all(v == (77.0, 3, 5) for v in before.values())

    pipeline.run(store, "g", refs, settings, force=True)

    after = {r.photo_id: r for r in store.read_analysis("g")}
    for pid, (pct, cid, gid) in before.items():
        # score 의 쓰기는 백분위·연사·그룹(categorize 의 컬럼)을 건드리지 않는다
        assert (after[pid].technical_pct, after[pid].cluster_id, after[pid].embed_group_id) == (pct, cid, gid)
        assert after[pid].subjects in ("bride", "groom", "couple", "group", "unknown")


def test_writes_in_batches_and_stops_at_deadline(tmp_path, fake_runners):
    store, refs, scored, settings = _world(tmp_path, n=12)   # 12 장 전부 대상(force), 배치 4
    clock = {"left": 1000.0}

    def remaining():
        clock["left"] -= 500   # 배치마다 500s 씩 사라진다 → 두 번째 경계에서 예산(60s+배치) 미달
        return clock["left"]

    result = pipeline.run(store, "g", refs, settings, force=True, remaining_seconds=remaining)

    assert result["stopped"] is True
    assert result["processed"] % settings.knobs.write_batch == 0 and 0 < result["processed"] < len(refs)
    assert result["remaining"] == len(refs) - result["processed"]
    # 멈추기 전 배치는 이미 저장돼 있다 — 다시 부르면 그만큼 건너뛴다
    again = pipeline.run(store, "g", refs, settings)
    assert again["skipped"] == result["processed"]


def test_no_deadline_without_callback(tmp_path, fake_runners):
    store, refs, _, settings = _world(tmp_path)
    result = pipeline.run(store, "g", refs, settings, force=True)
    assert result["stopped"] is False and result["processed"] == len(refs)


def test_clip_runs_in_batches_and_runners_share_one_decoded_image(tmp_path, fake_runners):
    store, refs, _, settings = _world(tmp_path, n=12)
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        knobs=Knobs(write_batch=4, clip_batch=5, arniqa_long_edge=640))

    result = pipeline.run(store, "g", refs, settings, force=True)

    assert result["processed"] == 12 and fake_runners.batch_sizes == [5, 5, 2]
    # ARNIQA 는 경로가 아니라 이미 디코드된 PIL 이미지를 받고, 손잡이의 해상도를 넘겨받는다
    assert all(isinstance(x, Image.Image) for x in fake_runners.arniqa.seen)
    assert fake_runners.arniqa.long_edge == 640


def test_broken_photo_fails_alone_inside_a_batch(tmp_path, fake_runners):
    store, refs, scored, settings = _world(tmp_path, n=8)
    broken = refs[-1]                                # 아직 점수가 없는 쪽에서 하나를 깨뜨린다
    Path(broken.path).write_bytes(b"not a jpeg")

    result = pipeline.run(store, "g", refs, settings, force=True)

    assert result["failed"] == [broken.photo_id] and result["processed"] == 7
    assert fake_runners.batch_sizes == [7]          # 깨진 장은 디코드에서 빠지고 나머지 7장이 한 묶음
    ids, _ = store.read_clip_embeddings("g")
    assert broken.photo_id not in ids and len(ids) == 7


def test_batch_failure_falls_back_to_single_embeds(tmp_path, fake_runners):
    store, refs, _, settings = _world(tmp_path, n=6)

    def boom(sources):
        fake_runners.batch_sizes.append(len(sources))
        raise RuntimeError("batch oom")

    fake_runners.embed_batch = boom
    result = pipeline.run(store, "g", refs, settings, force=True)

    assert result["processed"] == 6 and result["failed"] == []
    assert fake_runners.embed_calls == 6


def test_arniqa_runs_in_batches_and_falls_back_alone(tmp_path, fake_runners):
    store, refs, _, settings = _world(tmp_path, n=10)
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        knobs=Knobs(write_batch=4, clip_batch=8, arniqa_batch=3))

    result = pipeline.run(store, "g", refs, settings, force=True)

    assert result["processed"] == 10 and fake_runners.arniqa.batch_sizes == [3, 3, 2, 2]

    fake_runners.arniqa.batch_sizes.clear()
    fake_runners.arniqa.seen.clear()

    def boom(sources):
        fake_runners.arniqa.batch_sizes.append(len(sources))
        raise RuntimeError("batch oom")

    fake_runners.arniqa.score_prepared = boom
    result = pipeline.run(store, "g", refs, settings, force=True)
    assert result["processed"] == 10 and result["failed"] == [] and len(fake_runners.arniqa.seen) == 10


def test_decode_prefetch_threads_give_same_result_and_keep_order(tmp_path, fake_runners):
    store, refs, _, settings = _world(tmp_path, n=13)
    settings = Settings(out_root=settings.out_root, dataset_root=settings.dataset_root,
                        knobs=Knobs(write_batch=4, clip_batch=5, decode_workers=3))
    broken = refs[3]
    Path(broken.path).write_bytes(b"not a jpeg")

    result = pipeline.run(store, "g", refs, settings, force=True)

    assert result["processed"] == 12 and result["failed"] == [broken.photo_id]
    assert fake_runners.batch_sizes == [4, 5, 3]
    ids, _ = store.read_clip_embeddings("g")   # 깨진 장은 _world 가 미리 넣어 둔 옛 벡터가 남는다 — 그것만 빼고 순서 비교
    assert [i for i in ids if i != broken.photo_id] == [r.photo_id for r in refs if r is not broken]


def test_arniqa_batches_group_by_shape_and_keep_order(tmp_path):
    """실제 ArniqaRunner.score_batch — 모델은 가짜(입력 합을 점수로), 크기가 다른 장이 섞여도 순서가 지켜진다."""
    import torch

    from score.runners.arniqa import ArniqaRunner

    calls = []

    class _Model:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, x, x_ds, return_embedding=False, scale_score=True):
            calls.append(tuple(x.shape))
            return x.float().mean(dim=(1, 2, 3)) + x_ds.float().mean(dim=(1, 2, 3))

    runner = ArniqaRunner.__new__(ArniqaRunner)
    runner.long_edge, runner.device, runner.fp16, runner._model = 64, "cpu", False, _Model()
    runner._mean, runner._std = torch.zeros(1, 3, 1, 1), torch.ones(1, 3, 1, 1)
    portrait = Image.new("RGB", (32, 64), (10, 20, 30))
    landscape = Image.new("RGB", (64, 32), (200, 210, 220))
    out = runner.score_batch([portrait, landscape, portrait, landscape])
    single = [runner.score(im)["technical_score"] for im in (portrait, landscape)]

    assert [o["technical_score"] for o in out] == pytest.approx([single[0], single[1], single[0], single[1]])
    assert calls[:2] == [(2, 3, 64, 32), (2, 3, 32, 64)]
    assert torch.tensor(single[0]) != torch.tensor(single[1])


def test_images_fit_long_edge_never_upscales_and_accepts_paths_or_images(tmp_path):
    from score import images

    big = Image.new("RGB", (3200, 1600))
    small = Image.new("RGB", (300, 200))
    assert images.fit_long_edge(big, 1600).size == (1600, 800)
    assert images.fit_long_edge(small, 1600) is small
    path = tmp_path / "a.jpg"
    Image.new("RGB", (2000, 1000)).save(path)
    assert images.as_image(str(path), 1000).size == (1000, 500)
    assert images.as_image(images.load_image(str(path)), 800).size == (800, 400)


def test_classical_measure_same_for_path_and_decoded_image(tmp_path):
    from score import classical, images

    rng = np.random.default_rng(0)
    path = tmp_path / "n.jpg"
    Image.fromarray(rng.integers(0, 255, (400, 600, 3), dtype=np.uint8)).save(path, quality=95)
    assert classical.measure(str(path)) == classical.measure(images.load_image(str(path)))


# ── categorize 와의 계약 ───────────────────────────────────────────────────────
def _literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", None) or ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path}: {name} 없음")


def test_model_version_and_parents_match_categorize_module():
    other = MODULE_ROOT.parent / "categorize" / "categorize" / "config.py"
    assert _literal(other, "MODEL_VERSION") == MODEL_VERSION
    assert _literal(other, "PARENTS") == PARENTS


# ── handler ───────────────────────────────────────────────────────────────────
class _Context:
    function_name = "wes-score"

    def __init__(self, remaining_ms: int = 600_000) -> None:
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self) -> int:
        return self.remaining_ms


@pytest.fixture
def fake_handler(monkeypatch):
    """handler 는 이제 photoIds 하나만 받는다(#98) — job.run 을 가짜로 두고 인자만 본다."""
    calls = {"run": []}

    def run(gallery_id, settings, remaining_seconds, photo_ids):
        calls["run"].append({"gallery_id": gallery_id, "photo_ids": photo_ids,
                             "remaining": remaining_seconds() if remaining_seconds else None})
        return {"gallery": str(gallery_id), "processed": len(photo_ids)}

    monkeypatch.setattr(handler.job, "run", run)
    return calls


def test_handler_rejects_legacy_gallery_payload(fake_handler):
    """옛 갤러리 페이로드(#54 조정자)는 조용히 전수 스캔하지 않고 에러다(#98)."""
    with pytest.raises(ValueError, match="photoIds"):
        handler.handler({"galleryId": 7}, _Context())
    with pytest.raises(ValueError, match="photoIds"):
        handler.handler({"galleryId": 7, "jobId": 3, "force": True}, _Context())
    with pytest.raises(ValueError, match="galleryId"):
        handler.handler({"photoIds": [1]}, _Context())
    assert fake_handler["run"] == []


# ── v2 (#75): Scorer 재사용 · claim_batch · GPU 워커 루프 · photoIds 폴백 ─────────────────────────────
def test_scorer_is_reused_across_runs(tmp_path, fake_runners, monkeypatch):
    store, refs, _, settings = _world(tmp_path, n=6)
    fake_mod = sys.modules["score.runners"]          # fixture 가 끼운 가짜 모듈
    made = []
    original = fake_mod.LaionRunner
    monkeypatch.setattr(fake_mod, "LaionRunner", lambda **kw: made.append(1) or original(**kw))

    scorer = pipeline.Scorer(settings)
    a = pipeline.run(store, "g", refs[:3], settings, force=True, scorer=scorer)
    b = pipeline.run(store, "g", refs[3:], settings, force=True, scorer=scorer)

    assert made == [1] and a["processed"] == 3 and b["processed"] == 3
    assert a["perStageSeconds"]["load"] == b["perStageSeconds"]["load"] == round(scorer.load_seconds, 1)
    # scorer 없이 부르면 예전처럼 하나 새로 만든다
    pipeline.run(store, "g", refs[:1], settings, force=True)
    assert made == [1, 1]


class _Cur:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return None

    def execute(self, sql, params=None):
        self.conn.executed.append((" ".join(sql.split()), params))

    def executemany(self, sql, params):
        self.conn.executed.append((" ".join(sql.split()), list(params)))

    def fetchall(self):
        return self.conn.rows


class _Conn:
    def __init__(self, rows=()):
        self.rows = list(rows)
        self.executed = []
        self.commits = 0
        self.rollbacks = 0

    def cursor(self):
        return _Cur(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1

    def close(self):
        pass


def test_claim_batch_locks_photo_analysis_rows_without_embedding_and_skips_locked():
    from score.store import DbStore

    conn = _Conn(rows=[(11, "previews/a.jpg", None, "Canon", "R5"), (12, "previews/b.jpg", None, None, None)])
    store = DbStore(SimpleNamespace(), conn)

    refs = store.claim_batch(32, exclude=[99])

    sql, params = conn.executed[0]
    assert "FROM photo_analysis a" in sql and "FOR UPDATE OF a SKIP LOCKED" in sql
    assert "a.embedding IS NOT NULL AND a.clip_embedding IS NULL" in sql
    assert "p.preview_key IS NOT NULL" in sql and "p.deleted_at IS NULL" in sql and "g.deleted_at IS NULL" in sql
    assert "status" not in sql                                    # v2: status 는 보지 않는다
    assert params == ([99], 32)
    assert [(r.photo_id, r.preview_key, r.camera) for r in refs] == [("11", "previews/a.jpg", "Canon R5"), ("12", "previews/b.jpg", None)]
    assert conn.commits == 0                                      # 잠금은 호출자가 commit/rollback 할 때까지


def test_load_db_and_load_by_ids_do_not_depend_on_embedded_status():
    from score.gallery import load_by_ids, load_db

    conn = _Conn(rows=[(5, "previews/e.jpg", None, None, None), (3, "previews/c.jpg", None, "Sony", "A7")])
    refs = load_by_ids(conn, [3, 5, 8])
    sql, params = conn.executed[0]
    assert "p.id = ANY(%s)" in sql and "status" not in sql and params == ([3, 5, 8],)
    assert [r.photo_id for r in refs] == ["3", "5"] and refs[0].camera == "Sony A7"

    conn = _Conn(rows=[])
    load_db(conn, None, 7, Path("/tmp"), download=False)
    sql, _ = conn.executed[0]
    assert "status" not in sql and "preview_key IS NOT NULL" in sql


@pytest.fixture
def fake_worker(monkeypatch, tmp_path):
    """gpu_worker 의 바깥(DB·S3·러너)을 전부 가짜로. claim 은 큐에서 꺼내고, run 은 처리 장수를 돌려준다."""
    from score import gpu_worker

    state = {"claims": [], "queue": [], "runs": [], "rollbacks": 0, "stopped": 0, "excludes": []}

    class Store:
        def __init__(self, settings, connection):
            pass

        def claim_batch(self, n, exclude=None):
            state["excludes"].append(list(exclude or []))
            batch = state["queue"].pop(0) if state["queue"] else []
            state["claims"].append(len(batch))
            return batch

        def rollback(self):
            state["rollbacks"] += 1

        def commit(self):
            state["commits"] = state.get("commits", 0) + 1

        def write_errors(self, ids, error):
            state.setdefault("errors", []).append((list(ids), error))

    def run(store, gallery, refs, settings, force=False, scorer=None, **kw):
        state["runs"].append([r.photo_id for r in refs])
        if any(r.photo_id == "boom" for r in refs):
            raise RuntimeError("batch boom")
        failed = [r.photo_id for r in refs if r.photo_id == "7"]
        return {"processed": len(refs) - len(failed), "failed": failed}

    monkeypatch.setattr(gpu_worker.db, "connect", lambda s: _Conn())
    monkeypatch.setattr(gpu_worker, "DbStore", Store)
    monkeypatch.setattr(gpu_worker, "PreviewStorage", lambda bucket: None)
    monkeypatch.setattr(gpu_worker, "download_previews", lambda storage, refs, d, workers=8, missing=None: refs)
    monkeypatch.setattr(gpu_worker.pipeline, "Scorer", lambda settings: SimpleNamespace(warm_up=lambda: 0.0))
    monkeypatch.setattr(gpu_worker.pipeline, "run", run)
    monkeypatch.setattr(gpu_worker, "stop_self", lambda: state.__setitem__("stopped", state["stopped"] + 1) or True)
    monkeypatch.setattr(gpu_worker.time, "sleep", lambda s: None)
    state["settings"] = Settings(out_root=tmp_path, dataset_root=tmp_path, s3_bucket="b", work_dir=tmp_path / "w",
                                 worker_batch=4, worker_poll_seconds=0.0, worker_idle_stop_seconds=1)
    state["module"] = gpu_worker
    return state


def _refs(*ids):
    return [PhotoRef(photo_id=str(i), path=None, preview_key=f"previews/{i}.jpg") for i in ids]


def test_gpu_worker_processes_batches_then_stops_itself_when_idle(fake_worker, monkeypatch):
    w = fake_worker
    w["queue"] = [_refs(1, 2, 3), _refs(4)]
    clock = {"t": 0.0}
    monkeypatch.setattr(w["module"].time, "monotonic", lambda: clock.__setitem__("t", clock["t"] + 0.6) or clock["t"])

    summary = w["module"].loop(w["settings"])

    assert w["runs"] == [["1", "2", "3"], ["4"]]
    assert summary["processed"] == 4 and summary["batches"] == 2 and summary["idleStopped"] is True
    assert w["stopped"] == 1
    assert w["rollbacks"] >= 1                                     # 빈 집기마다 트랜잭션을 닫는다


def test_gpu_worker_rolls_back_failed_batch_and_excludes_poison_photos(fake_worker):
    w = fake_worker
    w["queue"] = [_refs("boom", 2), _refs(7, 8), _refs(9)]

    summary = w["module"].loop(w["settings"], stop_on_idle=False, max_batches=2)

    assert w["runs"][0] == ["boom", "2"] and w["rollbacks"] >= 1   # 예외 → rollback, 루프는 계속
    assert summary["batches"] == 2 and summary["processed"] == 2 and summary["failed"] == 1
    assert w["excludes"][-1] == [7]                                # 실패한 장은 다음 집기에서 뺀다
    assert w["errors"] == [(["7"], "SCORE_FAILED")]                # 그리고 photo_analysis.error 로 남긴다(#85)
    assert w["commits"] >= 1


def test_claim_batch_skips_rows_marked_with_error():
    """wes V15(#85): error 가 찍힌 행은 집지 않는다 — 부분 인덱스 idx_photo_analysis_unscored 와 같은 조건."""
    from score.store import DbStore

    conn = _Conn(rows=[])
    DbStore(SimpleNamespace(), conn).claim_batch(32)
    sql, _ = conn.executed[0]
    assert "a.clip_embedding IS NULL AND a.error IS NULL" in sql


def test_write_errors_upserts_error_and_leaves_commit_to_caller():
    from score.store import DbStore

    conn = _Conn()
    store = DbStore(SimpleNamespace(), conn)
    store.write_errors(["11", "12", "x"], "PREVIEW_MISSING")
    sql, params = conn.executed[0]
    assert "INSERT INTO photo_analysis (photo_id, error" in sql and "ON CONFLICT (photo_id) DO UPDATE" in sql
    assert params == [(11, "PREVIEW_MISSING"), (12, "PREVIEW_MISSING")] and conn.commits == 0
    store.commit()
    assert conn.commits == 1
    store.write_errors([], "SCORE_FAILED")                          # 빈 목록은 SQL 을 내지 않는다
    assert len(conn.executed) == 1


def _client_error(code):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code, "Message": code}}, "HeadObject")


def test_download_previews_drops_missing_keys_but_raises_other_errors(tmp_path):
    from score.gallery import download_previews

    class Storage:
        def download(self, key, dest):
            if key.endswith("/2.jpg"):
                raise _client_error("404")
            if key.endswith("/3.jpg"):
                raise _client_error("Throttling")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"jpg")
            return dest

    missing = []
    got = download_previews(Storage(), _refs(1, 2), tmp_path, workers=1, missing=missing)
    assert [r.photo_id for r in got] == ["1"] and missing == ["2"]           # 404 는 그 장만 빠진다
    with pytest.raises(Exception):
        download_previews(Storage(), _refs(3), tmp_path, workers=2, missing=[])  # 그 외는 예외(일시 오류일 수 있다)
    with pytest.raises(Exception):
        download_previews(Storage(), _refs(2), tmp_path, workers=1)            # missing 없이 부르면 옛 동작


def test_gpu_worker_marks_missing_previews_and_keeps_going(fake_worker, monkeypatch):
    """미리보기가 S3 에 없는 장(#85): 배치를 죽이지 않고 error 를 쓰며, 집은 게 전부 없어도 유휴로 세지 않는다."""
    w = fake_worker

    def download(storage, refs, d, workers=8, missing=None):
        kept = []
        for r in refs:
            (missing.append(r.photo_id) if r.photo_id in ("2", "9") else kept.append(r))
        return kept

    monkeypatch.setattr(w["module"], "download_previews", download)
    w["queue"] = [_refs(1, 2), _refs(9), _refs(3)]

    summary = w["module"].loop(w["settings"], stop_on_idle=False, max_batches=2)

    assert w["runs"] == [["1"], ["3"]]
    assert sorted(w["errors"]) == [(["2"], "PREVIEW_MISSING"), (["9"], "PREVIEW_MISSING")]
    assert summary["processed"] == 2 and summary["failed"] == 2 and summary["batches"] == 2


def test_no_idle_stop_drains_the_queue_then_exits_without_stopping_the_instance(fake_worker, monkeypatch):
    """#103: `--no-idle-stop` 은 EC2 정지만 막는다 — 큐를 끝까지 비우고 유휴가 되면 종료한다.
    옛 동작(종료까지 막아 영원히 도는 것) 때문에 wes 로컬 스크립트가 --once 를 붙여야 했다."""
    w = fake_worker
    w["queue"] = [_refs(1, 2), _refs(3)]
    clock = {"t": 0.0}
    monkeypatch.setattr(w["module"].time, "monotonic", lambda: clock.__setitem__("t", clock["t"] + 0.6) or clock["t"])

    summary = w["module"].loop(w["settings"], stop_on_idle=False)

    assert w["runs"] == [["1", "2"], ["3"]]          # 한 배치가 아니라 큐 전체
    assert summary["processed"] == 3
    assert w["stopped"] == 0                         # StopInstances 는 부르지 않는다
    assert summary["idleStopped"] is False


def test_idle_stop_seconds_zero_keeps_looping(fake_worker, monkeypatch):
    """0 은 "유휴여도 끝나지 않는다" — 종료를 원치 않는 실행의 명시적 손잡이다."""
    w = fake_worker
    w["queue"] = [_refs(1)]
    w["settings"] = Settings(out_root=w["settings"].out_root, dataset_root=w["settings"].dataset_root, s3_bucket="b",
                             work_dir=w["settings"].work_dir, worker_poll_seconds=0.0, worker_idle_stop_seconds=0)
    polls = {"n": 0}

    def sleep(_):
        polls["n"] += 1
        if polls["n"] >= 3:
            raise KeyboardInterrupt                  # 무한 루프를 밖에서 끊는다
    monkeypatch.setattr(w["module"].time, "sleep", sleep)

    with pytest.raises(KeyboardInterrupt):
        w["module"].loop(w["settings"], stop_on_idle=True)

    assert w["stopped"] == 0                         # 0 이면 정지도 종료도 하지 않는다


def test_gpu_worker_once_returns_after_one_batch(fake_worker):
    w = fake_worker
    w["queue"] = [_refs(1), _refs(2)]
    summary = w["module"].loop(w["settings"], once=True)
    # 한 배치만 점수. 미리 잠가 둔 다음 배치는 rollback 으로 돌려준다(실 DB 에서는 행이 다시 미처리로)
    assert summary["batches"] == 1 and w["stopped"] == 0 and w["runs"] == [["1"]] and w["rollbacks"] >= 1


def test_handler_photo_ids_scores_only(fake_handler):
    """운영 계약: photoIds 만 처리하고 끝난다 — 잡·체인·재호출 없음."""
    result = handler.handler({"galleryId": 7, "photoIds": ["5", 6]}, _Context())

    assert fake_handler["run"] == [{"gallery_id": 7, "photo_ids": [5, 6], "remaining": 600.0}]
    assert result["processed"] == 2 and "chained" not in result


def test_job_photo_ids_path_downloads_and_scores(monkeypatch, tmp_path):
    conn = _Conn()
    monkeypatch.setattr(job.db, "connect", lambda s: conn)
    monkeypatch.setattr(job, "PreviewStorage", lambda bucket: None)
    monkeypatch.setattr(job, "load_by_ids", lambda c, ids: _refs(*ids))
    monkeypatch.setattr(job, "download_previews", lambda storage, refs, d, workers=8, missing=None: refs)
    seen = {}
    monkeypatch.setattr(job.pipeline, "run", lambda store, g, refs, settings, force=False, **kw: seen.update(
        force=force, ids=[r.photo_id for r in refs]) or {"processed": len(refs)})

    result = job.run(gallery_id=7, settings=Settings(out_root=tmp_path, dataset_root=tmp_path, s3_bucket="b",
                                                     db_host="h", db_name="d", db_user="u"), photo_ids=[3, 1])

    assert seen == {"force": True, "ids": ["3", "1"]}
    assert result["processed"] == 2 and result["photoIds"] == 2
    assert conn.commits == 1 and result["failed"] == []


def test_job_photo_ids_path_writes_errors_for_missing_and_failed(monkeypatch, tmp_path):
    """Lambda 폴백도 워커와 같은 표시(#85): 404 는 PREVIEW_MISSING, 점수 실패는 SCORE_FAILED."""
    conn = _Conn()
    monkeypatch.setattr(job.db, "connect", lambda s: conn)
    monkeypatch.setattr(job, "PreviewStorage", lambda bucket: None)
    monkeypatch.setattr(job, "load_by_ids", lambda c, ids: _refs(*ids))

    def download(storage, refs, d, workers=8, missing=None):
        missing.append("5")
        return [r for r in refs if r.photo_id != "5"]

    monkeypatch.setattr(job, "download_previews", download)
    monkeypatch.setattr(job.pipeline, "run", lambda store, g, refs, settings, force=False, **kw: {"processed": 1, "failed": ["4"]})

    result = job.run(gallery_id=7, settings=Settings(out_root=tmp_path, dataset_root=tmp_path, s3_bucket="b",
                                                     db_host="h", db_name="d", db_user="u"), photo_ids=[3, 4, 5])

    written = [(params, sql) for sql, params in conn.executed if "error" in sql]
    assert [p for p, _ in written] == [[(5, "PREVIEW_MISSING")], [(4, "SCORE_FAILED")]]
    assert sorted(result["failed"]) == ["4", "5"] and conn.commits == 1


def test_gpu_worker_runs_real_pipeline_with_label_gallery(fake_runners, tmp_path, monkeypatch):
    """#81 회귀: 워커는 gallery 자리에 라벨 "worker" 를 넘긴다 — 실제 pipeline.run 이 DbStore 의 int(gallery) 조회를 타면 안 된다.
    가짜 DbStore 는 실제처럼 read_* 에서 int() 캐스팅을 한다."""
    from score import gpu_worker

    img_root = tmp_path / "imgs"
    img_root.mkdir()
    for i in range(1, 6):
        Image.new("RGB", (32, 32), (10, 20, 30)).save(img_root / f"{i}.jpg")
    queue = [[PhotoRef(photo_id=str(i), path=str(img_root / f"{i}.jpg"), preview_key=f"p/{i}.jpg") for i in (1, 2, 3)],
             [PhotoRef(photo_id=str(i), path=str(img_root / f"{i}.jpg"), preview_key=f"p/{i}.jpg") for i in (4, 5)]]
    written = []

    class Store:
        def __init__(self, settings, connection):
            pass

        def claim_batch(self, n, exclude=None):
            return queue.pop(0) if queue else []

        def read_analysis(self, gallery):
            int(gallery)                                    # 실제 DbStore 와 같은 캐스팅
            return []

        def read_clip_embeddings(self, gallery):
            int(gallery)
            return [], np.zeros((0, 0))

        def write_scores(self, gallery, rows, clips):
            written.extend(r.photo_id for r in rows)

        def rollback(self):
            pass

    monkeypatch.setattr(gpu_worker.db, "connect", lambda s: _Conn())
    monkeypatch.setattr(gpu_worker, "DbStore", Store)
    monkeypatch.setattr(gpu_worker, "PreviewStorage", lambda bucket: None)
    monkeypatch.setattr(gpu_worker, "download_previews", lambda storage, refs, d, workers=8, missing=None: refs)
    monkeypatch.setattr(gpu_worker, "stop_self", lambda: True)
    monkeypatch.setattr(gpu_worker.time, "sleep", lambda s: None)
    settings = Settings(out_root=tmp_path, dataset_root=tmp_path, s3_bucket="b", work_dir=tmp_path / "w",
                        worker_batch=3, worker_poll_seconds=0.0, worker_idle_stop_seconds=0, worker_max_batches=2)

    summary = gpu_worker.loop(settings, stop_on_idle=False)

    assert summary["processed"] == 5 and summary["aborted"] is False and sorted(written) == ["1", "2", "3", "4", "5"]


def test_gpu_worker_aborts_after_consecutive_failures(fake_worker):
    w = fake_worker
    w["queue"] = [_refs("boom")] * 10
    w["settings"] = Settings(out_root=w["settings"].out_root, dataset_root=w["settings"].dataset_root, s3_bucket="b",
                             work_dir=w["settings"].work_dir, worker_poll_seconds=0.0, worker_max_consecutive_failures=3)

    summary = w["module"].loop(w["settings"], stop_on_idle=False)

    assert summary["aborted"] is True and summary["batches"] == 0 and len(w["runs"]) == 3


def test_gpu_worker_stop_self_pins_region_from_imds(monkeypatch):
    """워커 컨테이너에는 AWS_REGION 이 없다 — 리전을 IMDS 에서 읽어 ec2 클라이언트에 명시해야 StopInstances 가 된다
    (2026-09-09 운영 NoRegionError)."""
    from score import gpu_worker

    calls = {}

    class Ec2:
        def stop_instances(self, InstanceIds):
            calls["ids"] = InstanceIds

    def client(service, region_name=None):
        calls["service"], calls["region"] = service, region_name
        return Ec2()

    monkeypatch.setattr(gpu_worker, "_imds", lambda path: {"instance-id": "i-1", "placement/region": "ap-northeast-2"}[path])
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=client))

    assert gpu_worker.stop_self() is True
    assert calls == {"service": "ec2", "region": "ap-northeast-2", "ids": ["i-1"]}


def test_gpu_worker_stop_self_skips_outside_ec2(monkeypatch):
    from score import gpu_worker

    monkeypatch.setattr(gpu_worker, "_imds", lambda path: None)
    monkeypatch.setitem(sys.modules, "boto3", SimpleNamespace(client=lambda *a, **k: pytest.fail("boto3 를 부르면 안 된다")))

    assert gpu_worker.stop_self() is False
