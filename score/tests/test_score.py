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

from score import chain, handler, job, pipeline
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


# ── 샤딩(#54) ─────────────────────────────────────────────────────────────────
def test_shard_select_partitions_in_fixed_order_and_validates():
    refs = list(range(10))
    parts = [job.Shard(i, 4).select(refs) for i in range(4)]
    assert parts == [[0, 4, 8], [1, 5, 9], [2, 6], [3, 7]]
    assert sorted(sum(parts, [])) == refs                       # 빠짐·겹침 없음
    assert job.Shard(0, 1).select(refs) == refs
    with pytest.raises(ValueError):
        job.Shard(4, 4)
    assert job.Shard.from_payload({"index": "2", "total": "4"}) == job.Shard(2, 4)
    assert job.Shard.from_payload(None) is None


def test_plan_shards_by_photo_count_with_cap():
    s = Settings(out_root=Path("o"), dataset_root=Path("d"), shard_photos=250, max_shards=8)
    assert [job.plan_shards(n, s) for n in (0, 1, 250, 251, 822, 5000)] == [1, 1, 1, 2, 4, 8]
    assert job.plan_shards(822, Settings(out_root=Path("o"), dataset_root=Path("d"), shard_photos=0)) == 1


def test_handler_coordinator_fans_out_shards_and_does_not_chain(fake_handler):
    fake_handler["make_run"](coordinator=True, shards=3, targets=700)

    result = handler.handler({"galleryId": 7, "jobId": 3, "force": True}, _Context())

    payloads = fake_handler["invoked"]
    assert [p["shard"] for p in payloads] == [{"index": i, "total": 3} for i in range(3)]
    assert all(p["galleryId"] == 7 and p["jobId"] == 3 and p["force"] is True and p["runStartedAt"] for p in payloads)
    assert len({p["runStartedAt"] for p in payloads}) == 1     # 샤드 셋이 같은 시작 시각
    assert result["fannedOut"] is True and fake_handler["chained"] == [] and fake_handler["reinvoked"] == []


def test_handler_shard_run_chains_only_when_last(fake_handler):
    fake_handler["make_run"](processed=5, stopped=False, remaining=0, lastShard=False)
    handler.handler({"galleryId": 7, "jobId": 3, "shard": {"index": 0, "total": 2}, "runStartedAt": "2026-09-06T00:00:00+00:00"},
                    _Context())
    run = fake_handler["run"][0]
    assert run["shard"] == job.Shard(0, 2) and run["fan_out"] is None and run["run_started_at"] == "2026-09-06T00:00:00+00:00"
    assert fake_handler["chained"] == []

    fake_handler["make_run"](processed=5, stopped=False, remaining=0, lastShard=True)
    result = handler.handler({"galleryId": 7, "jobId": 3, "shard": {"index": 1, "total": 2}}, _Context())
    assert fake_handler["chained"] == [(7, 3)] and result["chained"] is True


def test_reinvoke_payload_keeps_shard_and_run_started_at_but_not_force():
    p = handler._payload(7, 3, force=False, shard=job.Shard(1, 4), run_started_at="2026-09-06T00:00:00+00:00")
    assert p == {"galleryId": 7, "jobId": 3, "force": False, "shard": {"index": 1, "total": 4},
                 "runStartedAt": "2026-09-06T00:00:00+00:00"}
    assert handler._payload(7, None, force=False, shard=None, run_started_at=None) == {"galleryId": 7, "force": False}


def test_since_rescoring_skips_only_rows_written_after_run_start(tmp_path, fake_runners):
    """force 실행의 시작 시각(since) 이후에 쓴 점수만 '있음' — 재호출이 force 를 잃어도 옛 점수는 다시 계산한다."""
    from datetime import datetime, timedelta, timezone

    store, refs, scored, settings = _world(tmp_path, n=8)   # 앞 4장은 옛 점수 (_world 가 지금 막 썼다)
    since = datetime.now(timezone.utc)                        # 이 시각 이후 점수만 "있음"

    first = pipeline.run(store, "g", refs, settings, since=since)      # 옛 점수 4장도 대상
    assert first["processed"] == 8 and first["skipped"] == 0

    again = pipeline.run(store, "g", refs, settings, since=since)      # 방금 쓴 8장은 since 이후 → 전부 건너뜀
    assert again["skipped"] == 8 and again["processed"] == 0

    later = pipeline.run(store, "g", refs, settings, since=datetime.now(timezone.utc) + timedelta(seconds=5))
    assert later["processed"] == 8                                      # 미래 시각 기준이면 다시 전부


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


# ── chain ─────────────────────────────────────────────────────────────────────
def test_chain_without_executor_returns_false(tmp_path):
    settings = Settings(out_root=tmp_path, dataset_root=tmp_path)
    assert settings.chain_configured is False
    assert chain.invoke_categorize(settings, 7, 3) is False


def test_chain_subprocess_passes_gallery_and_job(tmp_path, monkeypatch):
    calls = []

    class _Proc:
        pid = 42

    monkeypatch.setattr(chain.subprocess, "Popen", lambda argv: calls.append(argv) or _Proc())
    settings = Settings(out_root=tmp_path, dataset_root=tmp_path, categorize_command="python -m categorize")

    assert chain.invoke_categorize(settings, 7, 3) is True
    assert calls == [["python", "-m", "categorize", "--gallery-id", "7", "--job-id", "3"]]


# ── handler ───────────────────────────────────────────────────────────────────
class _Context:
    function_name = "wes-score"

    def __init__(self, remaining_ms: int = 600_000) -> None:
        self.remaining_ms = remaining_ms

    def get_remaining_time_in_millis(self) -> int:
        return self.remaining_ms


@pytest.fixture
def fake_handler(monkeypatch):
    calls = {"run": [], "reinvoked": [], "chained": [], "failed": [], "invoked": []}

    def make_run(**result):
        def run(gallery_id, force, settings, job_id, remaining_seconds, shard=None, run_started_at=None, fan_out=None):
            calls["run"].append({"gallery_id": gallery_id, "force": force, "job_id": job_id,
                                 "remaining": remaining_seconds() if remaining_seconds else None,
                                 "shard": shard, "run_started_at": run_started_at, "fan_out": fan_out})
            if result.get("coordinator"):
                result["fannedOut"] = fan_out(result["shards"], run_started_at)
            return {"gallery": str(gallery_id), **result}
        monkeypatch.setattr(handler.job, "run", run)

    monkeypatch.setattr(handler, "reinvoke",
                        lambda ctx, g, j, shard=None, run_started_at=None: calls["reinvoked"].append((g, j)) or True)
    monkeypatch.setattr(handler, "_invoke_self", lambda ctx, g, payload: calls["invoked"].append(payload) or True)
    monkeypatch.setattr(handler.chain, "invoke_categorize", lambda s, g, j: calls["chained"].append((g, j)) or True)
    monkeypatch.setattr(handler, "_fail_job", lambda j, e: calls["failed"].append(j))
    calls["make_run"] = make_run
    return calls


def test_handler_passes_job_and_deadline_then_chains(fake_handler):
    fake_handler["make_run"](processed=3, stopped=False, remaining=0)

    result = handler.handler({"galleryId": "7", "jobId": "3", "force": True}, _Context(123_000))

    run = fake_handler["run"][0]
    assert (run["gallery_id"], run["force"], run["job_id"], run["remaining"], run["shard"]) == (7, True, 3, 123.0, None)
    assert run["run_started_at"] and run["fan_out"] is not None     # force → 시작 시각, wes 호출 → 조정자 가능
    assert fake_handler["chained"] == [(7, 3)] and result["chained"] is True
    assert fake_handler["reinvoked"] == []


def test_handler_reinvokes_when_stopped_with_progress(fake_handler):
    fake_handler["make_run"](processed=8, stopped=True, remaining=40)

    result = handler.handler({"galleryId": 7, "jobId": 3}, _Context())

    assert fake_handler["reinvoked"] == [(7, 3)] and result["reinvoked"] is True
    assert fake_handler["chained"] == []          # 아직 끝나지 않았다 — 체인은 마지막 호출이


def test_handler_does_not_reinvoke_without_progress(fake_handler):
    fake_handler["make_run"](processed=0, stopped=True, remaining=40, failed=["a"])

    result = handler.handler({"galleryId": 7}, _Context())

    assert fake_handler["reinvoked"] == [] and "reinvoked" not in result


def test_handler_without_job_does_not_chain(fake_handler):
    fake_handler["make_run"](processed=1, stopped=False, remaining=0)

    result = handler.handler({"galleryId": 7}, None)

    assert fake_handler["chained"] == [] and "chained" not in result
    assert fake_handler["run"][0]["remaining"] is None


def test_handler_fails_job_when_chain_fails(fake_handler, monkeypatch):
    fake_handler["make_run"](processed=1, stopped=False, remaining=0)
    monkeypatch.setattr(handler.chain, "invoke_categorize", lambda s, g, j: False)

    result = handler.handler({"galleryId": 7, "jobId": 3}, _Context())

    assert result["chained"] is False and fake_handler["failed"] == [3]


def test_reinvoke_without_function_name_returns_false():
    assert handler.reinvoke(SimpleNamespace(), 7, 3) is False


def test_was_skipped_distinguishes_lock_skip_from_all_photos_skipped():
    # 잠금 건너뜀은 문자열, 정상 실행은 "건너뛴 사진 수"(int)다. 전부 건너뛴 재실행(processed 0, skipped N)은
    # 끝난 실행이라 체인이 열려야 한다 — 진위 검사로 오판하던 회귀를 막는다.
    assert job.was_skipped({"skipped": job.ALREADY_RUNNING, "processed": 0})
    assert not job.was_skipped({"skipped": 822, "processed": 0, "stopped": False})
    assert not job.was_skipped({"skipped": 0, "processed": 822})
    assert not job.was_skipped({})


def test_handler_chains_when_all_photos_already_scored(fake_handler):
    # 재실행: 822장 전부 이미 점수가 있어 processed 0, skipped 822. 끝난 실행이므로 categorize 를 이어 불러야 한다.
    fake_handler["make_run"](processed=0, skipped=822, stopped=False, remaining=0)

    result = handler.handler({"galleryId": 7, "jobId": 3}, _Context())

    assert fake_handler["chained"] == [(7, 3)] and result["chained"] is True


def test_handler_returns_early_on_lock_skip(fake_handler):
    fake_handler["make_run"](processed=0, skipped=job.ALREADY_RUNNING, stopped=False, remaining=0)

    result = handler.handler({"galleryId": 7, "jobId": 3}, _Context())

    assert fake_handler["chained"] == [] and "chained" not in result


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

    def run(store, gallery, refs, settings, force=False, scorer=None, **kw):
        state["runs"].append([r.photo_id for r in refs])
        if any(r.photo_id == "boom" for r in refs):
            raise RuntimeError("batch boom")
        failed = [r.photo_id for r in refs if r.photo_id == "7"]
        return {"processed": len(refs) - len(failed), "failed": failed}

    monkeypatch.setattr(gpu_worker.db, "connect", lambda s: _Conn())
    monkeypatch.setattr(gpu_worker, "DbStore", Store)
    monkeypatch.setattr(gpu_worker, "PreviewStorage", lambda bucket: None)
    monkeypatch.setattr(gpu_worker, "download_previews", lambda storage, refs, d, workers=8: refs)
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


def test_gpu_worker_once_returns_after_one_batch(fake_worker):
    w = fake_worker
    w["queue"] = [_refs(1), _refs(2)]
    summary = w["module"].loop(w["settings"], once=True)
    # 한 배치만 점수. 미리 잠가 둔 다음 배치는 rollback 으로 돌려준다(실 DB 에서는 행이 다시 미처리로)
    assert summary["batches"] == 1 and w["stopped"] == 0 and w["runs"] == [["1"]] and w["rollbacks"] >= 1


def test_handler_photo_ids_scores_only_without_job_or_chain(fake_handler, monkeypatch):
    calls = []

    def run(gallery_id, settings, remaining_seconds=None, photo_ids=None, **kw):
        calls.append({"gallery_id": gallery_id, "photo_ids": photo_ids, "kw": kw})
        return {"gallery": str(gallery_id), "processed": len(photo_ids), "photoIds": len(photo_ids), "stopped": True}
    monkeypatch.setattr(handler.job, "run", run)

    result = handler.handler({"galleryId": 7, "jobId": 3, "photoIds": ["5", 6]}, None)

    assert calls == [{"gallery_id": 7, "photo_ids": [5, 6], "kw": {}}]
    assert fake_handler["chained"] == [] and fake_handler["reinvoked"] == [] and "chained" not in result


def test_job_photo_ids_path_downloads_scores_and_skips_lock_and_jobs(monkeypatch, tmp_path):
    from score import gallery as gallery_mod

    conn = _Conn()
    monkeypatch.setattr(job.db, "connect", lambda s: conn)
    monkeypatch.setattr(job, "try_lock_gallery", lambda *a, **k: (_ for _ in ()).throw(AssertionError("잠금 없음")))
    monkeypatch.setattr(job.jobs, "start", lambda *a, **k: (_ for _ in ()).throw(AssertionError("잡 없음")))
    monkeypatch.setattr(job, "PreviewStorage", lambda bucket: None)
    monkeypatch.setattr(gallery_mod, "load_by_ids", lambda c, ids: _refs(*ids))
    monkeypatch.setattr(job, "download_previews", lambda storage, refs, d, workers=8: refs)
    seen = {}
    monkeypatch.setattr(job.pipeline, "run", lambda store, g, refs, settings, force=False, **kw: seen.update(
        force=force, ids=[r.photo_id for r in refs]) or {"processed": len(refs)})

    result = job.run(gallery_id=7, settings=Settings(out_root=tmp_path, dataset_root=tmp_path, s3_bucket="b",
                                                     db_host="h", db_name="d", db_user="u"), photo_ids=[3, 1])

    assert seen == {"force": True, "ids": ["3", "1"]}
    assert result["processed"] == 2 and result["photoIds"] == 2


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
    monkeypatch.setattr(gpu_worker, "download_previews", lambda storage, refs, d, workers=8: refs)
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
