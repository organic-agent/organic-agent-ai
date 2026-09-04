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

from score import chain, handler, pipeline
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

    def embed(self, path):
        self.embed_calls += 1
        return _unit(self.rng.normal(size=self.dim))

    def embed_texts(self, prompts):
        return np.stack([_unit(self.rng.normal(size=self.dim)) for _ in prompts])

    def score_from_embedding(self, emb):
        return 5.5


class _FakeArniqa:
    def score(self, path):
        return {"technical_score": 0.6}


@pytest.fixture
def fake_runners(monkeypatch):
    """pipeline.run 이 함수 안에서 import 하는 torch 러너·classical 을 가짜로 바꾼다."""
    laion = _FakeLaion()
    runners = types.ModuleType("score.runners")
    runners.LaionRunner = lambda: laion
    runners.ArniqaRunner = lambda: _FakeArniqa()
    classical = types.ModuleType("score.classical")
    classical.measure = lambda path: {"sharpness": 100.0, "highlight_clip": 0.0, "shadow_clip": 0.0, "mean_luma": 120.0}
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
    calls = {"run": [], "reinvoked": [], "chained": [], "failed": []}

    def make_run(**result):
        def run(gallery_id, force, settings, job_id, remaining_seconds):
            calls["run"].append({"gallery_id": gallery_id, "force": force, "job_id": job_id,
                                 "remaining": remaining_seconds() if remaining_seconds else None})
            return {"gallery": str(gallery_id), **result}
        monkeypatch.setattr(handler.job, "run", run)

    monkeypatch.setattr(handler, "reinvoke", lambda ctx, g, j: calls["reinvoked"].append((g, j)) or True)
    monkeypatch.setattr(handler.chain, "invoke_categorize", lambda s, g, j: calls["chained"].append((g, j)) or True)
    monkeypatch.setattr(handler, "_fail_job", lambda j, e: calls["failed"].append(j))
    calls["make_run"] = make_run
    return calls


def test_handler_passes_job_and_deadline_then_chains(fake_handler):
    fake_handler["make_run"](processed=3, stopped=False, remaining=0)

    result = handler.handler({"galleryId": "7", "jobId": "3", "force": True}, _Context(123_000))

    assert fake_handler["run"] == [{"gallery_id": 7, "force": True, "job_id": 3, "remaining": 123.0}]
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
