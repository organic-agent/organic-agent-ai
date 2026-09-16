"""잡 — handler(Lambda) 와 __main__(CLI) 이 같은 함수를 부른다.

run_train   마감된(CLOSED) 라벨 갤러리를 전부 모아 → 홀드아웃 + 게이트 → 전체로 학습 → 저장.
            게이트를 통과한 실행만 active. 테이블이 없으면 로컬 파일에만 남긴다.
run_sanity  갤러리 하나 in-sample — 학습 → 같은 갤러리 재정렬 → 양성이 후보 범위에 드는가 (요구사항 (1)).
            성능 지표가 아니라 파이프라인 확인이다. prior 단독 값을 함께 남겨 훗날 비교의 첫 줄로 쓴다.
"""

from __future__ import annotations

import logging
import time

from preference.config.settings import Settings
from preference.domain.features import SCALAR_NAMES
from preference.domain.gallery import LabeledGallery
from preference.repository.db_store import DbStore
from preference.repository.local_store import LocalStore
from preference.repository.store import Store
from preference.service.evaluate import evaluate_gallery, gate
from preference.service.features import build
from preference.service.train import make_sample, train

log = logging.getLogger(__name__)


def _load(store: Store, gallery_ids: list[str], selected_override: dict[str, list[str]] | None = None) -> list[LabeledGallery]:
    out = []
    for g in gallery_ids:
        data = store.read_gallery(g)
        selected = (selected_override or {}).get(g) or store.read_selected(g)
        if not selected:
            log.warning("갤러리 %s: 선택 사진이 없어 건너뛴다", g)
            continue
        out.append(LabeledGallery(data=data, selected_ids=selected))
    return out


def run_train(store: Store, settings: Settings, gallery_ids: list[str] | None = None,
              fallback: LocalStore | None = None) -> dict:
    knobs = settings.knobs
    ids = gallery_ids or store.list_closed_galleries()
    galleries = _load(store, ids)
    if not galleries:
        return {"status": "skipped", "reason": "라벨 갤러리 없음", "gallery_ids": ids}
    versions = {(lg.data.embedding_model, lg.data.model_version) for lg in galleries}
    if len(versions) > 1:
        raise SystemExit(f"갤러리들의 모델 버전이 섞여 있다 — 한 행에 담을 수 없다: {versions}")

    t0 = time.time()
    feats = {lg.data.gallery_id: build(lg.data) for lg in galleries}
    verdict = gate(galleries, knobs, feats)
    model = train(galleries, knobs, feats)
    model.holdout = verdict
    model.active = bool(verdict["passed"])
    row = model.to_row()
    model_id = store.write_model(row)
    where = "db" if isinstance(store, DbStore) else "local"
    if model_id is None and fallback is not None:
        model_id = fallback.write_model(row)
        where = "local"
    elapsed = time.time() - t0
    log.info("학습 끝 — 갤러리 %d · 양성 %d · λ=%.3f · 게이트 %s · %s#%s · %.1fs",
             model.n_galleries, model.n_positives, model.lam, verdict["passed"], where, model_id, elapsed)
    return {
        "status": "ok", "model_id": model_id, "stored": where, "n_galleries": model.n_galleries,
        "n_positives": model.n_positives, "lambda": model.lam, "active": model.active,
        "gate": {k: verdict[k] for k in ("passed", "reasons", "n_galleries") if k in verdict}
                | {k: verdict[k] for k in ("mean_diff", "p_value", "worse_ratio") if k in verdict},
        "elapsed_s": round(elapsed, 1),
    }


def run_sanity(store: Store, settings: Settings, gallery_id: str, selected_ids: list[str]) -> dict:
    knobs = settings.knobs
    lg = _load(store, [gallery_id], {gallery_id: selected_ids})[0]
    f = build(lg.data)
    s = make_sample(lg)
    t0 = time.time()
    model = train([lg], knobs, {gallery_id: f})
    fit_s = time.time() - t0
    full = evaluate_gallery(lg, model, knobs, f, lam_override=1.0)      # 학습된 가중치로 재정렬 (요구사항 (1))
    damped = evaluate_gallery(lg, model, knobs, f)                      # λ(n=1) 그대로 — 연결됐을 때의 실제 점수
    pos = lg.positive_idx
    cid = lg.data.cluster_id
    pos_clusters = {int(c) for c in cid[pos] if c >= 0}
    in_pos_clusters = int(sum(1 for c in cid if c >= 0 and int(c) in pos_clusters))
    return {
        "gallery_id": gallery_id, "n_photos": lg.data.n,
        "n_selected_given": len(selected_ids), "n_positives_matched": int(pos.size),
        "n_negatives": int((s.y == 0).sum()), "n_excluded_siblings": in_pos_clusters - int(sum(1 for i in pos if cid[i] >= 0)),
        "k": full.k, "lambda_n1": model.lam, "fit_seconds": round(fit_s, 2),
        "metrics_lambda_1": full.metrics, "metrics_lambda_n": damped.metrics,
        "passed": full.metrics["pref"]["recall_cluster"] >= 0.999,   # 클러스터 기준 — 형제 중 어느 장인지는 이 특징으로 못 가른다
        "w_scalar": {name: round(float(v), 4) for name, v in zip(SCALAR_NAMES, model.w_scalar)},
        "embedding_model": lg.data.embedding_model, "model_version": lg.data.model_version,
    }
