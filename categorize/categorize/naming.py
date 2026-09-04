"""v3 naming — 임베딩 그룹에 (큰 분류, 컨셉 이름)을 붙여 `ai_concept_assignments`에 남긴다.

ai-folder-structure.md의 ②~④ 구현. 층마다 잘하는 도구:

    ② 이름   크기순으로 사진 커버리지 목표(naming_coverage, 상한 naming_max_groups)까지 고른
             그룹의 대표 1~2장(spread 크면 2장)을 Bedrock Sonnet에 (청크 호출 → 통합 텍스트 호출 1회)
             · 부모 = 닫힌 고정 목록 (JSON 스키마 enum으로 강제, 목록 밖이면 '기타'+proposed)
             · 컨셉 = 열린 이름
    ③ 배정   K 밖 소그룹 → concat 공간에서 이름 붙은 그룹 중심과 최근접. 거리 > τ 면 '기타/기타'
             + needs_review. CLIP 텍스트는 안 쓴다 — 컨셉 층은 텍스트로 못 가른다
    ④ 검증   score 가 사진마다 저장한 CLIP zero-shot 부모 라벨(sub_scores.clip_parent) → 그룹 다수결.
             VLM 부모와 다르거나 confidence < 기준이면 needs_review. 검증 전용 — 판정은 VLM의 것.
             여기서 CLIP 텍스트 인코더를 올리지 않는다 — 이 모듈은 torch 없이 돈다(#26·#35)

Bedrock 이미지 호출은 그룹 수 상한으로 절대 상한이 잡힌다(⌈이미지 수/naming_chunk⌉+1회) —
갤러리가 커져도 비용은 커버리지 목표와 상한이 정한 범위를 넘지 않는다.
FULL 잡의 끝에서도, NAMING 단독 잡에서도 같은 `run()`이 돈다 (분석 재실행 없음 —
clip_embedding을 저장해 둔 이유).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from categorize.config import PARENTS, Settings
from categorize.pipeline import concat_space
from categorize.llm import LlmClient, jpeg_bytes
from categorize.store import ConceptAssignment, Store

log = logging.getLogger(__name__)

ETC = "기타"


def majority(labels: list[str | None]) -> str | None:
    """score 가 사진마다 저장한 clip_parent 의 그룹 다수결. None 은 표에서 뺀다. 표가 없으면 None.
    CLIP 텍스트 인코더는 올리지 않는다 — 이 모듈은 torch 없이 돈다."""
    votes = [lab for lab in labels if lab]
    if not votes:
        return None
    counts: dict[str, int] = {}
    for lab in votes:
        counts[lab] = counts.get(lab, 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], -votes.index(kv[0])))[0]

SYSTEM = """당신은 웨딩 사진 스튜디오의 갤러리 정리를 돕는다. 비슷한 배경·구도로 찍힌 사진
묶음(그룹)마다 대표 사진이 1~2장 온다. 대표가 2장인데 서로 다른 세트로 보이면 그 그룹의
confidence를 낮춰라. 그룹마다 폴더 이름 두 층을 정한다.

- parent(큰 분류): 주어진 목록에서만 고른다. 어느 것에도 맞지 않으면 "기타"를 고르고
  proposed_parent에 더 맞는 큰 분류 이름을 제안한다 (맞는 것이 있으면 proposed_parent는 null).
- concept(컨셉): 촬영 세트를 부를 짧은 한국어 이름 (2~12자). 배경·소품·상황으로 짓는다
  (예: "소파", "케이크 테이블", "해변", "정원 산책"). 흑백/컬러 같은 스타일이나
  인물 구성(신부 단독 등)은 컨셉 이름에 넣지 않는다. 같은 세트로 보이는 그룹들에는 같은
  concept 이름을 붙인다.
- confidence: 이 배정에 대한 확신 0~1.

모든 그룹에 대해 답한다. 이미지에 보이지 않는 것을 지어내지 않는다."""

MERGE_SYSTEM = """웨딩 갤러리 폴더 이름 목록을 정리한다. 같은 촬영 세트·컨셉을 가리키는데 표기만
다른 concept 이름들(예: "소파 세트"와 "쇼파")을 하나의 표준 이름으로 통일한다. parent는 바꾸지
않는다. 다른 세트를 억지로 합치지 않는다. 모든 그룹을 다시 돌려준다."""


def _schema(parents: list[str], with_confidence: bool = True) -> dict:
    props: dict = {
        "group_id": {"type": "integer"},
        "parent": {"type": "string", "enum": parents},
        "proposed_parent": {"type": ["string", "null"]},
        "concept": {"type": "string"},
    }
    required = ["group_id", "parent", "proposed_parent", "concept"]
    if with_confidence:
        # Bedrock InvokeModel의 output_config 스키마는 number에 minimum/maximum을 못 쓴다
        # (400: "properties maximum, minimum are not supported"). 범위는 SYSTEM 프롬프트의
        # "0~1"에 맡기고, 읽는 쪽에서 클램프한다.
        props["confidence"] = {"type": "number", "description": "0~1"}
        required.append("confidence")
    return {
        "type": "object",
        "properties": {"groups": {"type": "array", "items": {
            "type": "object", "properties": props, "required": required,
            "additionalProperties": False}}},
        "required": ["groups"],
        "additionalProperties": False,
    }


@dataclass
class _Group:
    gid: int
    members: list[int]              # ordered-row 인덱스
    centroid: np.ndarray            # concat 공간, 정규화됨
    rep_row: int                    # 대표 사진의 ordered-row 인덱스 (중심 최근접)
    far_row: int                    # 중심에서 가장 먼 멤버 — spread 클 때 두 번째 대표
    spread: float                   # 그룹 내 평균 중심 거리(1-cos). 이질성의 척도


def _build_groups(gids: np.ndarray, X: np.ndarray) -> list[_Group]:
    """크기 내림차순. 대표 = concat 공간에서 그룹 중심에 가장 가까운 사진."""
    by: dict[int, list[int]] = {}
    for i, g in enumerate(gids):
        if int(g) >= 0:
            by.setdefault(int(g), []).append(i)
    out = []
    for gid, members in by.items():
        c = X[members].mean(axis=0)
        c = c / max(float(np.linalg.norm(c)), 1e-8)
        sims = X[members] @ c
        rep = members[int(np.argmax(sims))]
        far = members[int(np.argmin(sims))]
        out.append(_Group(gid=gid, members=members, centroid=c, rep_row=rep,
                          far_row=far, spread=float(np.mean(1.0 - sims))))
    out.sort(key=lambda g: (-len(g.members), g.gid))
    return out


def _vlm_name(llm: LlmClient, chunks: list[list[tuple[_Group, list[bytes]]]], parents: list[str],
              k) -> tuple[dict[int, dict], int]:
    """청크 vision 호출들 → {gid: {parent, proposed_parent, concept, confidence}}, 호출 수."""
    named: dict[int, dict] = {}
    calls = 0
    schema = _schema(parents)
    for chunk in chunks:
        parts: list = [("text", f"큰 분류 목록: {', '.join(parents)}\n그룹 {len(chunk)}개의 대표 사진이다.")]
        for g, imgs in chunk:
            suffix = " — 대표 2장" if len(imgs) > 1 else ""
            parts.append(("text", f"[그룹 {g.gid}] {len(g.members)}장{suffix}"))
            for img in imgs:
                parts.append(("image", img))
        out = llm.complete_json(SYSTEM, parts, schema, k.naming_max_tokens)
        calls += 1
        wanted = {g.gid for g, _ in chunk}
        for item in out.get("groups", []):
            gid = int(item["group_id"])
            if gid in wanted:
                named[gid] = item
        missing = wanted - set(named)
        if missing:
            log.warning("VLM 응답에 그룹 누락: %s — nearest 배정으로 넘긴다", sorted(missing))
    return named, calls


def _merge_names(llm: LlmClient, named: dict[int, dict], sizes: dict[int, int], parents: list[str],
                 k) -> dict[int, dict]:
    """통합 텍스트 호출 1회 — 청크 사이 concept 표기 통일. 실패하면 원본 유지."""
    lines = [f"그룹 {gid}: parent={d['parent']}, concept={d['concept']}, {sizes[gid]}장"
             for gid, d in sorted(named.items())]
    try:
        out = llm.complete_json(MERGE_SYSTEM, "\n".join(lines), _schema(parents, with_confidence=False),
                                k.naming_max_tokens)
    except Exception as exc:  # noqa: BLE001 — 통합은 다듬기다. 실패해도 청크 결과로 간다
        log.warning("이름 통합 호출 실패 (%s) — 청크 결과 유지", exc)
        return named
    for item in out.get("groups", []):
        gid = int(item["group_id"])
        if gid in named and str(item["parent"]) == named[gid]["parent"]:
            named[gid]["concept"] = str(item["concept"])
    return named


def run(store: Store, gallery: str, settings: Settings, llm: LlmClient | None,
        job_id: int | None = None) -> dict:
    if llm is None:
        raise RuntimeError("naming 은 Bedrock 이 필요하다 — --llm 으로 실행하라 (AWS 자격 필요)")
    started = time.monotonic()
    k = settings.knobs
    parents = PARENTS

    rows = [r for r in store.read_analysis(gallery) if r.embed_group_id >= 0]
    if not rows:
        raise RuntimeError(f"갤러리 {gallery}: embed_group_id 가 없다 — FULL(SCORE→CATEGORIZE) 분석이 먼저다")
    emb_ids, E = store.read_embeddings(gallery)
    clip_ids, C = store.read_clip_embeddings(gallery)
    emb_map, clip_map = dict(zip(emb_ids, E)), dict(zip(clip_ids, C))
    rows = [r for r in rows if r.photo_id in emb_map and r.photo_id in clip_map]
    ids = [r.photo_id for r in rows]
    E = np.stack([emb_map[i] for i in ids])
    C = np.stack([clip_map[i] for i in ids])
    X = concat_space(E, C)
    gids = np.array([r.embed_group_id for r in rows], dtype=int)

    groups = _build_groups(gids, X)

    # ② VLM 대상 선정 — 크기 내림차순으로 사진 커버리지 목표까지, 그룹 수 상한 안에서.
    total_photos = sum(len(g.members) for g in groups)
    top: list[_Group] = []
    covered = 0
    for g in groups:
        if len(top) >= k.naming_max_groups:
            break
        if top and covered >= k.naming_coverage * total_photos:
            break
        top.append(g)
        covered += len(g.members)

    # 대표 이미지 — spread 큰(이질적) 그룹은 중심 최근접 + 최원점 2장 (review-v3-design.md (2))
    with_img: list[tuple[_Group, list[bytes]]] = []
    extra_reps = 0
    for g in top:
        rep_rows = [g.rep_row]
        if g.spread > k.naming_spread_extra and g.far_row != g.rep_row:
            rep_rows.append(g.far_row)
        imgs: list[bytes] = []
        for row in rep_rows:
            path = store.preview_path(gallery, rows[row].photo_id)
            if path is not None:
                imgs.append(jpeg_bytes(path, k.naming_image_long_edge))
        if not imgs:
            log.warning("그룹 %d 대표 사진(%s) 이미지 없음 — nearest 배정으로", g.gid, rows[g.rep_row].photo_id)
            continue
        if len(imgs) > 1:
            extra_reps += 1
        with_img.append((g, imgs))

    # 청크는 그룹 단위를 깨지 않으면서 이미지 수(naming_chunk)로 자른다 — 비용 상한의 단위가 이미지라서.
    chunks: list[list[tuple[_Group, list[bytes]]]] = []
    cur: list[tuple[_Group, list[bytes]]] = []
    cur_imgs = 0
    for item in with_img:
        if cur and cur_imgs + len(item[1]) > k.naming_chunk:
            chunks.append(cur)
            cur, cur_imgs = [], 0
        cur.append(item)
        cur_imgs += len(item[1])
    if cur:
        chunks.append(cur)
    named, calls = _vlm_name(llm, chunks, parents, k)
    if len(chunks) > 1 and named:
        named = _merge_names(llm, named, {g.gid: len(g.members) for g in groups}, parents, k)
        calls += 1

    named_groups = [g for g in groups if g.gid in named]
    if not named_groups:
        raise RuntimeError(f"갤러리 {gallery}: VLM 이 어떤 그룹에도 이름을 붙이지 못했다")

    # 배정 만들기 — vlm(K 안) / nearest(K 밖·이미지 없음·응답 누락)
    named_centroids = np.stack([g.centroid for g in named_groups])
    assignments: list[ConceptAssignment] = []
    counts = {"vlm": 0, "nearest": 0, "review": 0}
    for g in groups:
        # ④ 저장된 사진별 CLIP 부모 라벨의 그룹 다수결 — SCORE 가 계산해 둔 것
        clip_parent = majority([rows[i].sub_scores.get("clip_parent") for i in g.members])
        if g.gid in named:
            d = named[g.gid]
            parent, concept = str(d["parent"]), str(d["concept"])
            conf = min(1.0, max(0.0, float(d["confidence"])))
            review = conf < k.review_confidence or (
                parent != ETC and clip_parent is not None and clip_parent != parent)
            assignments.append(ConceptAssignment(
                embed_group_id=g.gid, parent_name=parent, concept_name=concept,
                confidence=conf, assigned_by="vlm",
                proposed_parent=(str(d["proposed_parent"]) if parent == ETC and d.get("proposed_parent") else None),
                clip_parent=clip_parent, needs_review=review))
            counts["vlm"] += 1
        else:
            sims = named_centroids @ g.centroid
            j = int(np.argmax(sims))
            dist = 1.0 - float(sims[j])
            if dist > k.nearest_tau:
                parent, concept, review = ETC, ETC, True
            else:
                src = named[named_groups[j].gid]
                parent, concept = str(src["parent"]), str(src["concept"])
                review = parent != ETC and clip_parent is not None and clip_parent != parent
            assignments.append(ConceptAssignment(
                embed_group_id=g.gid, parent_name=parent, concept_name=concept,
                confidence=round(max(0.0, 1.0 - dist), 3), assigned_by="nearest",
                clip_parent=clip_parent, needs_review=review))
            counts["nearest"] += 1
        if assignments[-1].needs_review:
            counts["review"] += 1

    store.write_assignments(gallery, job_id, assignments)

    per_parent: dict[str, int] = {}
    for a in assignments:
        per_parent[a.parent_name] = per_parent.get(a.parent_name, 0) + 1
    return {
        "gallery": gallery, "pipeline": "v3", "mode": "naming",
        "photos": len(rows), "groups": len(groups),
        "vlmGroups": counts["vlm"], "nearestGroups": counts["nearest"],
        "needsReview": counts["review"], "parents": per_parent,
        "coverage": round(covered / total_photos, 3) if total_photos else 0.0,
        "extraReps": extra_reps,
        "llmCalls": calls, "elapsedSeconds": round(time.monotonic() - started, 1),
    }
