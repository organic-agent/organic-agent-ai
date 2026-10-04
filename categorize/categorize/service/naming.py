"""이름 — 구간을 컨셉(1층)으로 묶고, 1층 안의 그룹(2층)에 이름을 붙여 `concept_assignments` 에 남긴다.

    ① 1층   구간(service/segment.py)마다 대표 4장 2×2 타일을 **한 호출**에 보내 "같은 컨셉끼리 묶고 이름을 지어라".
             시간 모드에 K(conceptCount)가 있으면 "정확히 K개"로 강제한다. 이미지 모드에서는 강제하지 않는다 —
             K를 강제하면 흰 스튜디오 둘을 합치고 야외를 쪼갰다(갤러리 18, ARI 0.59~0.75 → 강제 없이 0.80).
             이름은 갤러리 안에서 서로 달라야 한다 — wes 가 1층을 이름으로 묶는다(AiFolderPlanner).
    ② 2층   1층마다 크기순 커버리지(naming_coverage)까지 대표 1~2장을 1층 이름과 함께 보낸다(1층당 한 호출, 동시에).
             나머지 소그룹은 같은 1층 안 최근접 이름. 그룹이 하나뿐인 1층은 호출하지 않고 `전체`.

Bedrock 호출 = 1층 1회 + 2층 (그룹이 둘 이상인 1층 수)회. 이미지는 1층 ≤ concept_max_units 장, 2층 1층당 ≤ naming_chunk 장.
프롬프트에 컨셉 이름 예시를 주지 않는다 — 스튜디오마다 부르는 이름이 다르고, 사용자가 고친다.
"""

from __future__ import annotations

import io
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from categorize.config.settings import Settings
from categorize.domain.analysis import ConceptAssignment, Store
from categorize.domain.run import Grouped
from categorize.infrastructure.bedrock import LlmClient, jpeg_bytes
from categorize.service.segment import TIME

log = logging.getLogger(__name__)

#: 그룹이 하나뿐인 1층의 2층 이름.
DETAIL_ALL = "전체"

CONCEPT_SYSTEM = """당신은 웨딩 스튜디오 촬영 갤러리를 정리한다. 스튜디오는 '컨셉' 단위로 촬영한다 — 컨셉은 배경 세트와
의상·연출의 조합이고, 한 컨셉을 찍는 동안 구도·포즈·인물 구성·거리(전신/클로즈업)는 계속 바뀐다.

사진은 '구간'으로 나뉘어 있다. 구간마다 대표 4장을 2×2로 붙인 이미지가 오고, 왼쪽 위에 구간 번호가 있다.
구간을 컨셉별로 묶어라.

- 같은 컨셉을 시간을 두고 다시 찍을 수 있다 — 떨어진 구간도 같은 컨셉이면 묶는다.
- 반지·부케 같은 디테일 클로즈업이나 준비 컷 위주 구간은 따로 컨셉을 만들지 말고 가장 가까운 컨셉에 붙인다.
- 판단 근거는 배경 세트(벽·바닥·조명·장소)와 의상이다. 포즈·구도·인물 수는 근거가 아니다.
- 컨셉 이름은 스튜디오 직원이 부를 법한 짧은 한국어(2~10자). 컨셉끼리 이름이 겹치면 안 된다.
- confidence: 그 컨셉 묶음에 대한 확신 0~1.
- 모든 구간을 정확히 한 컨셉에 넣는다."""

DETAIL_SYSTEM = """웨딩 스튜디오 갤러리의 한 컨셉 안을 세부 폴더로 나눈다. 컨셉 이름과, 비슷한 사진 묶음(그룹)마다
대표 사진 1~2장이 온다. 그룹마다 세부 이름을 정한다.

- 세부 이름: 이 컨셉 안에서 그룹을 구별하는 짧은 한국어(2~12자) — 배경의 부분·소품·장소·상황으로 짓는다
  (예: 소파, 케이크 테이블, 계단, 창가). 컨셉 이름을 되풀이하지 않는다.
- 흑백/컬러 같은 스타일이나 인물 구성(신부 단독 등)은 이름에 넣지 않는다.
- 같은 세트로 보이는 그룹에는 같은 이름을 붙인다.
- confidence: 이 이름에 대한 확신 0~1. 대표가 2장인데 서로 다른 세트로 보이면 낮춘다.

모든 그룹에 대해 답한다. 이미지에 보이지 않는 것을 지어내지 않는다."""


def _concept_schema() -> dict:
    # Bedrock InvokeModel 의 output_config 스키마는 number 에 minimum/maximum 을 못 쓴다 — 범위는 프롬프트에 두고 읽는 쪽이 클램프한다.
    return {
        "type": "object",
        "properties": {"concepts": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "segments": {"type": "array", "items": {"type": "integer"}},
                           "confidence": {"type": "number", "description": "0~1"}},
            "required": ["name", "segments", "confidence"], "additionalProperties": False}}},
        "required": ["concepts"], "additionalProperties": False,
    }


def _detail_schema() -> dict:
    return {
        "type": "object",
        "properties": {"groups": {"type": "array", "items": {
            "type": "object",
            "properties": {"group_id": {"type": "integer"}, "detail": {"type": "string"},
                           "confidence": {"type": "number", "description": "0~1"}},
            "required": ["group_id", "detail", "confidence"], "additionalProperties": False}}},
        "required": ["groups"], "additionalProperties": False,
    }


def _clamp(v) -> float:
    return min(1.0, max(0.0, float(v)))


# ── ① 1층 ────────────────────────────────────────────────────────────────────
@dataclass
class Concept:
    name: str
    confidence: float


def _tile(paths: list[str], label: int, edge: int) -> bytes:
    """대표 사진들을 2×2 로 붙이고 왼쪽 위에 구간 번호. 사진이 4장보다 적으면 빈 칸은 회색."""
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    half = edge // 2
    out = Image.new("RGB", (edge, edge), (128, 128, 128))
    for j, p in enumerate(paths[:4]):
        im = Image.open(p)
        im.draft("RGB", (edge, edge))
        im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((half, half))
        out.paste(im, ((j % 2) * half + (half - im.width) // 2, (j // 2) * half + (half - im.height) // 2))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default(size=max(16, edge // 20))
    except TypeError:                      # Pillow < 10.1
        font = ImageFont.load_default()
    draw.rectangle((0, 0, edge // 8, edge // 14), fill=(0, 0, 0))
    draw.text((6, 2), str(label), fill=(255, 255, 0), font=font)
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def _unique(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for n in names:
        n = n.strip() or "컨셉"
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n} {seen[n]}")
    return out


def name_concepts(store: Store, gallery: str, grouped: Grouped, settings: Settings, llm: LlmClient,
                  concept_count: int | None = None) -> tuple[list[int], list[Concept], dict]:
    """구간 → 컨셉. (구간별 컨셉 번호, 컨셉 목록, 결과 요약). 호출 1회."""
    k = settings.knobs
    rows, seg = grouped.rows, grouped.segmentation
    units = seg.units
    picks = {u: [members[int(round(q * (len(members) - 1)))] for q in (0.1, 0.37, 0.63, 0.9)]
             for u, members in enumerate(units)}
    paths = store.preview_paths(gallery, sorted({rows[i].photo_id for p in picks.values() for i in p}))

    force_k = concept_count if (concept_count and seg.mode == TIME) else None
    if force_k:
        head = f"구간 {len(units)}개 (촬영 시각순). 이 촬영의 컨셉은 **정확히 {force_k}개**다."
    elif concept_count:
        head = (f"구간 {len(units)}개 (비슷한 사진끼리 묶었고 순서에는 의미가 없다). 사용자는 컨셉이 {concept_count}개라고 기억한다 — "
                "확실히 같은 컨셉만 합치고, 애매하면 나눠 둬라.")
    else:
        head = f"구간 {len(units)}개. 컨셉 개수는 모른다 — 사진을 보고 정하라."
    parts: list = [("text", head)]
    sent: list[int] = []
    for u, members in enumerate(units):
        files = [paths[rows[i].photo_id] for i in picks[u] if rows[i].photo_id in paths]
        if not files:
            log.warning("[naming] 구간 %d: 대표 사진 미리보기 없음 — 이웃 컨셉으로", u)
            continue
        parts.append(("text", f"[구간 {u}] {len(members)}장"))
        parts.append(("image", _tile(files, u, k.concept_tile_edge)))
        sent.append(u)
    if not sent:
        raise RuntimeError(f"갤러리 {gallery}: 구간 대표 사진을 하나도 받지 못했다")

    out = llm.complete_json(CONCEPT_SYSTEM, parts, _concept_schema(), k.naming_max_tokens)
    raw = [c for c in out.get("concepts", []) if c.get("segments")]
    names = _unique([str(c["name"]) for c in raw])
    concepts = [Concept(name=n, confidence=_clamp(c["confidence"])) for n, c in zip(names, raw)]
    of_unit: dict[int, int] = {}
    for ci, c in enumerate(raw):
        for u in c["segments"]:
            if 0 <= int(u) < len(units) and int(u) not in of_unit:
                of_unit[int(u)] = ci
    if not concepts:
        raise RuntimeError(f"갤러리 {gallery}: VLM 이 컨셉을 하나도 돌려주지 않았다")

    missing = [u for u in range(len(units)) if u not in of_unit]
    if missing:
        log.warning("[naming] 컨셉 응답에 없는 구간 %s — %s", missing,
                    "앞뒤 구간의 컨셉으로" if seg.mode == TIME else "가장 닮은 구간의 컨셉으로")
        E = grouped.E
        cent = {u: E[units[u]].mean(axis=0) for u in of_unit}
        for u in missing:
            if seg.mode == TIME:
                near = [v for v in (u - 1, u + 1, u - 2, u + 2) if v in of_unit]
                v = near[0] if near else next(iter(of_unit))
            else:
                c = E[units[u]].mean(axis=0)
                v = max(cent, key=lambda w: float(cent[w] @ c))
            of_unit[u] = of_unit[v]

    used = sorted(set(of_unit.values()))                 # 구간을 하나도 못 받은 컨셉은 버린다
    remap = {old: new for new, old in enumerate(used)}
    concepts = [concepts[i] for i in used]
    unit_concept = [remap[of_unit[u]] for u in range(len(units))]
    if force_k and len(concepts) != force_k:
        log.warning("[naming] 컨셉 %d개를 요청했는데 %d개가 왔다", force_k, len(concepts))
    summary = {"segmentMode": seg.mode, "segments": len(units), "conceptCountRequested": concept_count,
               "conceptCountForced": bool(force_k), "concepts": [c.name for c in concepts],
               "missingSegments": len(missing)}
    return unit_concept, concepts, summary


# ── ② 2층 ────────────────────────────────────────────────────────────────────
@dataclass
class _EmbedGroup:
    embed_group_id: int
    members: list[int]              # 행 인덱스
    centroid: np.ndarray            # concat 공간, 정규화됨
    sample_row: int                 # 중심 최근접
    far_sample_row: int             # 중심에서 가장 먼 멤버 — spread 클 때 두 번째 대표
    spread: float                   # 그룹 내 평균 중심 거리(1-cos)


def _build_embed_groups(members_by_group: dict[int, list[int]], X: np.ndarray) -> list[_EmbedGroup]:
    """크기 내림차순. 대표 = concat 공간에서 그룹 중심에 가장 가까운 사진."""
    out = []
    for gid, members in members_by_group.items():
        c = X[members].mean(axis=0)
        c = c / max(float(np.linalg.norm(c)), 1e-8)
        sims = X[members] @ c
        out.append(_EmbedGroup(gid, members, c, members[int(np.argmax(sims))], members[int(np.argmin(sims))],
                               float(np.mean(1.0 - sims))))
    out.sort(key=lambda g: (-len(g.members), g.embed_group_id))
    return out


def _sample_images(store: Store, gallery: str, rows, samples: dict[int, list[int]], long_edge: int) -> dict[int, list[bytes]]:
    """대표 사진 → LLM 에 보낼 JPEG. 경로는 한 번에 받고(배치 SELECT + 병렬 다운로드), 축소도 스레드로 겹친다."""
    wanted = sorted({rows[i].photo_id for rr in samples.values() for i in rr})
    paths = store.preview_paths(gallery, wanted) if wanted else {}

    def shrink(pid: str) -> tuple[str, bytes]:
        return pid, jpeg_bytes(paths[pid], long_edge)

    with ThreadPoolExecutor(max_workers=min(8, max(1, len(paths)))) as pool:
        encoded = dict(pool.map(shrink, list(paths)))
    return {gid: [encoded[rows[i].photo_id] for i in rr if rows[i].photo_id in encoded] for gid, rr in samples.items()}


def name_details(store: Store, gallery: str, grouped: Grouped, concepts: list[Concept], concept_of_row: list[int],
                 settings: Settings, llm: LlmClient, job_id: int | None = None) -> dict:
    """1층마다 2층 이름 → concept_assignments. rows 의 embed_group_id 는 채워져 있어야 한다(1층 안에서 전역 유일)."""
    k = settings.knobs
    rows, X = grouped.rows, grouped.X

    by_concept: dict[int, dict[int, list[int]]] = {}
    for i, r in enumerate(rows):
        by_concept.setdefault(concept_of_row[i], {}).setdefault(r.embed_group_id, []).append(i)
    groups_of = {ci: _build_embed_groups(gm, X) for ci, gm in by_concept.items()}

    # VLM 대상 — 1층마다 커버리지까지, 이미지 naming_chunk 장 안에서
    samples: dict[int, list[int]] = {}
    targets: dict[int, list[_EmbedGroup]] = {}
    for ci, gs in groups_of.items():
        if len(gs) < 2:
            continue
        total = sum(len(g.members) for g in gs)
        chosen, covered, n_img = [], 0, 0
        for g in gs:
            reps = [g.sample_row] + ([g.far_sample_row] if g.spread > k.naming_spread_extra and g.far_sample_row != g.sample_row else [])
            if chosen and (covered >= k.naming_coverage * total or n_img + len(reps) > k.naming_chunk):
                break
            chosen.append(g)
            samples[g.embed_group_id] = reps
            covered += len(g.members)
            n_img += len(reps)
        targets[ci] = chosen
    images = _sample_images(store, gallery, rows, samples, k.naming_image_long_edge)

    def one(ci: int) -> tuple[int, dict[int, dict] | None]:
        parts: list = [("text", f"컨셉: {concepts[ci].name}\n그룹 {len(targets[ci])}개의 대표 사진이다.")]
        wanted = set()
        for g in targets[ci]:
            imgs = images.get(g.embed_group_id, [])
            if not imgs:
                continue
            parts.append(("text", f"[그룹 {g.embed_group_id}] {len(g.members)}장" + (" — 대표 2장" if len(imgs) > 1 else "")))
            parts.extend(("image", img) for img in imgs)
            wanted.add(g.embed_group_id)
        if not wanted:
            return ci, None                         # 호출하지 않았다
        out = llm.complete_json(DETAIL_SYSTEM, parts, _detail_schema(), k.naming_max_tokens)
        got = {int(d["group_id"]): d for d in out.get("groups", []) if int(d["group_id"]) in wanted}
        if wanted - set(got):
            log.warning("[naming] 컨셉 '%s' 응답에 그룹 누락 %s — 최근접으로", concepts[ci].name, sorted(wanted - set(got)))
        return ci, got

    # 1층끼리 독립이다 — 동시에 보낸다. 예외는 전파한다(2층 이름이 산출물 자체라 삼키지 않는다).
    with ThreadPoolExecutor(max_workers=max(1, min(k.naming_parallel, len(targets) or 1))) as pool:
        named_of = dict(pool.map(one, list(targets)))

    assignments: list[ConceptAssignment] = []
    counts = {"vlm": 0, "nearest": 0}
    for ci, gs in groups_of.items():
        concept = concepts[ci]
        named = named_of.get(ci) or {}
        named_groups = [g for g in gs if g.embed_group_id in named]
        for g in gs:
            if len(gs) == 1:
                assignments.append(ConceptAssignment(g.embed_group_id, concept.name, DETAIL_ALL, concept.confidence, "vlm"))
                counts["vlm"] += 1
            elif g.embed_group_id in named:
                d = named[g.embed_group_id]
                assignments.append(ConceptAssignment(g.embed_group_id, concept.name, str(d["detail"]).strip() or DETAIL_ALL,
                                                     _clamp(d["confidence"]), "vlm"))
                counts["vlm"] += 1
            elif named_groups:
                sims = np.stack([h.centroid for h in named_groups]) @ g.centroid
                j = int(np.argmax(sims))
                dist = 1.0 - float(sims[j])
                src = named_groups[j]
                # confidence 는 VLM 의 자기 확신이 아니라 1 - 중심 거리다 — 다른 축의 값이 한 컬럼에 온다.
                assignments.append(ConceptAssignment(g.embed_group_id, concept.name, str(named[src.embed_group_id]["detail"]).strip(),
                                                     round(max(0.0, 1.0 - dist), 3), "nearest"))
                counts["nearest"] += 1
            else:                                   # 이 1층의 2층 호출이 아무것도 못 받았다
                assignments.append(ConceptAssignment(g.embed_group_id, concept.name, DETAIL_ALL, concept.confidence, "nearest"))
                counts["nearest"] += 1

    store.write_assignments(gallery, job_id, assignments)
    per_concept: dict[str, list[str]] = {}
    for a in assignments:
        details = per_concept.setdefault(a.concept_name, [])
        if a.detail_name not in details:
            details.append(a.detail_name)
    return {"embedGroups": len(assignments), "vlmGroups": counts["vlm"], "nearestGroups": counts["nearest"],
            "folders": per_concept,
            "llmCalls": sum(1 for v in named_of.values() if v is not None)}
