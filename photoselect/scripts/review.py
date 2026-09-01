"""[개발 도구 — 런타임 아님] 초안을 HTML 한 장으로 — 사진·점수·이유를 나란히 놓고, 클릭으로 담기/별점/쌍 비교를 기록해
`evidence.json`으로 내보낸다. 그 파일을 다음 `draft` 라운드가 읽는다.

즉 이 페이지가 **2단계(개인화) 테스트 데이터의 생산 도구**다. 실제 고객이 없어도 팀원이
이 페이지로 30장을 보고 반응하면 그게 곧 evidence다. 브라우저 localStorage에 쌓이고,
"내보내기" 버튼이 JSON을 내려준다. 서비스에서는 프론트+DB가 이 역할을 한다.

    python scripts/review.py --gallery "dataset1/류지혜고객님 (2)" [--round N] [--pairs 12] [--seed S]
"""

from __future__ import annotations

import base64
import html
import io
import json
import sys
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # pip install -e 없이도 돈다

from photoselect.v1.axes import AXES  # noqa: E402 — 온보딩 쌍(v1 전용)에만 쓴다
from photoselect.v1.store import PhotoAnalysis, Recommendation  # noqa: E402 — 타입 힌트용. v2 행도 같은 필드를 갖는다

THUMB = 420


def _thumb_b64(path: str) -> str:
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        img.thumbnail((THUMB, THUMB))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return ""


def render(gallery: str, recs: list[Recommendation], analysis: dict[str, PhotoAnalysis],
           paths: dict[str, str], pair_candidates: list[tuple[str, str, str]], out: Path,
           summary: dict) -> Path:
    cards = []
    for r in recs:
        a = analysis.get(r.photo_id)
        bd = r.score_breakdown
        tags = " · ".join(f"{ax}={getattr(a, ax)}" for ax in AXES) if a else ""
        cards.append(f"""
<div class="card" data-id="{html.escape(r.photo_id)}">
  <img src="{_thumb_b64(paths.get(r.photo_id, ''))}" loading="lazy">
  <div class="meta">
    <div class="rank">#{r.rank} <span class="pid">{html.escape(r.photo_id.split('/')[-1])}</span></div>
    <div class="reason">{html.escape(r.reason)}</div>
    <div class="bd">score {bd.get('score')} · prior {bd.get('prior_z')} · pref {bd.get('pref_z')} ·
      tech {bd.get('technical_pct')} · aes {bd.get('aesthetic_pct')} · cluster {bd.get('cluster_id')}</div>
    <div class="tags">{html.escape(tags)}</div>
    <div class="ctl">
      <button class="sel">담기</button>
      <span class="stars">{''.join(f'<button class="star" data-v="{v}">{v}</button>' for v in range(1, 6))}</span>
    </div>
  </div>
</div>""")

    pairs_html = []
    for a, b, axis in pair_candidates:
        pairs_html.append(f"""
<div class="pair" data-a="{html.escape(a)}" data-b="{html.escape(b)}" data-axis="{axis}">
  <div class="axis">{axis}: {html.escape(getattr(analysis[a], axis))} vs {html.escape(getattr(analysis[b], axis))}</div>
  <div class="two">
    <img data-pick="a" src="{_thumb_b64(paths.get(a, ''))}" loading="lazy">
    <img data-pick="b" src="{_thumb_b64(paths.get(b, ''))}" loading="lazy">
  </div>
</div>""")

    doc = f"""<!doctype html><meta charset="utf-8"><title>photoselect · {html.escape(gallery)}</title>
<style>
body{{font-family:-apple-system,sans-serif;margin:16px;background:#111;color:#eee}}
h1{{font-size:18px}} .sum{{font-size:12px;color:#aaa;white-space:pre-wrap}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px}}
.card{{background:#1c1c1c;border-radius:8px;overflow:hidden;border:2px solid transparent}}
.card.on{{border-color:#4caf50}} .card img{{width:100%;display:block}}
.meta{{padding:8px;font-size:12px}} .rank{{font-weight:600}} .pid{{color:#888;font-weight:400}}
.reason{{margin:4px 0;color:#ffd54f}} .bd,.tags{{color:#999;font-size:11px}}
.ctl{{margin-top:6px}} button{{cursor:pointer;background:#333;color:#eee;border:0;border-radius:4px;padding:3px 8px;margin-right:4px}}
button.on{{background:#4caf50}} .star.on{{background:#ff9800}}
.pair{{margin:12px 0;background:#1c1c1c;padding:8px;border-radius:8px}} .two{{display:flex;gap:8px}}
.two img{{width:49%;cursor:pointer;border:3px solid transparent}} .two img.on{{border-color:#4caf50}}
.axis{{font-size:12px;color:#aaa;margin-bottom:6px}}
#bar{{position:sticky;top:0;background:#111;padding:8px 0;border-bottom:1px solid #333;z-index:9}}
</style>
<div id="bar">
<h1>photoselect · {html.escape(gallery)} · round {recs[0].round if recs else '-'}</h1>
<div class="sum">{html.escape(json.dumps(summary, ensure_ascii=False, indent=1))}</div>
<button id="export">evidence.json 내보내기</button> <button id="clear">전부 지우기</button>
<span id="cnt" style="font-size:12px;color:#aaa"></span>
</div>
<h2 style="font-size:15px">이번 라운드 {len(recs)}장 — 가져갈 사진은 '담기'. 여기 나온 사진은 다음 라운드에 다시 안 나온다</h2>
<div class="grid">{''.join(cards)}</div>
{('<h2 style="font-size:15px;margin-top:24px">온보딩 쌍 비교 ' + str(len(pairs_html)) + '개 — 더 마음에 드는 쪽을 클릭 (첫 라운드 한 번만)</h2>' + ''.join(pairs_html)) if pairs_html else ''}
<script>
const KEY='photoselect:'+{json.dumps(gallery)};
let st=JSON.parse(localStorage.getItem(KEY)||'{{"selected":[],"ratings":{{}},"pairs":[]}}');
function save(){{localStorage.setItem(KEY,JSON.stringify(st));paint();}}
function paint(){{
  document.querySelectorAll('.card').forEach(c=>{{const id=c.dataset.id;
    c.classList.toggle('on',st.selected.includes(id));
    c.querySelector('.sel').classList.toggle('on',st.selected.includes(id));
    c.querySelectorAll('.star').forEach(s=>s.classList.toggle('on',+s.dataset.v<=(st.ratings[id]||0)));}});
  document.querySelectorAll('.pair').forEach(p=>{{const hit=st.pairs.find(x=>x[0]===p.dataset.a&&x[1]===p.dataset.b||x[0]===p.dataset.b&&x[1]===p.dataset.a);
    p.querySelectorAll('img').forEach(i=>i.classList.toggle('on',!!hit&&hit[0]===(i.dataset.pick==='a'?p.dataset.a:p.dataset.b)));}});
  document.getElementById('cnt').textContent=` 담김 ${{st.selected.length}} · 별점 ${{Object.keys(st.ratings).length}} · 쌍 ${{st.pairs.length}}`;}}
document.querySelectorAll('.sel').forEach(b=>b.onclick=()=>{{const id=b.closest('.card').dataset.id;
  st.selected=st.selected.includes(id)?st.selected.filter(x=>x!==id):[...st.selected,id];save();}});
document.querySelectorAll('.star').forEach(b=>b.onclick=()=>{{const id=b.closest('.card').dataset.id;
  st.ratings[id]=+b.dataset.v;save();}});
document.querySelectorAll('.two img').forEach(i=>i.onclick=()=>{{const p=i.closest('.pair');
  const ch=i.dataset.pick==='a'?p.dataset.a:p.dataset.b, rj=i.dataset.pick==='a'?p.dataset.b:p.dataset.a;
  st.pairs=st.pairs.filter(x=>!(x[0]===p.dataset.a&&x[1]===p.dataset.b||x[0]===p.dataset.b&&x[1]===p.dataset.a));
  st.pairs.push([ch,rj,p.dataset.axis]);save();}});
document.getElementById('export').onclick=()=>{{const a=document.createElement('a');
  a.href='data:application/json;charset=utf-8,'+encodeURIComponent(JSON.stringify(st,null,1));
  a.download='evidence.json';a.click();}};
document.getElementById('clear').onclick=()=>{{if(confirm('지울까요?')){{st={{"selected":[],"ratings":{{}},"pairs":[]}};save();}}}};
paint();
</script>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def onboarding_pairs(rows: list[PhotoAnalysis], n: int = 12, pct_gap: float = 15.0, seed: int = 0,
                     exclude: set[frozenset] | None = None) -> list[tuple[str, str, str]]:
    """온보딩 쌍 후보 — plan.md §3-B 규칙: 같은 scene · 한 축만 다름 · 품질 근접 · 다른 클러스터.

    관측 태그로 판정하므로 태그 오차만큼 샌다(study/01 step3 [E]). 그래도 이 규칙이
    랜덤 쌍보다 훨씬 낫다는 것은 확인됐다(step2 [C]).
    """
    import random

    axes = [a for a in AXES if a != "scene"]
    cand = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            if a.scene != b.scene or a.cluster_id == b.cluster_id:
                continue
            diff = [ax for ax in axes if getattr(a, ax) != getattr(b, ax)]
            if len(diff) != 1:
                continue
            if abs(a.technical_pct - b.technical_pct) > pct_gap or abs(a.aesthetic_pct - b.aesthetic_pct) > pct_gap:
                continue
            if exclude and frozenset((a.photo_id, b.photo_id)) in exclude:
                continue          # 이미 답한 쌍은 다시 묻지 않는다
            cand.append((a.photo_id, b.photo_id, diff[0]))
    rng = random.Random(seed)
    rng.shuffle(cand)
    # 축마다 골고루
    out, per_axis = [], {ax: 0 for ax in axes}
    quota = max(1, n // len(axes))
    for c in cand:
        if per_axis[c[2]] < quota:
            out.append(c)
            per_axis[c[2]] += 1
        if len(out) >= n:
            break
    for c in cand:
        if len(out) >= n:
            break
        if c not in out:
            out.append(c)
    return out[:n]


def main(argv: list[str] | None = None) -> None:
    import argparse
    import sys as _sys

    ap = argparse.ArgumentParser(prog="review", description="초안 검수 HTML 생성 (로컬 LocalStore 전용)")
    ap.add_argument("--pipeline", choices=("v1", "v2"), default="v2", help="어느 버전의 out/ 을 읽나 (기본 v2)")
    ap.add_argument("--gallery", required=True)
    ap.add_argument("--round", type=int, help="기본: 마지막 라운드")
    ap.add_argument("--pairs", type=int, help="온보딩 쌍 비교 개수. 기본: 1라운드만 12, 이후 0")
    ap.add_argument("--seed", type=int, help="쌍 후보 시드. 기본: 라운드 번호 → 라운드마다 새 쌍, 이미 답한 쌍은 제외")
    args = ap.parse_args(argv)

    import importlib
    pipe = importlib.import_module(f"photoselect.{args.pipeline}")
    gal, store_mod, Settings = pipe.gallery, pipe.store, pipe.Settings
    settings = Settings.from_env()
    target_count = settings.score.target_count if args.pipeline == "v1" else settings.v2.target_count
    st = store_mod.LocalStore(settings.out_root)
    recs = st.read_recommendations(args.gallery)
    if not recs:
        _sys.exit("추천이 없다 — 먼저 draft를 돌릴 것")
    rnd = args.round or max(r.round for r in recs)
    recs = sorted([r for r in recs if r.round == rnd], key=lambda r: r.rank)
    rows = st.read_analysis(args.gallery)
    analysis = {r.photo_id: r for r in rows}
    paths = {ref.photo_id: ref.path for ref in gal.load_local(settings.dataset_root, args.gallery, jpg_only=False)}
    answered = {frozenset(p[:2]) for p in st.read_evidence(args.gallery).pairs}
    n_pairs = args.pairs if args.pairs is not None else (12 if rnd == 1 and args.pipeline == "v1" else 0)   # 온보딩 쌍은 v1 전용
    pairs = onboarding_pairs(rows, n=n_pairs, seed=args.seed if args.seed is not None else rnd, exclude=answered)
    ev = st.read_evidence(args.gallery)
    summary = {"round": rnd, "k": len(recs), "photos": len(rows),
               "selected": f"{len(ev.selected)}/{target_count}",
               "scenes": {s: sum(1 for r in recs if analysis[r.photo_id].scene == s)
                          for s in sorted({analysis[r.photo_id].scene for r in recs})}}
    out = render(args.gallery, recs, analysis, paths, pairs,
                 st._dir(args.gallery) / f"review-r{rnd}.html", summary)
    print(f"→ {out}\n   open '{out}'   (채점 후 '내보내기' → 그 파일을 같은 폴더에 evidence.json 으로)")


if __name__ == "__main__":
    main()
