"""VLM 태그 채점 시트 — 사람이 눈으로 매기는 로컬 HTML을 만든다.

────────────────────────────────────────────────────────────────────────────
왜 필요한가

`vlm_tag.py`가 뽑은 태그가 **맞는지는 모델이 알려주지 않는다.** 정답 라벨이 없으므로
사람이 봐야 한다. 50장 × 5축 = 250번 판정인데, CSV를 열어 놓고 사진을 하나씩 찾아
보면 한 시간이 걸린다. 이 스크립트는 사진과 태그를 나란히 놓고 클릭 두 번으로
넘어가게 만들어 그 시간을 10분대로 줄인다.

산출물은 **축별 정확도**다. 이 숫자가 있어야 다음을 말할 수 있다:
  · 고정 축 어휘가 실제 사진에서 성립하는가 (계약 확정)
  · 12B를 써야 하는가 7B로 충분한가 (원가 결정)
  · 어느 축이 약한가 → 프롬프트를 고칠지, 축 값을 바꿀지, 그 축을 덜 믿을지(conf)

주의: 여기서 나오는 정확도는 **4bit 로컬 실행의 하한**이다. 프로덕션(L4 FP16 + vLLM)과
같지 않다. 리포트에 반드시 모델·양자화·해상도를 함께 적는다.
────────────────────────────────────────────────────────────────────────────

■ 실행 방법

    # 1) 채점 시트 생성 (사진은 base64로 박아 넣어 파일 하나로 자족한다)
    .venv/bin/python vlm_review.py --tags out/vlm_tags.csv --out out/vlm_review.html
    open out/vlm_review.html

    # 2) 브라우저에서 축마다 ✓/✗ 를 누른다. ✗면 올바른 값을 고른다.
    #    진행 상황은 브라우저에 자동 저장되므로 중간에 닫아도 된다.
    #    다 하면 "채점 결과 CSV 저장" 버튼 → out/vlm_grade.csv 로 저장

    # 3) 축별 정확도 집계
    .venv/bin/python vlm_review.py score --grade ~/Downloads/vlm_grade.csv --tags out/vlm_tags.csv
"""

from __future__ import annotations

import argparse
import base64
import csv
import html
import io
import json
import sys
from pathlib import Path

from runners.common import load_image
from vlm_tag import AXES

THUMB_LONG_EDGE = 640


def thumb_data_uri(path: str) -> str:
    img = load_image(path, long_edge=THUMB_LONG_EDGE)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def build_html(rows: list[dict], out: Path) -> None:
    cards = []
    for i, r in enumerate(rows):
        try:
            uri = thumb_data_uri(r["path"])
        except Exception as e:  # 깨진 파일은 자리만 남기고 계속
            print(f"  썸네일 실패 {r['photo_id']}: {e}")
            uri = ""
        axis_rows = "".join(
            f"""<div class="ax" data-pid="{html.escape(r['photo_id'])}" data-axis="{ax}">
                 <span class="axname">{ax}</span>
                 <span class="pred">{html.escape(r.get(ax, '') or '—')}</span>
                 <button class="ok"  type="button">✓</button>
                 <button class="ng"  type="button">✗</button>
                 <select class="fix" hidden>
                   <option value="">올바른 값…</option>
                   {''.join(f'<option value="{v}">{v}</option>' for v in vals)}
                 </select>
               </div>"""
            for ax, vals in AXES.items()
        )
        cards.append(f"""<section class="card" id="c{i}">
  <div class="imgwrap"><img loading="lazy" src="{uri}" alt=""></div>
  <div class="meta">
    <div class="pid">{i + 1}/{len(rows)} · {html.escape(r['photo_id'])}</div>
    <div class="cap">{html.escape(r.get('caption', ''))}</div>
    {axis_rows}
    <div class="capjudge" data-pid="{html.escape(r['photo_id'])}">
      캡션: <button class="ok" type="button">쓸 만함</button>
            <button class="ng" type="button">못 씀</button>
    </div>
  </div>
</section>""")

    axes_json = json.dumps(list(AXES.keys()), ensure_ascii=False)
    doc = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8">
<title>VLM 태그 채점 — {len(rows)}장</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font: 15px/1.6 -apple-system, sans-serif; margin: 0; padding: 0 0 120px; }}
 header {{ position: sticky; top: 0; background: Canvas; border-bottom: 1px solid #8884;
          padding: 12px 20px; z-index: 10; }}
 h1 {{ font-size: 17px; margin: 0 0 6px; }}
 #stats {{ font-size: 13px; opacity: .85; font-variant-numeric: tabular-nums; }}
 #stats b {{ font-weight: 600; }}
 .card {{ display: grid; grid-template-columns: minmax(280px, 42%) 1fr; gap: 20px;
         padding: 20px; border-bottom: 1px solid #8883; align-items: start; }}
 .imgwrap img {{ width: 100%; border-radius: 8px; display: block; }}
 .pid {{ font-size: 12px; opacity: .6; word-break: break-all; margin-bottom: 6px; }}
 .cap {{ margin-bottom: 14px; padding: 8px 10px; background: #8881; border-radius: 6px; }}
 .ax, .capjudge {{ display: flex; align-items: center; gap: 8px; margin-bottom: 7px; }}
 .axname {{ width: 92px; font-size: 13px; opacity: .7; }}
 .pred {{ min-width: 96px; font-weight: 600; }}
 button {{ font: inherit; padding: 3px 11px; border-radius: 6px; border: 1px solid #8886;
          background: transparent; cursor: pointer; }}
 button.on.ok {{ background: #2e7d32; color: #fff; border-color: #2e7d32; }}
 button.on.ng {{ background: #c62828; color: #fff; border-color: #c62828; }}
 select {{ font: inherit; padding: 2px 6px; }}
 footer {{ position: fixed; bottom: 0; left: 0; right: 0; background: Canvas;
          border-top: 1px solid #8884; padding: 10px 20px; display: flex; gap: 10px; }}
 @media (max-width: 720px) {{ .card {{ grid-template-columns: 1fr; }} }}
</style></head><body>
<header>
  <h1>VLM 태그 채점 — 사진 {len(rows)}장 × 축 5개</h1>
  <div id="stats">아직 채점 없음</div>
</header>
{''.join(cards)}
<footer>
  <button id="save" type="button">채점 결과 CSV 저장</button>
  <button id="reset" type="button">전부 지우기</button>
  <span id="hint" style="font-size:13px;opacity:.7;align-self:center"></span>
</footer>
<script>
const AXES = {axes_json};
const KEY = "vlm_grade_v1";
let G = {{}};
try {{ G = JSON.parse(localStorage.getItem(KEY) || "{{}}"); }} catch (e) {{ G = {{}}; }}

function save() {{ try {{ localStorage.setItem(KEY, JSON.stringify(G)); }} catch (e) {{}} }}

function stats() {{
  const per = {{}};
  AXES.concat(["caption"]).forEach(a => per[a] = {{ok: 0, ng: 0}});
  Object.values(G).forEach(byAxis => Object.entries(byAxis).forEach(([a, v]) => {{
    if (per[a] && v.verdict) per[a][v.verdict]++;
  }}));
  const parts = AXES.concat(["caption"]).map(a => {{
    const n = per[a].ok + per[a].ng;
    return n ? `${{a}} <b>${{(100 * per[a].ok / n).toFixed(0)}}%</b> (${{per[a].ok}}/${{n}})`
             : `${{a}} —`;
  }});
  const total = Object.values(per).reduce((s, p) => s + p.ok + p.ng, 0);
  document.getElementById("stats").innerHTML =
    `채점 ${{total}} / ${{(AXES.length + 1) * {len(rows)}}} · ` + parts.join(" · ");
}}

function paint(el, pid, axis) {{
  const v = (G[pid] || {{}})[axis];
  el.querySelectorAll("button").forEach(b => b.classList.toggle("on", !!v && b.classList.contains(v.verdict)));
  const fix = el.querySelector(".fix");
  if (fix) {{
    fix.hidden = !(v && v.verdict === "ng");
    if (v && v.correct) fix.value = v.correct;
  }}
}}

document.querySelectorAll(".ax, .capjudge").forEach(el => {{
  const pid = el.dataset.pid, axis = el.dataset.axis || "caption";
  el.querySelectorAll("button").forEach(btn => btn.addEventListener("click", () => {{
    const verdict = btn.classList.contains("ok") ? "ok" : "ng";
    G[pid] = G[pid] || {{}};
    G[pid][axis] = Object.assign(G[pid][axis] || {{}}, {{verdict}});
    save(); paint(el, pid, axis); stats();
  }}));
  const fix = el.querySelector(".fix");
  if (fix) fix.addEventListener("change", () => {{
    G[pid] = G[pid] || {{}};
    G[pid][axis] = Object.assign(G[pid][axis] || {{}}, {{correct: fix.value}});
    save(); stats();
  }});
  paint(el, pid, axis);
}});
stats();

document.getElementById("save").addEventListener("click", () => {{
  const lines = ["photo_id,axis,verdict,correct"];
  Object.entries(G).forEach(([pid, byAxis]) => Object.entries(byAxis).forEach(([a, v]) => {{
    if (v.verdict) lines.push(`"${{pid}}",${{a}},${{v.verdict}},${{v.correct || ""}}`);
  }}));
  const url = URL.createObjectURL(new Blob([lines.join("\\n")], {{type: "text/csv"}}));
  const a = document.createElement("a");
  a.href = url; a.download = "vlm_grade.csv"; a.click();
  URL.revokeObjectURL(url);
  document.getElementById("hint").textContent = "저장됨 — vlm_review.py score 로 집계하세요";
}});

document.getElementById("reset").addEventListener("click", () => {{
  if (!confirm("채점을 전부 지웁니다.")) return;
  G = {{}}; save(); location.reload();
}});
</script></body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    size_mb = out.stat().st_size / 1e6
    print(f"채점 시트 → {out}  ({size_mb:.1f}MB, 사진 {len(rows)}장)")
    print(f"  open {out}")


def score(grade_path: Path, tags_path: Path) -> None:
    """채점 CSV → 축별 정확도. 어느 축이 약한지, 무엇으로 틀리는지까지 본다."""
    with grade_path.open(newline="") as f:
        grades = list(csv.DictReader(f))
    with tags_path.open(newline="") as f:
        tags = {r["photo_id"]: r for r in csv.DictReader(f)}

    n_photos = len({g["photo_id"] for g in grades})
    print(f"채점 {len(grades)}건 · 사진 {n_photos}장 · 태그 원본 {tags_path}")
    print(f"\n{'축':<12}{'정확도':>10}{'맞음':>7}{'틀림':>7}")
    print("-" * 36)

    confusion: dict[str, list[str]] = {}
    for axis in [*AXES.keys(), "caption"]:
        sel = [g for g in grades if g["axis"] == axis]
        if not sel:
            continue
        ok = sum(1 for g in sel if g["verdict"] == "ok")
        print(f"{axis:<12}{100 * ok / len(sel):>9.0f}%{ok:>7}{len(sel) - ok:>7}")
        for g in sel:
            if g["verdict"] == "ng" and g.get("correct"):
                pred = tags.get(g["photo_id"], {}).get(axis, "?")
                confusion.setdefault(axis, []).append(f"{pred}→{g['correct']}")

    if confusion:
        print("\n[오답 패턴] 같은 방향으로 반복되면 프롬프트나 축 정의를 고칠 신호다")
        for axis, pairs in confusion.items():
            counts: dict[str, int] = {}
            for p in pairs:
                counts[p] = counts.get(p, 0) + 1
            top = sorted(counts.items(), key=lambda kv: -kv[1])[:6]
            print(f"  {axis:<11} " + "  ".join(f"{p}×{c}" for p, c in top))

    print("\n※ 이 숫자는 로컬 4bit 실행의 하한이다. 리포트에 모델·양자화·해상도를 함께 적을 것.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--tags", type=Path, default=Path("out/vlm_tags.csv"))
    ap.add_argument("--out", type=Path, default=Path("out/vlm_review.html"))
    s = sub.add_parser("score", help="채점 CSV → 축별 정확도")
    s.add_argument("--grade", type=Path, required=True)
    s.add_argument("--tags", type=Path, default=Path("out/vlm_tags.csv"))
    args = ap.parse_args()

    if not args.tags.exists():
        sys.exit(f"태그 CSV가 없다: {args.tags}\n  .venv/bin/python vlm_tag.py --n 50")

    if args.cmd == "score":
        score(args.grade, args.tags)
        return

    with args.tags.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if not r.get("error")]
    build_html(rows, args.out)


if __name__ == "__main__":
    main()
