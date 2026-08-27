"""로컬 CLI. Lambda(handler.py)·EC2 루프와 같은 job.run()을 부른다.

    python -m photoselect analyze --list
    python -m photoselect analyze --gallery "dataset1/류지혜고객님 (2)" [--limit 50] [--no-vlm] [--force]
    python -m photoselect draft   --gallery "dataset1/류지혜고객님 (2)" [--k 30] [--selection-id S]
    python -m photoselect review  --gallery "dataset1/류지혜고객님 (2)"   → out/<gallery>/review.html

실행 환경: torch·mediapipe가 있는 venv (scripts/spike/.venv 또는 requirements.txt로 새로).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from photoselect import gallery as gal
from photoselect import store as store_mod
from photoselect.config import Settings


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="photoselect")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="A 전수 분석")
    a.add_argument("--gallery")
    a.add_argument("--list", action="store_true", help="갤러리 이름과 장수 목록")
    a.add_argument("--limit", type=int, help="앞에서 N장만 (빠른 확인용)")
    a.add_argument("--no-vlm", action="store_true", help="VLM 태그 생략 (Ollama 없을 때)")
    a.add_argument("--force", action="store_true", help="이미 분석된 사진도 다시")
    a.add_argument("--all-formats", action="store_true", help="HEIC 포함 (기본은 JPG만)")

    d = sub.add_parser("draft", help="B 초안 한 라운드")
    d.add_argument("--gallery", required=True)
    d.add_argument("--k", type=int)
    d.add_argument("--selection-id", help="evidence-<id>.json 을 읽는다 (없으면 evidence.json)")
    d.add_argument("--round", type=int, help="라운드 번호 강제 (기본: 마지막+1)")
    d.add_argument("--target", type=int, help="셀렉 목표 장수 (기본 config.target_count)")
    d.add_argument("--llm", action="store_true", help="Bedrock으로 이유 문장·피드백 번역 (AWS 자격 필요, 텍스트만 전송)")

    x = sub.add_parser("reset", help="추천·evidence 초기화 (분석 결과는 유지) — 처음부터 다시")
    x.add_argument("--gallery", required=True)

    r = sub.add_parser("review", help="초안 검수 HTML 생성")
    r.add_argument("--gallery", required=True)
    r.add_argument("--round", type=int, help="기본: 마지막 라운드")
    r.add_argument("--pairs", type=int, help="온보딩 쌍 비교 개수. 기본: 1라운드만 12, 이후 0")
    r.add_argument("--seed", type=int, help="쌍 후보 시드. 기본: 라운드 번호 → 라운드마다 새 쌍, 이미 답한 쌍은 제외")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")
    settings = Settings.from_env()
    st = store_mod.LocalStore(settings.out_root)

    if args.cmd == "analyze":
        if args.list:
            for g, n in gal.list_galleries(settings.dataset_root):
                print(f"{n:>6}  {g}")
            return
        if not args.gallery:
            sys.exit("--gallery 또는 --list")
        from photoselect.analyze import job as analyze_job
        refs = gal.load_local(settings.dataset_root, args.gallery, limit=args.limit, jpg_only=not args.all_formats)
        result = analyze_job.run(st, args.gallery, refs, settings, force=args.force, use_vlm=not args.no_vlm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "draft":
        from photoselect.draft import job as draft_job
        llm = None
        if args.llm:
            from photoselect.llm.client import BedrockClient
            llm = BedrockClient(settings.llm.aws_region, settings.llm.model_id)
        result = draft_job.run(st, args.gallery, settings, selection_id=args.selection_id,
                               round_no=args.round, top_k=args.k, target=args.target, llm=llm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "reset":
        d = Path(settings.out_root) / args.gallery.replace("/", "__")
        for name in ("recommendations.jsonl", "evidence.json"):
            f = d / name
            if f.exists():
                f.unlink()
                print(f"삭제: {f}")
        for f in d.glob("review-r*.html"):
            f.unlink()
        print("→ 분석 결과는 유지. draft부터 다시 (브라우저의 '전부 지우기'도 눌러야 localStorage가 비워진다)")
        return

    if args.cmd == "review":
        from photoselect.review import html as review_html
        recs = st.read_recommendations(args.gallery)
        if not recs:
            sys.exit("추천이 없다 — 먼저 draft를 돌릴 것")
        rnd = args.round or max(r.round for r in recs)
        recs = sorted([r for r in recs if r.round == rnd], key=lambda r: r.rank)
        rows = st.read_analysis(args.gallery)
        analysis = {r.photo_id: r for r in rows}
        paths = {ref.photo_id: ref.path for ref in gal.load_local(settings.dataset_root, args.gallery, jpg_only=False)}
        answered = {frozenset(p[:2]) for p in st.read_evidence(args.gallery).pairs}
        n_pairs = args.pairs if args.pairs is not None else (12 if rnd == 1 else 0)   # 온보딩은 첫 라운드 한 번
        pairs = review_html.onboarding_pairs(rows, n=n_pairs,
                                             seed=args.seed if args.seed is not None else rnd,
                                             exclude=answered)
        ev = st.read_evidence(args.gallery)
        summary = {"round": rnd, "k": len(recs), "photos": len(rows),
                   "selected": f"{len(ev.selected)}/{settings.score.target_count}",
                   "scenes": {s: sum(1 for r in recs if analysis[r.photo_id].scene == s)
                              for s in sorted({analysis[r.photo_id].scene for r in recs})}}
        out = review_html.render(args.gallery, recs, analysis, paths, pairs,
                                 Path(settings.out_root) / args.gallery.replace("/", "__") / f"review-r{rnd}.html", summary)
        print(f"→ {out}\n   open '{out}'   (채점 후 '내보내기' → 그 파일을 같은 폴더에 evidence.json 으로)")
        return


if __name__ == "__main__":
    main()
