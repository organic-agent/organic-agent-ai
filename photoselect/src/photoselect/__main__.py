"""로컬 CLI. Lambda(handler.py)·EC2 루프와 같은 job.run()을 부른다.

파이프라인은 `--pipeline v1|v2|v3`(기본 v2, 환경변수 PHOTOSELECT_PIPELINE). v1 은 VLM·얼굴·BT 원본,
v2 는 docs/plan-v2-slim.md 의 슬림 파이프라인, v3 은 폴더화(V45: 임베딩 그룹 + naming, 추천 없음).
버전은 `photoselect/v1`·`v2`·`v3` 에 따로 산다.

v3 (폴더화 테스트):
    python -m photoselect --pipeline v3 analyze --db --gallery 12 [--llm]     # FULL (--llm 이면 naming까지)
    python -m photoselect --pipeline v3 naming  --db --gallery 12 --job-id J  # naming만 다시
    python -m photoselect --pipeline v3 worker --llm                          # 웹 버튼(FULL·NAMING 잡) 처리
    python -m photoselect --pipeline v3 compare --db --selection-id S --a A --b B   # 비교샷 (동기, torch 미사용)

    python -m photoselect analyze --list
    python -m photoselect analyze --gallery "dataset1/류지혜고객님 (2)" [--limit 50] [--no-vlm] [--force]
    python -m photoselect draft   --gallery "dataset1/류지혜고객님 (2)" [--k 30] [--selection-id S]
    python scripts/review.py      --gallery "dataset1/류지혜고객님 (2)"   → out/<gallery>/review-rN.html (개발 도구)

DB 모드(--db): 갤러리는 photos.gallery_id 숫자, 추천은 --selection-id(photo_selections.id) 필수.
    python -m photoselect analyze --db --gallery 12 [--job-id J] [--no-vlm] [--force]
    python -m photoselect draft   --db --gallery 12 --selection-id 3 [--job-id J] [--llm]
    python -m photoselect worker  [--no-vlm] [--llm] [--poll 2] [--once]
        → ai_analysis_jobs·ai_selection_jobs 의 PENDING 을 집어 A·B 를 돌린다. 웹 버튼이 만든 잡을 여기가 처리한다.
접속은 DB_HOST/DB_PORT/DB_NAME/DB_USER/DB_PASSWORD/DB_SSLMODE, 미리보기는 S3_BUCKET (wes scripts/local-ai.sh 참조).

실행 환경: torch·mediapipe가 있는 venv (scripts/spike/.venv 또는 requirements.txt로 새로).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path

from photoselect import worker
from photoselect.config import Settings as BaseSettings


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="photoselect")
    ap.add_argument("--pipeline", choices=("v1", "v2", "v3"),
                    help="파이프라인 (기본: 환경변수 PHOTOSELECT_PIPELINE, 없으면 v2). "
                         "v1 = VLM·얼굴·BT 원본, v3 = 폴더화(V45: 임베딩 그룹 + naming)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="A 전수 분석")
    a.add_argument("--gallery")
    a.add_argument("--db", action="store_true", help="wes DB를 읽고 쓴다 (--gallery 는 gallery_id 숫자)")
    a.add_argument("--job-id", type=int, help="ai_analysis_jobs.id — 있으면 RUNNING→DONE/FAILED 를 여기서 기록")
    a.add_argument("--list", action="store_true", help="갤러리 이름과 장수 목록")
    a.add_argument("--limit", type=int, help="앞에서 N장만 (빠른 확인용)")
    a.add_argument("--no-vlm", action="store_true", help="VLM 태그 생략 (Ollama 없을 때)")
    a.add_argument("--force", action="store_true", help="이미 분석된 사진도 다시")
    a.add_argument("--all-formats", action="store_true", help="HEIC 포함 (기본은 JPG만)")
    a.add_argument("--llm", action="store_true",
                   help="v3 전용: 분석 뒤 naming(Bedrock)까지 이어 돈다. --job-id 가 있는 v3 잡은 필수")

    nm = sub.add_parser("naming", help="v3 naming — 임베딩 그룹에 (큰 분류, 컨셉) 이름·배정 (Bedrock 필요)")
    nm.add_argument("--gallery", required=True, help="로컬은 갤러리 이름, --db 면 gallery_id 숫자")
    nm.add_argument("--db", action="store_true", help="wes DB를 읽고 쓴다 (ai_concept_assignments 는 --job-id 필수)")
    nm.add_argument("--job-id", type=int, help="ai_analysis_jobs.id (mode=NAMING) — 상태 전이와 배정의 FK")

    d = sub.add_parser("draft", help="B 초안 한 라운드")
    d.add_argument("--gallery", help="로컬은 필수. --db 면 --selection-id 로 찾을 수 있어 생략 가능")
    d.add_argument("--db", action="store_true", help="wes DB를 읽고 쓴다 (--selection-id 필수)")
    d.add_argument("--job-id", type=int, help="ai_selection_jobs.id — 있으면 상태·round 를 여기서 기록")
    d.add_argument("--k", type=int)
    d.add_argument("--selection-id", help="evidence-<id>.json 을 읽는다 (없으면 evidence.json)")
    d.add_argument("--round", type=int, help="라운드 번호 강제 (기본: 마지막+1)")
    d.add_argument("--target", type=int, help="셀렉 목표 장수 (기본 config.target_count)")
    d.add_argument("--llm", action="store_true", help="Bedrock으로 이유 문장·피드백 번역 (AWS 자격 필요. v2는 사진도 보낸다 — 크로스 리전 프로필이라 국외로 나간다)")

    cp = sub.add_parser("compare", help="v3 비교샷 — 두 사진 중 AI 판정 + 이유 (동기, torch 미사용)")
    cp.add_argument("--gallery", help="로컬 모드 갤러리 이름. --db 면 셀렉에서 찾으므로 생략 가능")
    cp.add_argument("--db", action="store_true", help="wes DB를 읽고 쓴다 (--selection-id 필수)")
    cp.add_argument("--selection-id", help="photo_selections.id — 캐시·담김 판단의 단위")
    cp.add_argument("--a", required=True, help="사진 a (로컬은 파일 상대경로, --db 는 photos.id)")
    cp.add_argument("--b", required=True, help="사진 b")
    cp.add_argument("--no-llm", action="store_true", help="템플릿 판정만 (Bedrock 없이)")

    x = sub.add_parser("reset", help="추천·evidence 초기화 (분석 결과는 유지) — 처음부터 다시")
    x.add_argument("--gallery", required=True)

    w = sub.add_parser("worker", help="잡 폴링 워커 — PENDING 잡을 집어 A·B 실행 (--db 고정)")
    w.add_argument("--no-vlm", action="store_true", help="분석에서 VLM 태그 생략 (Ollama 없을 때)")
    w.add_argument("--llm", action="store_true", help="추천 이유 문장·피드백 번역을 Bedrock 으로")
    w.add_argument("--poll", type=float, default=2.0, help="빈 큐일 때 대기 초 (기본 2)")
    w.add_argument("--once", action="store_true", help="쌓인 잡만 처리하고 종료")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(name)s | %(message)s")
    base = BaseSettings.from_env()
    if args.pipeline:
        base = replace(base, pipeline=args.pipeline)
    pipe = worker.pipeline_of(base)
    settings = pipe.settings_from(base)
    gal, store_mod = pipe.gallery, pipe.store
    use_db = getattr(args, "db", False)
    st = store_mod.LocalStore(settings.out_root, dataset_root=settings.dataset_root)

    if args.cmd == "analyze":
        if args.list:
            for g, n in gal.list_galleries(settings.dataset_root):
                print(f"{n:>6}  {g}")
            return
        if not args.gallery:
            sys.exit("--gallery 또는 --list")
        analyze_job = pipe.analyze_module()
        if use_db:
            _run_db_analyze(args, settings, pipe)
            return
        refs = gal.load_local(settings.dataset_root, args.gallery, limit=args.limit, jpg_only=not args.all_formats)
        result = analyze_job.run(st, args.gallery, refs, settings, force=args.force, use_vlm=not args.no_vlm)
        if args.llm and settings.pipeline == "v3":
            result["naming"] = pipe.naming_module().run(st, args.gallery, settings,
                                                        pipe.bedrock_client(settings), job_id=None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "naming":
        if settings.pipeline != "v3":
            sys.exit("naming 은 v3 전용이다 — --pipeline v3 (또는 PHOTOSELECT_PIPELINE=v3)")
        llm = pipe.bedrock_client(settings)
        if args.db:
            from photoselect import jobs
            dbst = store_mod.DbStore(settings)
            if args.job_id is not None and not jobs.claim(dbst.conn, jobs.ANALYSIS, args.job_id):
                sys.exit(f"잡 {args.job_id} 은 PENDING 이 아니다 (없거나 다른 워커가 집었다)")
            result = worker.run_analysis_job_v3(dbst, args.job_id, int(args.gallery), "NAMING",
                                                settings, llm=llm)
        else:
            result = pipe.naming_module().run(st, args.gallery, settings, llm, job_id=None)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "draft":
        draft_job = pipe.draft_module()
        llm = pipe.bedrock_client(settings) if args.llm else None
        if use_db:
            _run_db_draft(args, settings, pipe, llm)
            return
        if not args.gallery:
            sys.exit("--gallery 가 필요하다")
        result = draft_job.run(st, args.gallery, settings, selection_id=args.selection_id,
                               round_no=args.round, top_k=args.k, target=args.target, llm=llm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "compare":
        if settings.pipeline != "v3":
            sys.exit("compare 는 v3 전용이다 — --pipeline v3 (또는 PHOTOSELECT_PIPELINE=v3)")
        compare_job = pipe.compare_module()
        llm = None if args.no_llm else pipe.compare_client(settings)
        if args.db:
            if not args.selection_id:
                sys.exit("--db 비교샷은 --selection-id (photo_selections.id) 가 필요하다")
            dbst = store_mod.DbStore(settings, selection_id=args.selection_id)
            gallery = args.gallery or dbst.gallery_of_selection(args.selection_id)
            result = compare_job.run(dbst, gallery, settings, args.a, args.b,
                                     selection_id=args.selection_id, llm=llm)
        else:
            if not args.gallery:
                sys.exit("--gallery 가 필요하다")
            result = compare_job.run(st, args.gallery, settings, args.a, args.b, llm=llm)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if args.cmd == "worker":
        llm = pipe.bedrock_client(settings) if args.llm else None
        worker.loop(settings, use_vlm=not args.no_vlm, llm=llm, poll_seconds=args.poll, once=args.once)
        return

    if args.cmd == "reset":
        d = st._dir(args.gallery)   # 버전별 폴더 (v1: out/<g>, v2: out/v2/<g>)
        for name in ("recommendations.jsonl", "evidence.json"):
            f = d / name
            if f.exists():
                f.unlink()
                print(f"삭제: {f}")
        for f in d.glob("review-r*.html"):
            f.unlink()
        print("→ 분석 결과는 유지. draft부터 다시 (브라우저의 '전부 지우기'도 눌러야 localStorage가 비워진다)")
        return



def _run_db_analyze(args, settings, pipe) -> None:
    """DB 모드 A. 잡 id가 있으면 그 행의 상태를 여기서 옮긴다 — 워커(worker.py)와 같은 함수."""
    from photoselect import jobs

    st = pipe.store.DbStore(settings)
    if args.job_id is not None and not jobs.claim(st.conn, jobs.ANALYSIS, args.job_id):
        sys.exit(f"잡 {args.job_id} 은 PENDING 이 아니다 (없거나 다른 워커가 집었다)")
    if settings.pipeline == "v3":
        llm = pipe.bedrock_client(settings) if args.llm else None
        result = worker.run_analysis_job_v3(st, args.job_id, int(args.gallery), "FULL", settings,
                                            llm=llm, force=args.force, limit=args.limit)
    else:
        result = worker.run_analysis_job(st, args.job_id, int(args.gallery), settings,
                                         force=args.force, use_vlm=not args.no_vlm, limit=args.limit)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _run_db_draft(args, settings, pipe, llm) -> None:
    """DB 모드 B. 워커와 같은 함수 — 갤러리는 셀렉에서 찾고, 라운드 번호는 잡 행에도 적는다."""
    from photoselect import jobs

    if not args.selection_id:
        sys.exit("--db 추천은 --selection-id (photo_selections.id) 가 필요하다")
    st = pipe.store.DbStore(settings, selection_id=args.selection_id)
    if args.job_id is not None and not jobs.claim(st.conn, jobs.SELECTION, args.job_id):
        sys.exit(f"잡 {args.job_id} 은 PENDING 이 아니다 (없거나 다른 워커가 집었다)")
    result = worker.run_selection_job(st, args.job_id, args.selection_id, settings, llm=llm,
                                      gallery=args.gallery, round_no=args.round, top_k=args.k, target=args.target)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
