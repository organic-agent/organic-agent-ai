"""VLM 고정 축 태그 — 로컬(Ollama) 실측.

────────────────────────────────────────────────────────────────────────────
무엇을 재는 스크립트인가

`plan.md` A-4/A-5(오픈 VLM 고정 축 태그 + 캡션)를 **맥북에서 먼저 돌려 보고**,
축 어휘 계약이 실제 사진에서 성립하는지 확인한다. 프로덕션(g6/L4 + vLLM)의
대체가 아니라 **품질 선행 검증**이다.

이 스크립트가 답할 수 있는 것
  · 고정 축 어휘가 실제 사진을 담아내는가 (unknown/none 비율)
  · 축별 태그 정확도 (사람이 눈으로 채점 — vlm_review.py로 채점 시트 생성)
  · 12B와 7B의 태그 품질 차이
  · 캡션이 이유 문장 소재로 쓸 만한가

이 스크립트가 답할 수 **없는** 것
  · 처리량 (1,000장 ≤ 6분) — vLLM 배치가 있는 L4에서만 측정 가능
  · 프로덕션 품질의 상한 — 여기는 4bit 양자화라 **하한**으로 읽어야 한다

데이터 원칙: 전부 로컬 추론이다. 이미지가 외부로 나가지 않는다 (CLAUDE.md).
────────────────────────────────────────────────────────────────────────────

■ 실행 방법

    # 1) 로컬 환경 준비 — 기동 + 모델. 명령의 정본은 이 스크립트다
    ./setup_ollama.sh                               # 처음이면 --install

    # 2) 매니페스트가 없으면 먼저
    python manifest.py build --root ../../../../dataset --out out/manifest.csv

    # 3) 태그 뽑기 (50장, 시드 고정 → 재실행하면 같은 50장)
    .venv/bin/python vlm_tag.py --manifest out/manifest.csv --n 50

    # 4) 다른 모델과 비교
    ./setup_ollama.sh --compare
    .venv/bin/python vlm_tag.py --manifest out/manifest.csv --n 50 \
        --model qwen2.5vl:7b --out out/vlm_tags_qwen7b.csv

중단해도 된다 — 같은 --out으로 다시 돌리면 이미 끝난 사진은 건너뛴다 (run.py와 같은 규약).

■ enum 강제

Ollama의 structured outputs(`format`에 JSON 스키마)를 쓴다. 프로덕션의
vLLM guided decoding과 같은 역할 — **모델이 어휘 밖의 값을 뱉을 수 없다.**
따라서 여기서 나오는 `unknown`/`none`은 파싱 실패가 아니라 **모델의 실제 선택**이다.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import manifest as manifest_mod
from runners.common import load_image

# 프롬프트·스키마의 정본은 서비스 모듈이다(2026-08-25 2차안). 여기서는 표본 뽑기·CSV 기록만 한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from photoselect.v1.analyze.vlm import SCHEMA, SYSTEM_PROMPT  # noqa: E402

# ── 고정 축 어휘 (feature-design.md §고정 축 어휘와 **반드시** 동일) ──────────
# 값을 바꾸면 계약이 깨진다. 바꿀 때는 문서·gallery.py·model_version을 함께.
AXES: dict[str, list[str]] = {
    "scene": ["prep", "entrance", "vow", "ring", "kiss", "family",
              "group", "bouquet", "walk", "snap", "detail", "unknown"],
    "framing": ["closeup", "half", "full", "wide"],
    "lighting": ["natural", "backlit", "indoor", "flash", "lowlight"],
    "expression": ["smile", "laugh", "serious", "candid", "eyes_closed", "none"],
    "subjects": ["bride", "groom", "couple", "family", "friends", "none"],
}

# 모델에게 주는 축 설명. 어휘 자체는 스키마가 강제하므로, 여기서는 **판단 기준**만 준다.
_AXIS_GUIDE_V1 = """\
scene — 결혼식 진행 순서상 어느 장면인가
  prep 준비(신부대기실·메이크업) · entrance 입장 · vow 서약/예식 · ring 반지 교환
  kiss 키스 · family 가족 사진 · group 단체 사진 · bouquet 부케
  walk 퇴장/행진 · snap 스냅/자유 컷 · detail 소품·공간·음식 등 사물 위주
  unknown 위 어느 것도 아니거나 판단 불가
framing — 인물이 프레임을 차지하는 크기
  closeup 얼굴 위주 · half 상반신 · full 전신 · wide 원경/공간이 주가 되는 컷
lighting — 지배적인 광원
  natural 자연광 · backlit 역광(인물 뒤에서 빛) · indoor 실내 조명
  flash 플래시 직광 · lowlight 어둡고 노이즈 있는 저조도
expression — 주 인물의 표정
  smile 미소 · laugh 크게 웃음 · serious 진지/무표정 · candid 의식하지 않은 자연스러움
  eyes_closed 눈 감김 · none 얼굴이 없거나 식별 불가
subjects — 화면의 주 인물 구성
  bride 신부 단독 · groom 신랑 단독 · couple 신랑신부 · family 가족
  friends 친구/하객 · none 인물 없음\
"""

_SYSTEM_PROMPT_V1 = f"""\
당신은 웨딩 사진을 분류하는 도구다. 사진 한 장을 보고 아래 5개 축을 각각 하나씩 고르고,
한국어 캡션 1문장을 쓴다.

{_AXIS_GUIDE_V1}

caption — 고객(신랑신부)에게 그대로 보여도 되는 톤의 한국어 1문장.
  예: "야외 자연광 아래 두 분이 마주 보며 웃는 컷"
  사진에 보이는 것만 쓴다. 추측·칭찬·감상은 넣지 않는다.

반드시 JSON만 출력한다. 설명 문장을 덧붙이지 않는다.\
"""

# Ollama structured outputs 스키마 = 프로덕션의 vLLM guided decoding과 같은 역할
_SCHEMA_V1 = {
    "type": "object",
    "properties": {
        **{ax: {"type": "string", "enum": vals} for ax, vals in AXES.items()},
        "caption": {"type": "string"},
    },
    "required": [*AXES.keys(), "caption"],
}

FIELDS = ["photo_id", "group", "path", *AXES.keys(), "caption", "sec", "model", "error"]


# ── 표본 뽑기 ───────────────────────────────────────────────────────────────
def sample_rows(rows: list[dict], n: int, seed: int) -> list[dict]:
    """데이터셋(최상위 폴더)별로 균등하게, 그 안에서는 그룹별로 균등하게 뽑는다.

    왜 단순 무작위가 아닌가: dataset2가 7,189장이라 무작위로 50장을 뽑으면 44장이
    거기서 나온다. 스냅(HEIC)과 본식 갤러리는 사진 성격이 달라서, 태그 품질도
    따로 봐야 한다. 시드를 고정하므로 재실행·모델 비교 시 **같은 50장**이 쓰인다.
    """
    by_top: dict[str, list[dict]] = {}
    for r in rows:
        by_top.setdefault(r["group"].split("/")[0], []).append(r)

    rng = random.Random(seed)
    picked: list[dict] = []
    tops = sorted(by_top)
    for i, top in enumerate(tops):
        # 나머지를 앞쪽 데이터셋에 몰아주지 않도록 몫을 균등 분배
        quota = n // len(tops) + (1 if i < n % len(tops) else 0)
        by_group: dict[str, list[dict]] = {}
        for r in by_top[top]:
            by_group.setdefault(r["group"], []).append(r)
        groups = sorted(by_group)
        for j, gname in enumerate(groups):
            q = quota // len(groups) + (1 if j < quota % len(groups) else 0)
            pool = by_group[gname]
            picked += rng.sample(pool, min(q, len(pool)))
    picked.sort(key=lambda r: r["photo_id"])
    return picked


# ── Ollama 호출 ─────────────────────────────────────────────────────────────
def to_jpeg_b64(path: str, long_edge: int) -> str:
    """미리보기 파생본을 흉내낸다 — EXIF 회전 보정 + 리사이즈 + JPEG.

    서비스에서도 VLM은 원본이 아니라 embedder의 preview JPEG를 읽는다(CLAUDE.md).
    HEIC 디코드는 runners.common이 pillow-heif로 처리한다.
    """
    img = load_image(path, long_edge=long_edge)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def tag_one(host: str, model: str, img_b64: str, timeout: float) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "이 사진을 분류하라.", "images": [img_b64]},
        ],
        "format": SCHEMA,     # ← enum 강제
        "stream": False,
        "options": {"temperature": 0.0, "num_predict": 256},
    }
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return json.loads(body["message"]["content"])


# ── 요약 ────────────────────────────────────────────────────────────────────
def summarize(records: list[dict], model: str) -> None:
    ok = [r for r in records if not r.get("error")]
    print(f"\n{'=' * 70}\n요약 — {model} · 성공 {len(ok)}/{len(records)}장\n{'=' * 70}")

    if not ok:
        return

    secs = sorted(float(r["sec"]) for r in ok)
    p = lambda q: secs[min(int(len(secs) * q), len(secs) - 1)]   # noqa: E731
    print(f"\n[속도] 중앙값 {statistics.median(secs):.2f}s · p90 {p(0.9):.2f}s · 합계 {sum(secs):.0f}s")
    print(f"       1,000장 환산 {statistics.median(secs) * 1000 / 60:.0f}분 "
          f"(프로덕션 목표는 L4+vLLM에서 6분 — 여기 숫자와 직접 비교 금지)")

    print("\n[축별 분포] — 한쪽 값에 쏠리면 그 축은 변별력이 없다는 신호")
    for ax, vals in AXES.items():
        counts = {v: sum(1 for r in ok if r[ax] == v) for v in vals}
        shown = " ".join(f"{v}={c}" for v, c in counts.items() if c)
        print(f"  {ax:<11} {shown}")

    # 어휘가 사진을 못 담으면 여기가 커진다 — 축 어휘 확장 판단의 근거
    unk = sum(1 for r in ok if r["scene"] == "unknown")
    non_e = sum(1 for r in ok if r["expression"] == "none")
    non_s = sum(1 for r in ok if r["subjects"] == "none")
    print(f"\n[어휘 커버리지] scene=unknown {unk}/{len(ok)} · "
          f"expression=none {non_e} · subjects=none {non_s}")

    lens = [len(r["caption"]) for r in ok]
    print(f"[캡션] 길이 중앙값 {statistics.median(lens):.0f}자 · 최단 {min(lens)} · 최장 {max(lens)}")
    print(f"  예시: {ok[0]['caption']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("out/manifest.csv"))
    ap.add_argument("--out", type=Path, default=Path("out/vlm_tags.csv"))
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0, help="표본 시드. 모델 비교 시 같게 둘 것")
    ap.add_argument("--model", default="gemma3:12b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--prompt", choices=["module", "v1"], default="module",
                    help="module=서비스 모듈의 현재 프롬프트 · v1=08-25 1차 프롬프트(재현성 측정용)")
    ap.add_argument("--long-edge", type=int, default=1024, help="VLM 입력 리사이즈 긴 변")
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    global SYSTEM_PROMPT, SCHEMA
    if args.prompt == "v1":
        SYSTEM_PROMPT, SCHEMA = _SYSTEM_PROMPT_V1, _SCHEMA_V1

    if not args.manifest.exists():
        sys.exit(f"매니페스트가 없다: {args.manifest}\n  python manifest.py build --root ../../../../dataset")

    rows = sample_rows(manifest_mod.load(args.manifest), args.n, args.seed)

    # 재개 — 이미 성공한 photo_id는 건너뛴다 (run.py와 같은 규약)
    done: dict[str, dict] = {}
    if args.out.exists():
        with args.out.open(newline="") as f:
            done = {r["photo_id"]: r for r in csv.DictReader(f) if not r.get("error")}

    todo = [r for r in rows if r["photo_id"] not in done]
    print(f"모델 {args.model} · 표본 {len(rows)}장 (완료 {len(done)} / 남은 {len(todo)})")
    if todo:
        print(f"  긴 변 {args.long_edge}px · enum 강제 ON · temperature 0")

    records = list(done.values())
    started = time.monotonic()
    for i, row in enumerate(todo, 1):
        rec = {"photo_id": row["photo_id"], "group": row["group"], "path": row["path"],
               "model": args.model, "error": ""}
        t0 = time.monotonic()
        try:
            out = tag_one(args.host, args.model, to_jpeg_b64(row["path"], args.long_edge), args.timeout)
            rec.update({ax: out.get(ax, "") for ax in AXES})
            rec["caption"] = out.get("caption", "").strip().replace("\n", " ")
        except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
            rec.update({ax: "" for ax in AXES})
            rec["caption"] = ""
        rec["sec"] = f"{time.monotonic() - t0:.2f}"
        records.append(rec)

        mark = "!" if rec["error"] else " "
        print(f"  [{i}/{len(todo)}]{mark} {rec['sec']:>6}s  "
              f"{rec.get('scene',''):<9}{rec.get('framing',''):<9}{rec.get('subjects',''):<9}"
              f"{rec['caption'][:36]}")

        # 매 장 저장 — 중단해도 진행분이 남는다
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            wr.writeheader()
            wr.writerows(sorted(records, key=lambda r: r["photo_id"]))

    if todo:
        print(f"\n총 {time.monotonic() - started:.0f}s → {args.out}")
    summarize(records, args.model)
    print(f"\n채점 시트를 만들려면: .venv/bin/python vlm_review.py --tags {args.out}")


if __name__ == "__main__":
    main()
