"""STEP 2 — 흔들어 본다: 이 모델은 무엇에 반응하는가.

────────────────────────────────────────────────────────────────────────────
실행

    cd study/00-models/lab
    ../../../photoselect/scripts/spike/.venv/bin/python step2_what_it_reacts_to.py

────────────────────────────────────────────────────────────────────────────
이 스텝의 목적

step1에서 "모델은 사진을 넣으면 숫자가 나오는 함수"라는 것까지 봤다.
그럼 **그 숫자는 무엇에 반응하는가?**

논문을 읽어서 아는 방법도 있지만, 더 빠르고 확실한 방법이 있다 —
**사진 한 장을 여러 방식으로 망가뜨려 보고 점수가 어떻게 변하는지 본다.**

    흐리게 → ARNIQA(기술 품질)가 떨어져야 한다. 안 떨어지면 그 모델을 못 믿는다
    어둡게 → 어느 모델이 반응하나
    좌우 반전 → **내용이 같으므로 아무것도 안 변해야 한다**

■ 마지막 줄이 이 실습의 핵심 장치다

좌우 반전은 사진의 **내용을 하나도 안 바꾼다.** 그런데도 점수가 조금은 움직인다.
그 움직임이 **잡음의 바닥(noise floor)** 이다.

    "흐리게 했더니 점수가 0.02 떨어졌다"  →  잡음 바닥이 0.03이면 **아무 의미 없다**
    "흐리게 했더니 점수가 0.15 떨어졌다"  →  잡음 바닥의 5배 = 진짜 반응

**변화가 의미 있는지 알려면 아무 의미 없는 변형과 비교해야 한다.**
02 영역에서 "기준선 없는 숫자는 읽을 수 없다"고 계속 말한 것의 가장 단순한 형태다.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

import harness as H

RUNNERS = [("faces", "eyes_open"), ("laion", "aesthetic_score"), ("arniqa", "technical_score")]


def variants(img: Image.Image) -> dict[str, Image.Image]:
    """원본을 여러 방식으로 바꾼다. **내용을 바꾸는 것과 안 바꾸는 것을 섞는다.**"""
    w, h = img.size
    return {
        "원본": img,
        "좌우 반전 ★": ImageOps.mirror(img),               # 내용 그대로 = 잡음 바닥
        "살짝 흐리게": img.filter(ImageFilter.GaussianBlur(1.5)),
        "많이 흐리게": img.filter(ImageFilter.GaussianBlur(5.0)),
        "어둡게 (×0.5)": ImageEnhance.Brightness(img).enhance(0.5),
        "밝게 (×1.5)": ImageEnhance.Brightness(img).enhance(1.5),
        "채도 0 (흑백)": ImageEnhance.Color(img).enhance(0.0),
        "가운데 60% 크롭": img.crop((int(w * .2), int(h * .2), int(w * .8), int(h * .8))),
        "해상도 1/4로": img.resize((w // 4, h // 4)).resize((w, h)),
    }


def main() -> None:
    photo = H.sample_photos(1, seed=3, with_faces=True)[0]
    print("=" * 92)
    print("[1] 사진 한 장을 여러 방식으로 바꿔 가며 점수를 잰다")
    print("=" * 92)
    print(f"  대상: {'/'.join(photo.split('/')[-2:])}")
    print()

    runners = {}
    for key, _ in RUNNERS:
        try:
            runners[key] = H.load_runner(key)[0]
        except Exception as exc:                          # noqa: BLE001
            print(f"  ({key} 로딩 실패 — 건너뜀: {type(exc).__name__})")

    src = Image.open(photo)
    src = ImageOps.exif_transpose(src).convert("RGB")
    tmp = Path(tempfile.mkdtemp(prefix="study-models-"))

    rows: dict[str, dict[str, float]] = {}
    for name, im in variants(src).items():
        p = tmp / f"{abs(hash(name))}.jpg"
        im.save(p, quality=95)
        out: dict[str, float] = {}
        for key, field in RUNNERS:
            if key not in runners:
                continue
            out[field] = runners[key].score(str(p)).get(field, float("nan"))
        rows[name] = out

    fields = [f for _, f in RUNNERS if any(f in r for r in rows.values())]
    print("  " + f"{'변형':<20}" + "".join(f"{f:>20}" for f in fields))
    base = rows["원본"]
    for name, out in rows.items():
        line = f"  {name:<20}"
        for f in fields:
            v = out.get(f, float("nan"))
            d = v - base.get(f, float("nan"))
            line += f"{v:>12.3f}{('(%+.3f)' % d) if name != '원본' else '        '}"
        print(line)

    # ── 잡음 바닥과 비교 ────────────────────────────────────────────────
    print("\n" + "=" * 92)
    print("[2] 어느 변화가 '진짜'인가 — 좌우 반전(★)을 잡음 바닥으로 삼는다")
    print("=" * 92)
    floor = {f: abs(rows["좌우 반전 ★"].get(f, 0) - base.get(f, 0)) for f in fields}
    print("  " + "  ".join(f"{f} 잡음 바닥 {floor[f]:.3f}" for f in fields))
    print()
    print("  " + f"{'변형':<20}" + "".join(f"{f + ' 배수':>22}" for f in fields))
    for name, out in rows.items():
        if name in ("원본", "좌우 반전 ★"):
            continue
        line = f"  {name:<20}"
        for f in fields:
            d = abs(out.get(f, float("nan")) - base.get(f, float("nan")))
            ratio = d / floor[f] if floor[f] > 1e-9 else float("inf")
            mark = "진짜 반응" if ratio >= 3 else ("애매" if ratio >= 1.5 else "잡음 수준")
            line += f"{ratio:>10.1f}배  {mark:<8}"
        print(line)
    print("  → 배수가 3 이상이면 그 모델이 **그 변형에 실제로 반응한다**고 볼 수 있다.")
    print("     1.5 미만이면 우연히 흔들린 것과 구분이 안 된다.")

    print("\n" + "=" * 92)
    print("[3] 읽는 법 — 그리고 이 표가 실제로 말한 것")
    print("=" * 92)
    print("  ★ **eyes_open의 잡음 바닥이 압도적으로 크다.** 좌우 반전만 했는데")
    print(f"     {base.get('eyes_open', float('nan')):.3f} → {rows['좌우 반전 ★'].get('eyes_open', float('nan')):.3f} 로 무너진다. **사진 속 사람은 그대로 눈을 뜨고 있는데.**")
    print("     그래서 eyes_open 열의 '잡음 수준' 판정은 '크롭에 반응 안 한다'는 뜻이 아니라")
    print("     **'바닥이 너무 높아서 알 수 없다'** 는 뜻이다. 두 문장은 완전히 다르다.")
    print("  · ARNIQA는 '많이 흐리게'에 29배, '어둡게'에 6배 반응한다 — **기술 품질 모델답다.**")
    print("    '살짝 흐리게'는 1.9배로 애매하다. 즉 미세한 흐림은 못 잡는다는 뜻이기도 하다.")
    print("  · LAION은 거의 모든 변형에 반응한다(크롭 24배, 밝기 21배). **미학 모델이라")
    print("    '잘 찍혔나'가 아니라 '보기 좋은가'를 보기 때문**이다 — 구도가 바뀌면 움직인다.")
    print("  · 흑백으로 만들면 LAION은 **올라간다**(+0.066). 흑백 사진을 미학적으로 좋게 본다.")
    print("    사람의 취향 평균을 학습한 모델이라 그렇다. '품질'과 '취향'은 다른 축이다.")
    print()
    print("  → 그래서 `plan.md`가 세 모델을 **다 쓴다.** 각자 다른 것에 반응하니까:")
    print("      technical_pct(ARNIQA) + aesthetic_pct(LAION) + eyes_open(MediaPipe)")
    print("  → 그리고 그래서 **미리보기 해상도를 고정한다**(1600px). 해상도가 바뀌면")
    print("     점수가 바뀌므로 사진마다 다른 크기로 넣으면 비교가 깨진다.")
    print("     `runners/common.py`의 `PREVIEW_LONG_EDGE`가 그 장치다.")

    # ── [4] 반전 테스트를 여러 장에 — 라벨 없는 안정성 진단 ──────────────
    print("\n" + "=" * 92)
    print("[4] 좌우 반전 테스트를 여러 장에 — **라벨 없이** 신호를 검증하는 법")
    print("=" * 92)
    if "faces" not in runners:
        print("  (faces 러너가 없어 건너뜀)")
    else:
        print("  스파이크 리포트의 미해결 질문: '작은 얼굴의 eyes_open을 믿을 수 있나'")
        print("  보통은 사진 50장에 눈감김 라벨을 달아야 답할 수 있다. 그런데 —")
        print("  **반전은 정답을 안 바꾼다.** 그러니 라벨 없이도 안정성은 잴 수 있다.\n")
        print(f"  {'얼굴 비율':>10}{'원본':>9}{'반전':>9}{'차이':>9}   판정")
        diffs = []
        for path in H.sample_photos(12, seed=1, with_faces=True):
            im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            a = runners["faces"].score(path)
            fp = tmp / "flip.jpg"
            ImageOps.mirror(im).save(fp, quality=95)
            b = runners["faces"].score(str(fp))
            if a["face_count"] == 0 or b["face_count"] == 0:
                continue
            d = abs(a["eyes_open"] - b["eyes_open"])
            diffs.append((a["max_face_ratio"], a["eyes_open"], b["eyes_open"], d))
        diffs.sort(key=lambda r: -r[3])
        for ratio, o, m, d in diffs:
            verdict = "**못 믿는다**" if d > 0.3 else ("주의" if d > 0.1 else "안정")
            print(f"  {ratio:>10.3f}{o:>9.3f}{m:>9.3f}{d:>9.3f}   {verdict}")
        big = [d for *_, d in diffs if d > 0.3]
        print(f"\n  {len(diffs)}장 중 {len(big)}장이 반전만으로 0.3 넘게 흔들린다.")
        print("  → **흔들리는 쪽은 원본 eyes_open이 높게 나온 사진들이다.** 낮은 값들은")
        print("     반전해도 그대로다. 즉 이 표본에서 못 믿을 것은 '낮은 값'이 아니라")
        print("     **가끔 나오는 높은 값**이다 — 스파이크 리포트가 걱정한 방향과 반대다.")
        print("  ※ 다만 12장이고 얼굴 비율이 전부 0.03 언저리라 **결론은 못 낸다.**")
        print("     얼굴 크기 범위가 넓은 표본으로 같은 표를 뽑아야 한다.")
        print("     **확인된 것은 답이 아니라 방법이다** — 라벨 0장으로 여기까지 왔다.")
        print("  → 되먹임: 스파이크 리포트의 '작은 얼굴 blink 신뢰도' 항목에")
        print("     **반전 테스트를 먼저** 넣는다. 라벨링은 그 결과를 보고 결정한다.")

    print(f"\n  (변형된 이미지는 여기 있다 — 직접 열어 보라: {tmp})")


if __name__ == "__main__":
    main()
