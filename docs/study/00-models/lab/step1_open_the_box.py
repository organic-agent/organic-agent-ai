"""STEP 1 — 상자를 열어 본다: 모델이란 결국 무엇인가.

────────────────────────────────────────────────────────────────────────────
실행

    cd study/00-models/lab
    ../../../photoselect/scripts/spike/.venv/bin/python step1_open_the_box.py

첫 실행은 가중치를 받느라 몇 분 걸릴 수 있다(ARNIQA는 torch.hub에서 받는다).
두 번째부터는 캐시라 빠르다. 인터넷이 필요하다.

────────────────────────────────────────────────────────────────────────────
이 스텝의 목적

"모델"이라는 말이 추상적으로 들리는 동안에는 아무것도 배울 수 없다.
그래서 첫 스텝은 이론이 하나도 없다. **만져 본다.**

    ① 모델은 파일이다        — 가중치 파일이 몇 MB인지 눈으로 본다
    ② 모델은 함수다          — 사진 하나 넣으면 숫자 몇 개가 나온다
    ③ 모델마다 뱉는 게 다르다  — 세 모델의 출력을 나란히 놓는다
    ④ 느리다                — 사진 한 장에 몇 초인지 재 본다 (원가로 이어진다)

우리 서비스가 A 배치(`plan.md` §3-A)에서 실제로 부르는 세 모델이다.
학습용 흉내가 아니라 `photoselect/scripts/spike/runners/`의 진짜 코드다.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import time

import harness as H

RUNNERS = [
    ("faces", "MediaPipe Face Landmarker", "얼굴 몇 개 · 눈 떴나 · 웃나 · 얼굴 크기"),
    ("laion", "LAION Aesthetic v2", "이 사진이 '보기 좋은가' 점수 하나"),
    ("arniqa", "ARNIQA", "이 사진이 '잘 찍혔는가' 점수 하나"),
]


def main() -> None:
    # ── ① 모델은 파일이다 ───────────────────────────────────────────────
    print("=" * 78)
    print("[1] 모델은 파일이다 — weights/ 안에 있는 것들")
    print("=" * 78)
    files = H.weight_files()
    if not files:
        print("  (아직 받은 가중치가 없다. 아래 실행에서 자동으로 받는다)")
    for name, mb in files:
        print(f"  {name:<44}{mb:>9.1f} MB")
    print("  → 모델 = **숫자가 잔뜩 든 파일 하나**다. 그 이상도 이하도 아니다.")
    print("     학습이란 이 파일 안의 숫자를 정하는 일이고, 추론이란 그 숫자로 계산하는 일이다.")
    print("  → 크기가 곧 원가다. 이 파일을 GPU 메모리에 올려야 하고(05 영역),")
    print("     서비스 이미지에 번들해야 한다(tech-stack.md).")

    # ── ② 모델을 로딩한다 ──────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[2] 모델을 메모리에 올린다 — 몇 초 걸리나")
    print("=" * 78)
    loaded = {}
    for key, title, _ in RUNNERS:
        print(f"  {title} 로딩 중 ...", end="", flush=True)
        try:
            runner, sec = H.load_runner(key)
        except Exception as exc:                     # noqa: BLE001
            print(f"  실패: {type(exc).__name__} — {exc}")
            continue
        loaded[key] = runner
        print(f"  {sec:.1f}초")
    print("  → **로딩은 사진 한 장 처리보다 훨씬 비싸다.** 그래서 배치로 돈다 —")
    print("     한 번 올려놓고 갤러리 전체를 흘려보낸다(A 배치가 갤러리당 1회인 이유).")
    print("     Lambda처럼 매번 새로 뜨는 환경에 큰 모델을 올리면 이 시간이 매번 든다.")

    if not loaded:
        raise SystemExit("\n러너를 하나도 못 올렸다. 위 오류를 먼저 해결할 것.")

    # ── ③ 사진을 넣어 본다 ─────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[3] 사진을 넣으면 숫자가 나온다 — 그게 전부다")
    print("=" * 78)
    photos = H.sample_photos(4, seed=0, with_faces=True)
    for path in photos:
        short = "/".join(path.split("/")[-2:])
        print(f"\n  {short}")
        for key, title, _ in RUNNERS:
            if key not in loaded:
                continue
            t0 = time.time()
            out = loaded[key].score(path)
            sec = time.time() - t0
            vals = "  ".join(f"{k}={v:.3f}" for k, v in out.items())
            print(f"    {title:<28}{vals}   ({sec:.2f}초)")

    print("\n  → 세 모델이 **같은 사진을 보고 다른 것을 말한다.** 이게 핵심이다:")
    print("       · MediaPipe  — 사진 안의 **무엇이 어디 있나**(얼굴·눈·입) → 구조 정보")
    print("       · LAION      — 사람이 **좋아할 만한가** → 취향의 평균")
    print("       · ARNIQA     — **잘 찍혔나**(흐림·노이즈·노출) → 기술 품질")
    print("     세 개를 다 쓰는 이유는 하나로는 답이 안 나오기 때문이다.")
    print("     흐릿하지만 예쁜 사진, 선명하지만 눈 감은 사진이 각각 존재한다.")

    # ── ④ 속도 = 원가 ─────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("[4] 속도가 곧 원가다")
    print("=" * 78)
    print(f"  {'모델':<28}{'사진당 평균':>12}{'1,000장 환산':>14}")
    for key, title, _ in RUNNERS:
        if key not in loaded:
            continue
        t0 = time.time()
        for path in photos:
            loaded[key].score(path)
        per = (time.time() - t0) / len(photos)
        print(f"  {title:<28}{per:>11.3f}초{per * 1000 / 60:>13.1f}분")
    print("  → 이 숫자가 `plan.md`의 '갤러리당 10분/1,000장' 예산과 직접 비교된다.")
    print("     스파이크 리포트의 1차 실측(M4 Pro CPU)과 같은 방식으로 잰 값이다.")
    print("  ※ 지금은 CPU 한 장씩이다. 실제로는 GPU + 병렬이라 훨씬 빠르다 —")
    print("     '얼마나 빨라지나'가 05 영역(서빙·GPU)의 주제다.")

    print("\n" + "=" * 78)
    print("여기까지가 '모델'의 전부다: 파일 → 메모리에 올림 → 입력 넣으면 출력.")
    print("남은 질문은 **그 출력이 무엇에 반응하는가**이고, step2가 그걸 흔들어 본다.")
    print("=" * 78)


if __name__ == "__main__":
    main()
