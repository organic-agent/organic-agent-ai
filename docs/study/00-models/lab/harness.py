"""스파이크 하네스의 러너를 학습용으로 빌려 온다.

⚠ 파일 이름이 `harness.py`인 이유: 처음에 `models.py`로 지었더니 **ARNIQA가 안 올라갔다.**
   torch.hub가 받아 둔 ARNIQA repo 안에 `models/` 패키지가 있고 `from models.resnet import ...`
   를 하는데, sys.path에 이 폴더가 먼저 있어서 우리 파일이 가로챘다.
   → 남의 코드를 import하는 폴더에서는 **흔한 모듈 이름을 쓰지 않는다**(`models`, `utils`, `config`).

────────────────────────────────────────────────────────────────────────────
왜 새로 안 짜는가

`photoselect/scripts/spike/runners/`에 **이미 돌아가는 코드가 있다.**
서비스가 실제로 쓸(쓸 예정인) 모델 세 개를 그대로 부르는 러너다.
학습용으로 흉내를 새로 만들면 "진짜는 다르겠지"라는 의심이 남는다 —
그냥 진짜를 부른다.

    mediapipe_faces.FacesRunner   → face_count, eyes_open, smile, max_face_ratio
    laion_aesthetic.LaionRunner   → aesthetic_score (+ CLIP 임베딩)
    arniqa.ArniqaRunner           → scale_score (기술 품질)

■ 실행 환경 (중요)

이 폴더의 스크립트는 **study/.venv가 아니라 스파이크 venv**로 돌린다.
torch·mediapipe·open_clip이 거기 설치돼 있기 때문이다.

    cd study/00-models/lab
    ../../../photoselect/scripts/spike/.venv/bin/python step1_open_the_box.py

편의를 위해 각 스크립트 맨 위에 그 명령을 적어 뒀다.
────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

_SPIKE = (Path(__file__).resolve().parent / ".." / ".." / ".."
          / "photoselect" / "scripts" / "spike").resolve()
if not _SPIKE.is_dir():                                  # pragma: no cover
    raise SystemExit(f"스파이크 하네스를 찾을 수 없다: {_SPIKE}")
sys.path.insert(0, str(_SPIKE))

MANIFEST = _SPIKE / "out" / "manifest.csv"
WEIGHTS = _SPIKE / "weights"


def sample_photos(n: int = 6, group: str | None = None, seed: int = 0,
                  with_faces: bool = False) -> list[str]:
    """매니페스트에서 사진 경로를 뽑는다. JPG만 — HEIC는 로딩이 느려 실습에 안 맞는다.

    with_faces=True면 이전 스모크 실행 결과(`out/smoke.csv`)에서 얼굴이 잡힌 사진만 고른다.
    웨딩 갤러리에는 디테일 컷·풍경이 많아 무작위로 뽑으면 얼굴 없는 사진만 나오기 쉽다.
    """
    if not MANIFEST.exists():
        raise SystemExit(
            f"매니페스트가 없다: {MANIFEST}\n"
            f"  cd {_SPIKE} && .venv/bin/python manifest.py build "
            f"--root ../../../../dataset --out out/manifest.csv"
        )
    import random

    face_ids: set[str] | None = None
    if with_faces:
        smoke = _SPIKE / "out" / "smoke.csv"
        if smoke.exists():
            with smoke.open(encoding="utf-8") as f:
                face_ids = {
                    r["photo_id"] for r in csv.DictReader(f)
                    if float(r.get("faces_face_count") or 0) > 0
                }

    rows = []
    with MANIFEST.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row["path"].lower().endswith((".jpg", ".jpeg")):
                continue
            if group and not row["group"].startswith(group):
                continue
            if face_ids is not None and row["photo_id"] not in face_ids:
                continue
            rows.append(row["path"])
    if not rows:
        raise SystemExit("조건에 맞는 JPG가 없다"
                         + (" (out/smoke.csv에 얼굴 있는 사진이 없다)" if with_faces else ""))
    random.Random(seed).shuffle(rows)
    return rows[:n]


def load_runner(name: str):
    """러너 하나를 만든다. 첫 호출은 가중치 로딩 때문에 몇 초 걸린다."""
    t0 = time.time()
    if name == "faces":
        from runners.mediapipe_faces import FacesRunner
        r = FacesRunner()
    elif name == "laion":
        from runners.laion_aesthetic import LaionAestheticRunner as R  # noqa: N814
        r = R()
    elif name == "arniqa":
        from runners.arniqa import ArniqaRunner
        r = ArniqaRunner()
    else:
        raise ValueError(name)
    return r, time.time() - t0


def weight_files() -> list[tuple[str, float]]:
    """가중치 파일과 크기(MB). '모델 = 파일'이라는 감각을 위해."""
    if not WEIGHTS.is_dir():
        return []
    return sorted(
        (p.name, p.stat().st_size / 1e6) for p in WEIGHTS.iterdir() if p.is_file()
    )
