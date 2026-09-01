"""고정 축 어휘 — 이 서버 전체의 **계약**이다.

`docs/feature-design.md` §고정 축 어휘와 반드시 같아야 한다. VLM 프롬프트(analyze/vlm.py),
취향 피처(draft/preference.py), 커버리지(draft/rerank.py)가 전부 여기서 읽는다.
값을 바꾸면 `MODEL_VERSION`도 올린다 — 기존 `photo_analysis` 행과 호환되지 않기 때문이다.
"""

from __future__ import annotations

AXES: dict[str, list[str]] = {
    "scene": [
        "prep", "entrance", "vow", "ring", "kiss", "family",
        "group", "bouquet", "walk", "snap", "detail", "unknown",
    ],
    "framing": ["closeup", "half", "full", "wide"],
    "lighting": ["natural", "backlit", "indoor", "flash", "lowlight"],
    "expression": ["smile", "laugh", "serious", "candid", "eyes_closed", "none"],
    "subjects": ["bride", "groom", "couple", "family", "friends", "none"],
}

#: 원핫 피처의 열 이름. 순서가 곧 취향 벡터 w의 인덱스다.
FEATURE_NAMES: list[str] = [f"{ax}={v}" for ax, vals in AXES.items() for v in vals]
FEATURE_INDEX: dict[str, int] = {n: i for i, n in enumerate(FEATURE_NAMES)}
D_TAG = len(FEATURE_NAMES)  # 33

#: 축 어휘 계약의 버전. 어휘가 바뀌면 올린다.
AXES_VERSION = "axes-v1"

#: 이유 문장용 한국어 라벨. 태그 값 → 사람이 읽는 말.
LABELS_KO: dict[str, dict[str, str]] = {
    "scene": {
        "prep": "준비", "entrance": "입장", "vow": "서약", "ring": "반지 교환", "kiss": "키스",
        "family": "가족", "group": "단체", "bouquet": "부케", "walk": "행진", "snap": "스냅",
        "detail": "디테일", "unknown": "",
    },
    "framing": {"closeup": "클로즈업", "half": "상반신", "full": "전신", "wide": "원경"},
    "lighting": {
        "natural": "자연광", "backlit": "역광", "indoor": "실내 조명",
        "flash": "플래시", "lowlight": "저조도",
    },
    "expression": {
        "smile": "미소", "laugh": "웃음", "serious": "진지한 표정", "candid": "자연스러운 순간",
        "eyes_closed": "", "none": "",
    },
    "subjects": {
        "bride": "신부", "groom": "신랑", "couple": "두 분", "family": "가족",
        "friends": "하객", "none": "",
    },
}


def onehot(tags: dict[str, str]) -> list[float]:
    """축→값 dict 하나를 33차원 0/1 벡터로. 모르는 값은 무시한다(오래된 행 호환)."""
    x = [0.0] * D_TAG
    for ax, v in tags.items():
        i = FEATURE_INDEX.get(f"{ax}={v}")
        if i is not None:
            x[i] = 1.0
    return x
