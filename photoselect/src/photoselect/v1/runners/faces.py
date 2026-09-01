"""A-1 얼굴 신호 — MediaPipe Face Landmarker (blendshapes).

산출: 얼굴 수, eyes_open(전원 눈 뜸의 최소값), smile(평균), 최대 얼굴 bbox 비율.
클러스터 대표 선정(A-7)의 재료다. **절대 판정이 아니라 연사 안의 상대 비교**에만 쓴다 —
study/00 step2가 보인 대로 이 신호는 작은 얼굴에서 크게 흔들린다(반전만으로 0.6).
"""

from __future__ import annotations

import numpy as np

from photoselect.v1.runners.common import fetch_weight, load_image

TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
VERSION = "face_landmarker-f16-v1"


class FacesRunner:
    name = "faces"
    version = VERSION

    def __init__(self) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        task_path = fetch_weight(TASK_URL, "face_landmarker_f16_v1.task")
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(task_path)),
            output_face_blendshapes=True,
            num_faces=8,
        )
        self._mp = mp
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def score(self, path: str) -> dict[str, float]:
        img = load_image(path)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.asarray(img))
        result = self._landmarker.detect(mp_image)

        n_faces = len(result.face_landmarks)
        if n_faces == 0:
            return {"face_count": 0.0, "eyes_open": float("nan"), "smile": float("nan"), "max_face_ratio": 0.0}

        eyes_open, smile = [], []
        for shapes in result.face_blendshapes:
            by_name = {s.category_name: s.score for s in shapes}
            blink = max(by_name.get("eyeBlinkLeft", 0.0), by_name.get("eyeBlinkRight", 0.0))
            eyes_open.append(1.0 - blink)
            smile.append((by_name.get("mouthSmileLeft", 0.0) + by_name.get("mouthSmileRight", 0.0)) / 2)

        max_ratio = 0.0
        for lm in result.face_landmarks:
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            max_ratio = max(max_ratio, (max(xs) - min(xs)) * (max(ys) - min(ys)))

        return {
            "face_count": float(n_faces),
            "eyes_open": float(min(eyes_open)),   # 한 명이라도 감았으면 낮다
            "smile": float(np.mean(smile)),
            "max_face_ratio": float(max_ratio),
        }
