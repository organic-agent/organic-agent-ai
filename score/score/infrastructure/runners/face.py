"""얼굴 값 — 주인공 얼굴의 눈 감김 · 웃음(#164). torch 없음(OpenCV YuNet 검출 + MediaPipe Face Landmarker).

    사진(긴 변 1024) → YuNet 얼굴 검출 → 주인공 2명 → 얼굴 상자의 1.8배 크롭(384) → Face Landmarker 블렌드셰이프
      eye_blink   가장 많이 감긴 얼굴의 눈 감김(두 눈 중 큰 쪽, 0~1)
      eye_smile   그 얼굴의 웃음(입꼬리 평균, 0~1) — 웃으면서 감은 눈을 가르는 재료
      smile       주인공 얼굴의 웃음 평균(0~1)
      face_n · face_measured   주인공 얼굴 수(0~2)와 그중 값을 잰 수

주인공 = 키 큰 순 2명 중 가장 큰 얼굴의 절반 이상인 것(지나가는 사람 · 배경 얼굴을 뺀다). 왼쪽부터.
옆얼굴 · 작은 얼굴은 랜드마크가 실패한다(미리보기 기준 얼굴의 12%) — 그때는 값이 None 이고 사진은 실패가 아니다.
값과 추천 실측은 docs/experiments/reco-rubric-2026-11 (g-gates · e-reco-v3).

모델 파일 둘은 고정 URL + 체크섬으로 받아 weights/ 에 캐시한다(빌드 시 굽는다). YuNet 은 MIT, MediaPipe 는 Apache-2.0.
검출기 · 랜드마커는 스레드 안전하지 않아 스레드마다 하나씩 만든다 — GPU 워커는 디코드 스레드(`decode_workers`)에서 돈다.
"""

from __future__ import annotations

import threading

import numpy as np

from PIL import Image

from score.domain.photo import FACE_UNMEASURED
from score.infrastructure.runners.common import fetch_weight

#: opencv_zoo 커밋 고정 — main 이 바뀌어도 같은 파일을 받는다.
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/f12e12798e8314f7c074a6656816c048dcc95b7a/"
             "models/face_detection_yunet/face_detection_yunet_2023mar.onnx")
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
LANDMARKER_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/"
                  "float16/1/face_landmarker.task")
LANDMARKER_SHA256 = "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"

#: 검출 점수 하한 · NMS · 후보 수. 실측과 같은 값.
DETECT_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 5000
#: 주인공으로 볼 얼굴 수, 그리고 가장 큰 얼굴 대비 최소 키 비율.
MAX_FACES = 2
MIN_HEIGHT_RATIO = 0.5
#: 랜드마커에 넣는 크롭 — 얼굴 상자(긴 변)의 배율과 한 변 픽셀.
CROP_MARGIN = 1.8
CROP_SIDE = 384
#: 랜드마커의 얼굴 검출 · 존재 신뢰도 하한. 기본 0.5 는 작은 얼굴을 너무 많이 버린다.
LANDMARK_CONFIDENCE = 0.3


def main_faces(boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """검출된 얼굴 상자 (x, y, w, h) 중 주인공만 — 키 큰 순 `MAX_FACES` 명에서 가장 큰 얼굴의 `MIN_HEIGHT_RATIO` 이상. 왼쪽부터."""
    if not boxes:
        return []
    tallest = sorted(boxes, key=lambda b: -b[3])
    mains = [b for b in tallest[:MAX_FACES] if b[3] >= MIN_HEIGHT_RATIO * tallest[0][3]]
    return sorted(mains, key=lambda b: b[0])


class FaceRunner:
    def __init__(self) -> None:
        self._yunet = str(fetch_weight(YUNET_URL, "face_detection_yunet_2023mar.onnx", YUNET_SHA256))
        self._landmarker = str(fetch_weight(LANDMARKER_URL, "face_landmarker.task", LANDMARKER_SHA256))
        self._local = threading.local()
        self._models()   # 한 번 만들어 본다 — 깨진 설치(라이브러리 · 모델 파일)가 첫 사진이 아니라 여기서 드러난다

    def _models(self):
        """이 스레드의 (검출기, 랜드마커). 처음 부를 때 만든다."""
        models = getattr(self._local, "models", None)
        if models is None:
            import cv2
            import mediapipe as mp

            detector = cv2.FaceDetectorYN.create(self._yunet, "", (320, 320), DETECT_THRESHOLD, NMS_THRESHOLD, TOP_K)
            landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=self._landmarker),
                output_face_blendshapes=True, num_faces=1,
                min_face_detection_confidence=LANDMARK_CONFIDENCE, min_face_presence_confidence=LANDMARK_CONFIDENCE))
            models = self._local.models = (detector, landmarker)
        return models

    def detect(self, image: Image.Image) -> list[tuple[float, float, float, float]]:
        """주인공 얼굴 상자 [(x, y, w, h)] — 픽셀, 왼쪽부터."""
        import cv2

        detector, _ = self._models()
        detector.setInputSize((image.width, image.height))
        _, found = detector.detect(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        return main_faces([tuple(float(v) for v in f[:4]) for f in (found if found is not None else [])])

    def blendshapes(self, image: Image.Image, box: tuple[float, float, float, float]) -> dict[str, float] | None:
        """얼굴 상자 하나의 블렌드셰이프. 랜드마크를 못 잡으면 None."""
        import mediapipe as mp

        _, landmarker = self._models()
        x, y, w, h = box
        side = max(w, h) * CROP_MARGIN
        cx, cy = x + w / 2, y + h / 2
        crop = image.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)))
        crop = crop.resize((CROP_SIDE, CROP_SIDE), Image.BICUBIC)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(crop))))
        if not result.face_blendshapes:
            return None
        return {c.category_name: float(c.score) for c in result.face_blendshapes[0]}

    def measure(self, image: Image.Image) -> dict[str, float | int | None]:
        """이미 디코드한 RGB 이미지(긴 변 1024) → 얼굴 값. 얼굴이 없으면 face_n 0 에 값은 None."""
        boxes = self.detect(image)
        faces = []
        for box in boxes:
            shapes = self.blendshapes(image, box)
            if shapes is not None:
                faces.append((max(shapes["eyeBlinkLeft"], shapes["eyeBlinkRight"]),
                              (shapes["mouthSmileLeft"] + shapes["mouthSmileRight"]) / 2))
        out: dict[str, float | int | None] = dict(FACE_UNMEASURED, face_n=len(boxes), face_measured=len(faces))
        if faces:
            blink, smile = max(faces, key=lambda f: f[0])
            out.update(eye_blink=blink, eye_smile=smile, smile=sum(s for _, s in faces) / len(faces))
        return out
