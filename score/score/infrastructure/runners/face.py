"""얼굴 값 — 주인공 얼굴의 눈 감김 · 웃음(#164 · #170). torch 없음(OpenCV YuNet 검출 + MediaPipe Face Landmarker).

    사진(긴 변 1024) → YuNet 얼굴 검출 → 주인공 2명 → 얼굴 상자의 1.8배 크롭 → 값을 잰다
      eye_blink   가장 많이 감긴 얼굴의 눈 감김(두 눈 중 큰 쪽, 0~1)
      eye_smile   그 얼굴의 웃음(입꼬리 평균, 0~1) — 웃으면서 감은 눈을 가르는 재료
      smile       주인공 얼굴의 웃음 평균(0~1)
      face_n · face_measured   주인공 얼굴 수(0~2)와 그중 값을 잰 수
      face_src    값을 잰 모델 — mp(MediaPipe) · lora(DINOv3 + LoRA 측정기)
      faces       얼굴마다 상자(사진 대비 비율) · 값 · 출처 — 나중에 크롭을 다시 만들어 측정기를 다시 학습하는 재료

값을 재는 모델은 둘 중 하나다(`Knobs.face_source`):
    mediapipe   이 러너가 Face Landmarker 블렌드셰이프로 잰다. 옆얼굴은 랜드마크가 실패해 값이 없다(갤러리에 따라 얼굴의 12~32%).
    lora        이 러너는 검출과 크롭만 하고(`landmarks=False`), 파이프라인이 크롭을 모아 `face_lora.FaceValueRunner` 로 잰다.
                MediaPipe 를 교사로 학습한 학생이라 값의 뜻 · 눈금이 같고, 랜드마크를 거치지 않아 옆얼굴에도 값을 낸다.
주인공 = 키 큰 순 2명 중 가장 큰 얼굴의 절반 이상인 것(지나가는 사람 · 배경 얼굴을 뺀다). 왼쪽부터.
값이 없는 얼굴은 None 이고 사진은 실패가 아니다. 실측은 docs/experiments/reco-rubric-2026-11 (g-gates · e-reco-v3 · j-lora-robust).

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
#: LoRA 러너에 넘기는 크롭 한 변 — 학습 때와 같다(같은 여백으로 잘라 256 으로 줄이고, 러너가 가운데 224 를 쓴다).
LORA_CROP_SIDE = 256
#: 얼굴 값의 출처.
SRC_MEDIAPIPE, SRC_LORA = "mp", "lora"


def main_faces(boxes: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    """검출된 얼굴 상자 (x, y, w, h) 중 주인공만 — 키 큰 순 `MAX_FACES` 명에서 가장 큰 얼굴의 `MIN_HEIGHT_RATIO` 이상. 왼쪽부터."""
    if not boxes:
        return []
    tallest = sorted(boxes, key=lambda b: -b[3])
    mains = [b for b in tallest[:MAX_FACES] if b[3] >= MIN_HEIGHT_RATIO * tallest[0][3]]
    return sorted(mains, key=lambda b: b[0])


def summarize(faces: list[dict]) -> dict[str, object]:
    """얼굴별 값 → 사진의 얼굴 값. `faces` 는 왼쪽부터, 각 {"box", "blink", "smile", "src"} 이고 못 잰 얼굴은 값이 None 이다."""
    valued = [f for f in faces if f["blink"] is not None]
    out: dict[str, object] = dict(FACE_UNMEASURED, face_n=len(faces), face_measured=len(valued), faces=faces)
    if valued:
        worst = max(valued, key=lambda f: f["blink"])
        out.update(eye_blink=worst["blink"], eye_smile=worst["smile"], smile=sum(f["smile"] for f in valued) / len(valued),
                   face_src=valued[0]["src"])
    return out


class FaceRunner:
    def __init__(self, landmarks: bool = True) -> None:
        """`landmarks=False` 면 검출과 크롭만 한다 — 값은 LoRA 측정기가 잰다. MediaPipe 를 올리지 않는다."""
        self.landmarks = landmarks
        self._yunet = str(fetch_weight(YUNET_URL, "face_detection_yunet_2023mar.onnx", YUNET_SHA256))
        self._landmarker = str(fetch_weight(LANDMARKER_URL, "face_landmarker.task", LANDMARKER_SHA256)) if landmarks else None
        self._local = threading.local()
        self._models()   # 한 번 만들어 본다 — 깨진 설치(라이브러리 · 모델 파일)가 첫 사진이 아니라 여기서 드러난다

    def _models(self):
        """이 스레드의 (검출기, 랜드마커). 처음 부를 때 만든다. 랜드마커는 `landmarks` 일 때만 있다."""
        models = getattr(self._local, "models", None)
        if models is None:
            import cv2

            detector = cv2.FaceDetectorYN.create(self._yunet, "", (320, 320), DETECT_THRESHOLD, NMS_THRESHOLD, TOP_K)
            landmarker = None
            if self.landmarks:
                import mediapipe as mp

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

    @staticmethod
    def crop(image: Image.Image, box: tuple[float, float, float, float]) -> Image.Image:
        """얼굴 상자(긴 변)의 `CROP_MARGIN` 배 정사각형. 사진 밖은 검게 채워진다."""
        x, y, w, h = box
        side = max(w, h) * CROP_MARGIN
        cx, cy = x + w / 2, y + h / 2
        return image.crop((round(cx - side / 2), round(cy - side / 2), round(cx + side / 2), round(cy + side / 2)))

    def blendshapes(self, image: Image.Image, box: tuple[float, float, float, float]) -> dict[str, float] | None:
        """얼굴 상자 하나의 블렌드셰이프. 랜드마크를 못 잡으면 None."""
        import mediapipe as mp

        _, landmarker = self._models()
        crop = self.crop(image, box).resize((CROP_SIDE, CROP_SIDE), Image.BICUBIC)
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(np.asarray(crop))))
        if not result.face_blendshapes:
            return None
        return {c.category_name: float(c.score) for c in result.face_blendshapes[0]}

    def measure(self, image: Image.Image) -> dict[str, object]:
        """이미 디코드한 RGB 이미지(긴 변 1024) → 얼굴 값. 얼굴이 없으면 face_n 0 에 값은 None.

        `landmarks=False` 면 값 대신 얼굴마다 (얼굴 번호, 256 크롭)을 `_pending` 에 담는다 — 저장하는 키가 아니다. 받은 쪽이 꺼내
        LoRA 측정기로 재고 `summarize` 로 사진의 값을 낸다."""
        faces, pending = [], []
        for box in self.detect(image):
            x, y, w, h = box
            face = {"box": [round(x / image.width, 4), round(y / image.height, 4), round(w / image.width, 4), round(h / image.height, 4)],
                    "blink": None, "smile": None, "src": None}
            if not self.landmarks:
                pending.append((len(faces), self.crop(image, box).resize((LORA_CROP_SIDE, LORA_CROP_SIDE), Image.LANCZOS)))
            elif (shapes := self.blendshapes(image, box)) is not None:
                face.update(blink=max(shapes["eyeBlinkLeft"], shapes["eyeBlinkRight"]),
                            smile=(shapes["mouthSmileLeft"] + shapes["mouthSmileRight"]) / 2, src=SRC_MEDIAPIPE)
            faces.append(face)
        out = summarize(faces)
        if not self.landmarks:
            out["_pending"] = pending
        return out
