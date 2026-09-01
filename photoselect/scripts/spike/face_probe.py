import sys, numpy as np
from PIL import Image, ImageOps
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
sys.path.insert(0, "src")
from photoselect.v1.runners.common import fetch_weight
from photoselect.v1.runners.faces import TASK_URL
task = str(fetch_weight(TASK_URL, "face_landmarker_f16_v1.task"))
def lm(conf=0.5):
    return vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mpp.BaseOptions(model_asset_path=task), output_face_blendshapes=False, num_faces=8,
        min_face_detection_confidence=conf))
L = {c: lm(c) for c in (0.5, 0.3)}
det = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
    base_options=mpp.BaseOptions(model_asset_path=str(fetch_weight(
        "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite","blaze_short.tflite"))),
    min_detection_confidence=0.3))
def prep(path, edge):
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    im.thumbnail((edge, edge), Image.LANCZOS); return im
base = "/Users/kanghyungjun/MyGIthub/organic-agent/dataset/dataset1/데이터셋1/"
files = sys.argv[1:]
print("file | edge | size | lm@.5 | lm@.3 | blaze@.3 | maxface_px")
for f in files:
    for edge in (1024, 1600, 2400, 4608):
        im = prep(base+f, edge); mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im))
        r5 = len(L[0.5].detect(mi).face_landmarks); r3 = len(L[0.3].detect(mi).face_landmarks)
        d = det.detect(mi).detections
        mx = max([x.bounding_box.width for x in d], default=0)
        print(f"{f} | {edge} | {im.size[0]}x{im.size[1]} | {r5} | {r3} | {len(d)} | {mx}")
