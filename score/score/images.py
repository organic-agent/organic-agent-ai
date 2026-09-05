"""이미지 로드 — torch 없는 공통 모듈. 러너·classical·pipeline 이 같은 픽셀을 본다.

서비스에서는 embedder 가 만든 preview JPEG(긴 변 **1024** = embedder `RESIZE_LONG_EDGE` 기본값, EXIF 회전 반영)을
읽는다. 로컬 데이터셋의 원본은 여기서 같은 모양으로 맞춘다 — 해상도가 바뀌면 점수가 바뀐다(study/00 step2).
(#51 실측 전까지 1600 으로 적혀 있었다 — S3 실물 683×1024 로 정정. ARNIQA 는 1024 아래로 내리면 순위가 무너진다.)

한 장은 **한 번만 디코드**한다(#51): pipeline 이 `load_image` 로 1600px PIL 이미지를 만들고, 러너들은
`as_image` 로 받아 각자 필요한 크기(ARNIQA 1024 · classical 1024 · CLIP 224)로 줄인다(미리보기가 1024 라 ARNIQA·classical 은 그대로). 경로(str)를 넘겨도
동작한다 — 스크립트·테스트용.
"""

from __future__ import annotations

from PIL import Image, ImageOps

try:  # 아이폰 HEIC. 없으면 그 사진들만 실패한다.
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:  # pragma: no cover
    pass

#: embedder 의 `resize_long_edge`(RESIZE_LONG_EDGE, 기본 1024) 와 같아야 한다.
PREVIEW_LONG_EDGE = 1024


def fit_long_edge(img: Image.Image, long_edge: int) -> Image.Image:
    """긴 변이 long_edge 보다 크면 LANCZOS 로 줄인다. 작으면 그대로(키우지 않는다)."""
    w, h = img.size
    scale = long_edge / max(w, h)
    if scale < 1:
        return img.resize((round(w * scale), round(h * scale)), Image.LANCZOS)
    return img


def load_image(path: str, long_edge: int = PREVIEW_LONG_EDGE) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("RGB")
    return fit_long_edge(img, long_edge)


def as_image(source: str | Image.Image, long_edge: int = PREVIEW_LONG_EDGE) -> Image.Image:
    """경로면 로드, 이미 PIL 이미지면 크기만 맞춘다."""
    if isinstance(source, Image.Image):
        return fit_long_edge(source, long_edge)
    return load_image(source, long_edge)
