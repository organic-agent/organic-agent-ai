"""A-4/A-5 고정 축 태그 + 캡션 — 오픈 VLM.

지금은 로컬 Ollama(structured outputs로 enum 강제). 프로덕션은 vLLM guided decoding —
프롬프트·스키마는 같고 호출부만 바뀐다. 이미지는 외부로 나가지 않는다(CLAUDE.md).

프롬프트는 `scripts/spike/vlm_tag.py`의 2차안이다: 08-25 실측에서 오답이 단일 방향이었던
두 축(lighting: indoor→natural, scene: snap→예식 장면)에 판단 규칙을 넣었다.
효과는 같은 50장 재채점으로 확인한다(spike-report "다음").
"""

from __future__ import annotations

import base64
import io
import json
import logging
import urllib.error
import urllib.request

from photoselect.analyze.runners.common import load_image
from photoselect.axes import AXES

log = logging.getLogger(__name__)

AXIS_GUIDE = """\
scene — 예식 진행 순서상의 장면. **진행 순서상의 장면임이 확실할 때만** 그 값을 쓰고,
  야외 산책·대기·자유로운 포즈처럼 애매하면 snap.
  prep 준비(신부대기실·메이크업) · entrance 입장 · vow 서약/예식 · ring 반지 교환
  kiss 키스 · family 가족 사진 · group 단체 사진 · bouquet 부케
  walk 예식장 안에서의 퇴장/행진 (야외 산책 컷은 snap) · snap 스냅/자유 컷
  detail 소품·공간·음식 등 사물 위주 · unknown 판단 불가
framing — 인물이 프레임을 차지하는 크기
  closeup 얼굴 위주 · half 상반신 · full 전신 · wide 원경/공간이 주가 되는 컷
lighting — 지배적인 광원. **실내 공간이면 창으로 들어온 빛이라도 indoor.**
  natural 야외 자연광 · backlit 역광(인물 뒤에서 빛) · indoor 실내(조명·창빛 모두)
  flash 플래시 직광(그림자가 딱딱함) · lowlight 어둡고 노이즈 있는 저조도
expression — 주 인물의 표정
  smile 미소 · laugh 크게 웃음 · serious 진지/무표정 · candid 의식하지 않은 자연스러움
  eyes_closed 눈 감김 · none 얼굴이 없거나 식별 불가
subjects — 화면의 주 인물 구성
  bride 신부 단독 · groom 신랑 단독 · couple 신랑신부 · family 가족
  friends 친구/하객 · none 인물 없음\
"""

SYSTEM_PROMPT = f"""\
당신은 웨딩 사진을 분류하는 도구다. 사진 한 장을 보고 아래 5개 축을 각각 하나씩 고르고,
한국어 캡션 1문장을 쓴다.

{AXIS_GUIDE}

caption — 고객(신랑신부)에게 그대로 보여도 되는 톤의 한국어 1문장.
  사진에 보이는 것만 쓴다. 추측·칭찬·감상은 넣지 않는다. 다른 사진과 구별되는 구체적 요소
  (장소·동작·소품)를 한 가지 이상 넣는다.

반드시 JSON만 출력한다.\
"""

SCHEMA = {
    "type": "object",
    "properties": {
        **{ax: {"type": "string", "enum": vals} for ax, vals in AXES.items()},
        "caption": {"type": "string"},
    },
    "required": [*AXES.keys(), "caption"],
}


def _to_jpeg_b64(path: str, long_edge: int) -> str:
    img = load_image(path, long_edge=long_edge)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


class OllamaTagger:
    name = "vlm"

    def __init__(self, host: str, model: str, long_edge: int, timeout: float) -> None:
        self.host, self.model, self.long_edge, self.timeout = host, model, long_edge, timeout
        self.version = f"vlm-ollama-{model}"

    def available(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/api/version", timeout=3):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def tag(self, path: str) -> dict[str, str]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "이 사진을 분류하라.",
                 "images": [_to_jpeg_b64(path, self.long_edge)]},
            ],
            "format": SCHEMA,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 256},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())
        out = json.loads(body["message"]["content"])
        # enum 강제라 어휘 밖 값은 안 나오지만, 방어적으로 한 번 더 거른다
        return {ax: (out.get(ax) if out.get(ax) in AXES[ax] else AXES[ax][-1]) for ax in AXES} | {
            "caption": str(out.get("caption", "")).strip()
        }
