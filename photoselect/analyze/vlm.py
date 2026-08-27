"""A-4/A-5 고정 축 태그 + 캡션 — 오픈 VLM.

지금은 로컬 Ollama(structured outputs로 enum 강제). 프로덕션은 vLLM guided decoding —
프롬프트·스키마는 같고 호출부만 바뀐다. 이미지는 외부로 나가지 않는다(CLAUDE.md).

프롬프트는 **2026-08-25 1차안 원문 그대로**다. 고치지 말 것 — 근거:
  · 1차 재실행은 같은 50장에서 250칸 전부 동일 (모델은 결정적, temperature 0)
  · v2(축 규칙 추가): scene 74→66, framing 94→82, lighting 84→80
  · v3(캡션 문장만 변경): scene 74→62, lighting 84→74 — **다른 부분을 건드려도 축 정확도가 흔들린다**
  gemma3:12b 4bit에서 프롬프트 튜닝은 이 표본으로 개선 불가. scene 74%는 모델 크기(31B 비교)와
  데이터(본식 갤러리 부재) 쪽 문제로 넘긴다. 상세: `scripts/spike/vlm_compare.py`, STATUS.md 1.1
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
scene — 결혼식 진행 순서상 어느 장면인가
  prep 준비(신부대기실·메이크업) · entrance 입장 · vow 서약/예식 · ring 반지 교환
  kiss 키스 · family 가족 사진 · group 단체 사진 · bouquet 부케
  walk 퇴장/행진 · snap 스냅/자유 컷 · detail 소품·공간·음식 등 사물 위주
  unknown 위 어느 것도 아니거나 판단 불가
framing — 인물이 프레임을 차지하는 크기
  closeup 얼굴 위주 · half 상반신 · full 전신 · wide 원경/공간이 주가 되는 컷
lighting — 지배적인 광원
  natural 자연광 · backlit 역광(인물 뒤에서 빛) · indoor 실내 조명
  flash 플래시 직광 · lowlight 어둡고 노이즈 있는 저조도
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
  예: "야외 자연광 아래 두 분이 마주 보며 웃는 컷"
  사진에 보이는 것만 쓴다. 추측·칭찬·감상은 넣지 않는다.

반드시 JSON만 출력한다. 설명 문장을 덧붙이지 않는다.\
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
