"""경량 3종 러너. 스파이크 하네스(`scripts/spike/runners/`)에서 이관했다 — 여기가 정본이다.

각 러너는 `name`, `version`, `score(path) -> dict[str, float]`를 갖는다.
`LaionRunner`는 `embed(path) -> np.ndarray`(CLIP 768d)도 준다 — 클러스터링이 쓴다.
"""

from photoselect.analyze.runners.arniqa import ArniqaRunner
from photoselect.analyze.runners.faces import FacesRunner
from photoselect.analyze.runners.laion import LaionRunner

__all__ = ["ArniqaRunner", "FacesRunner", "LaionRunner"]
