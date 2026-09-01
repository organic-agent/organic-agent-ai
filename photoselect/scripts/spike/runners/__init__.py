"""러너 레지스트리 — 이름으로 지연 로드 (선택한 러너의 의존성만 임포트)."""

from __future__ import annotations

from typing import Callable

REGISTRY: dict[str, Callable] = {}


def _register(name: str):
    def deco(factory: Callable):
        REGISTRY[name] = factory
        return factory

    return deco


@_register("faces")
def _faces():
    from .mediapipe_faces import FacesRunner

    return FacesRunner()


@_register("arniqa")
def _arniqa():
    from .arniqa import ArniqaRunner

    return ArniqaRunner()


@_register("laion_aesthetic")
def _laion():
    from .laion_aesthetic import LaionAestheticRunner

    return LaionAestheticRunner()


def create(name: str):
    if name not in REGISTRY:
        raise SystemExit(f"모르는 러너: {name} (가능: {', '.join(REGISTRY)})")
    return REGISTRY[name]()
