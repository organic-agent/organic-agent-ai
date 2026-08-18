"""Vercel 서버리스 진입점 — 모든 경로가 이 함수로 rewrite된다 (vercel.json)."""

from notionchat.api.main import app  # noqa: F401 — Vercel이 ASGI `app`을 감지해 서빙
