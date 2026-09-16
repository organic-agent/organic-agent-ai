"""저장소 계약 — service 는 이 Protocol 만 본다. 구현은 `local_store.LocalStore`(npz) · `db_store.DbStore`(Postgres)."""

from __future__ import annotations

from typing import Protocol

from preference.domain.gallery import GalleryData


class Store(Protocol):
    def read_gallery(self, gallery_id: str) -> GalleryData: ...
    def read_selected(self, gallery_id: str) -> list[str]: ...
    def list_closed_galleries(self) -> list[str]: ...
    def write_model(self, row: dict) -> int | None: ...
