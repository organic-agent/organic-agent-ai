"""로컬 저장소 — out/preference/<gallery>/data.npz + selected.json. DB 에서 한 번 내려받아(export) 두면 터널 없이 돈다."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from preference.domain.gallery import GalleryData


class LocalStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _dir(self, gallery_id: str) -> Path:
        d = self.root / "preference" / str(gallery_id)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def read_gallery(self, gallery_id: str) -> GalleryData:
        p = self._dir(gallery_id) / "data.npz"
        if not p.exists():
            raise FileNotFoundError(f"{p} 가 없다 — `python -m preference export --gallery-id {gallery_id}` 먼저")
        z = np.load(p, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return GalleryData(
            gallery_id=str(gallery_id),
            photo_ids=list(z["photo_ids"]), file_names=list(z["file_names"]),
            technical_pct=z["technical_pct"], aesthetic_pct=z["aesthetic_pct"], sharpness_pct=z["sharpness_pct"],
            subjects=list(z["subjects"]), cluster_id=z["cluster_id"], cluster_rank=z["cluster_rank"],
            embed_group_id=z["embed_group_id"], display_order=z["display_order"],
            embedding=z["embedding"], clip_embedding=z["clip_embedding"],
            embedding_model=meta.get("embedding_model", ""), model_version=meta.get("model_version", ""),
            shoot_type=meta.get("shoot_type"),
        )

    def write_gallery(self, gd: GalleryData) -> Path:
        p = self._dir(gd.gallery_id) / "data.npz"
        meta = json.dumps({"embedding_model": gd.embedding_model, "model_version": gd.model_version,
                           "shoot_type": gd.shoot_type})
        np.savez_compressed(
            p, meta=np.array(meta),
            photo_ids=np.array(gd.photo_ids), file_names=np.array(gd.file_names),
            technical_pct=gd.technical_pct, aesthetic_pct=gd.aesthetic_pct, sharpness_pct=gd.sharpness_pct,
            subjects=np.array(gd.subjects), cluster_id=gd.cluster_id, cluster_rank=gd.cluster_rank,
            embed_group_id=gd.embed_group_id, display_order=gd.display_order,
            embedding=gd.embedding.astype(np.float32), clip_embedding=gd.clip_embedding.astype(np.float32),
        )
        return p

    def read_selected(self, gallery_id: str) -> list[str]:
        p = self._dir(gallery_id) / "selected.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def write_selected(self, gallery_id: str, photo_ids: list[str]) -> None:
        (self._dir(gallery_id) / "selected.json").write_text(json.dumps(photo_ids), encoding="utf-8")

    def list_closed_galleries(self) -> list[str]:
        base = self.root / "preference"
        if not base.exists():
            return []
        return sorted(d.name for d in base.iterdir() if (d / "data.npz").exists() and (d / "selected.json").exists())

    def write_model(self, row: dict) -> int | None:
        """로컬은 항상 파일 — models/<n>.json."""
        d = self.root / "preference" / "models"
        d.mkdir(parents=True, exist_ok=True)
        n = len(list(d.glob("*.json"))) + 1
        (d / f"{n:04d}.json").write_text(json.dumps(row, ensure_ascii=False, indent=1), encoding="utf-8")
        return n
