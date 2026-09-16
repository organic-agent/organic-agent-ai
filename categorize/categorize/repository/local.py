"""로컬 저장소 — out/v3/<갤러리 slug>/ 아래 파일. score 의 LocalStore 와 같은 규약이라 두 CLI 를 이어 돌릴 수 있다.

DB 없이 데이터셋으로 돌릴 때(`--local`)와 테스트가 쓴다. 로컬은 임베더가 없어 CLIP 이 임베딩 역할을 겸한다.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from categorize.domain.analysis import ConceptAssignment, GalleryRead, PhotoAnalysis


class LocalStore:
    """out/v3/<갤러리 slug>/ 아래 파일 — score 의 LocalStore 와 같은 규약. 로컬은 임베더가 없어 CLIP 이 임베딩 역할을 겸한다."""

    def __init__(self, root: Path, dataset_root: Path | None = None) -> None:
        self.root = Path(root)
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def preview_paths(self, gallery: str, photo_ids: list[str]) -> dict[str, str]:
        if self.dataset_root is None:
            return {}
        found = {pid: self.dataset_root / pid for pid in photo_ids}
        return {pid: str(p) for pid, p in found.items() if p.is_file()}

    def _dir(self, gallery: str) -> Path:
        d = self.root / "v3" / gallery.replace("/", "__")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def read_analysis(self, gallery: str) -> list[PhotoAnalysis]:
        p = self._dir(gallery) / "analysis.jsonl"
        if not p.exists():
            return []
        with p.open(encoding="utf-8") as f:
            return [PhotoAnalysis(**json.loads(line)) for line in f if line.strip()]

    def _read_npy(self, gallery: str, name: str) -> tuple[list[str], np.ndarray]:
        d = self._dir(gallery)
        ids_p, emb_p = d / f"{name}_ids.json", d / f"{name}.npy"
        if not (ids_p.exists() and emb_p.exists()):
            return [], np.zeros((0, 0))
        return json.loads(ids_p.read_text(encoding="utf-8")), np.load(emb_p)

    def read_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        return self._read_npy(gallery, "embeddings")

    def read_clip_embeddings(self, gallery: str) -> tuple[list[str], np.ndarray]:
        return self._read_npy(gallery, "clip_embeddings")

    def read_gallery(self, gallery: str) -> GalleryRead:
        rows = [r for r in self.read_analysis(gallery) if r.model_version]
        emb_ids, E = self.read_embeddings(gallery)
        clip_ids, C = self.read_clip_embeddings(gallery)
        return GalleryRead(rows=rows,
                           embeddings=dict(zip(emb_ids, E)) if len(emb_ids) else {},
                           clip_embeddings=dict(zip(clip_ids, C)) if len(clip_ids) else {})

    def _write_rows(self, gallery: str, rows: list[PhotoAnalysis]) -> None:
        with (self._dir(gallery) / "analysis.jsonl").open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    def _write_npy(self, gallery: str, name: str, ids: list[str], emb: np.ndarray) -> None:
        d = self._dir(gallery)
        (d / f"{name}_ids.json").write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
        np.save(d / f"{name}.npy", emb)

    def write_groups(self, gallery: str, rows: list[PhotoAnalysis]) -> None:
        """CATEGORIZE 의 필드만 덮는다 — subjects · model_version 은 score 의 것이라 그대로."""
        by_id = {r.photo_id: r for r in self.read_analysis(gallery)}
        for r in rows:
            cur = by_id.get(r.photo_id)
            if cur is None:
                by_id[r.photo_id] = r
                continue
            for f in ("technical_pct", "aesthetic_pct", "sub_scores", "cluster_id", "cluster_rank", "embed_group_id"):
                setattr(cur, f, getattr(r, f))
        self._write_rows(gallery, list(by_id.values()))

    def write_analysis(self, gallery: str, rows: list[PhotoAnalysis],
                       embeddings: tuple[list[str], np.ndarray],
                       clip_embeddings: tuple[list[str], np.ndarray]) -> None:
        """전부 한 번에 (테스트·합성 데이터용). 임베더 벡터는 여기서만 쓴다."""
        self._write_rows(gallery, rows)
        for name, (ids, emb) in (("embeddings", embeddings), ("clip_embeddings", clip_embeddings)):
            self._write_npy(gallery, name, ids, emb)

    def write_assignments(self, gallery: str, job_id: int | None,
                          rows: list[ConceptAssignment]) -> None:
        p = self._dir(gallery) / "assignments.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps({"job_id": job_id, **asdict(r)}, ensure_ascii=False) + "\n")

    def read_assignments(self, gallery: str) -> list[ConceptAssignment]:
        p = self._dir(gallery) / "assignments.jsonl"
        if not p.exists():
            return []
        out = []
        with p.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    d.pop("job_id", None)
                    out.append(ConceptAssignment(**d))
        return out
