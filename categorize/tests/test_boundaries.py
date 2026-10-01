"""경계 — torch 없음 · score 와 공유하는 상수(파이프라인 버전)가 같은 값인가."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from categorize.config.settings import MODULE_ROOT, PIPELINE_VERSION


def test_module_never_imports_torch():
    code = ("import sys; import categorize.service.pipeline, categorize.service.naming, categorize.service.segment, "
            "categorize.service.job, categorize.controller.handler; assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=MODULE_ROOT)


def _literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", None) or ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path}: {name} 없음")


def test_pipeline_version_matches_score_module():
    """1층 고정 목록(CONCEPTS)은 더는 공유하지 않는다(컨셉 구간화 2026-09-30) — 파이프라인 버전만 같으면 된다."""
    other = MODULE_ROOT.parent / "score" / "score" / "config" / "settings.py"
    assert _literal(other, "PIPELINE_VERSION") == PIPELINE_VERSION
