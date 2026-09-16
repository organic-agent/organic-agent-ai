"""경계 — torch 없음 · score 와 공유하는 상수(MODEL_VERSION · PARENTS)가 같은 값인가."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from categorize.config.settings import MODEL_VERSION, MODULE_ROOT, PARENTS


def test_module_never_imports_torch():
    code = ("import sys; import categorize.service.pipeline, categorize.service.naming, categorize.service.job, "
            "categorize.controller.handler; assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=MODULE_ROOT)


def _literal(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        targets = getattr(node, "targets", None) or ([node.target] if isinstance(node, ast.AnnAssign) else [])
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{path}: {name} 없음")


def _score_settings() -> Path:
    """score 의 상수 파일 — 층 구조(#130) 뒤에는 config/settings.py, 그 전에는 config.py."""
    base = MODULE_ROOT.parent / "score" / "score"
    for candidate in (base / "config" / "settings.py", base / "config.py"):
        if candidate.is_file():
            return candidate
    raise AssertionError(f"score 설정 파일이 없다: {base}")


def test_model_version_and_parents_match_score_module():
    other = _score_settings()
    assert _literal(other, "MODEL_VERSION") == MODEL_VERSION
    assert _literal(other, "PARENTS") == PARENTS
    assert "기타" in PARENTS and len(PARENTS) == len(set(PARENTS))
