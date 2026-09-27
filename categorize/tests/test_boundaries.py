"""경계 — torch 없음 · score 와 공유하는 상수(파이프라인 버전 · 1층 목록)이 같은 값인가."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from categorize.config.settings import CONCEPTS, MODULE_ROOT, PIPELINE_VERSION


def test_module_never_imports_torch():
    code = ("import sys; import categorize.service.pipeline, categorize.service.naming, categorize.service.job, "
            "categorize.controller.handler; assert 'torch' not in sys.modules, 'torch imported'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=MODULE_ROOT)


#: score 설정 파일에서 1층 목록·파이프라인 버전의 변수 이름 — score 가 정한 이름이다.
# [GLOSSARY-1 2026-09-27] score 쪽 이름 변경(PARENTS → CONCEPTS, MODEL_VERSION → PIPELINE_VERSION)은 score 커밋에서 맞춘다.
SCORE_CONCEPTS_NAME = "PARENTS"
SCORE_PIPELINE_VERSION_NAME = "MODEL_VERSION"


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


def test_pipeline_version_and_concepts_match_score_module():
    other = _score_settings()
    assert _literal(other, SCORE_PIPELINE_VERSION_NAME) == PIPELINE_VERSION
    assert _literal(other, SCORE_CONCEPTS_NAME) == CONCEPTS
    assert "기타" in CONCEPTS and len(CONCEPTS) == len(set(CONCEPTS))
