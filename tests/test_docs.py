"""Keep docs/code honest.

* docs/code/api_reference.md must be what scripts/make_api_reference.py generates now.
* The hand-written pages in docs/code may only name files and `module.name` things that exist.
"""

from __future__ import annotations

import ast
import importlib.util
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "pdac_cfrna"
SCRIPTS = ROOT / "scripts"
DOCS = ROOT / "docs" / "code"
HAND_WRITTEN = sorted(p for p in DOCS.glob("*.md") if p.name != "api_reference.md")

MODULES = {p.stem: p for p in SRC.glob("*.py") if p.name != "__init__.py"}
FILE_SUFFIXES = {"py", "md", "json", "csv", "yaml", "png", "joblib", "diff", "lock"}
SYMBOL = re.compile(r"`([a-z_][a-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)(?:\(\))?`")
FILE = re.compile(r"`([A-Za-z0-9_./-]+\.(?:py|yaml))`")


def _generator():
    spec = importlib.util.spec_from_file_location(
        "make_api_reference", SCRIPTS / "make_api_reference.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _top_level_names(path: Path) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def test_api_reference_is_up_to_date() -> None:
    generator = _generator()
    current = (DOCS / "api_reference.md").read_text(encoding="utf-8")
    assert current == generator.build(), (
        "docs/code/api_reference.md is out of date: run "
        "`uv run python scripts/make_api_reference.py`"
    )


@pytest.mark.parametrize("page", HAND_WRITTEN, ids=lambda p: p.name)
def test_named_functions_and_classes_exist(page: Path) -> None:
    missing = []
    for module, name in SYMBOL.findall(page.read_text(encoding="utf-8")):
        if (
            module in MODULES
            and name not in FILE_SUFFIXES
            and name not in _top_level_names(MODULES[module])
        ):
            missing.append(f"{module}.{name}")
    assert not missing, f"{page.name} names things that do not exist: {sorted(set(missing))}"


@pytest.mark.parametrize("page", HAND_WRITTEN, ids=lambda p: p.name)
def test_named_files_exist(page: Path) -> None:
    missing = []
    for token in FILE.findall(page.read_text(encoding="utf-8")):
        if "/" in token:
            exists = (ROOT / token).is_file()
        elif token.endswith(".py"):
            exists = (SRC / token).is_file() or (SCRIPTS / token).is_file()
        else:
            continue  # bare .yaml names (dataset.yaml, resolved_config.yaml) are not checked
        if not exists:
            missing.append(token)
    assert not missing, f"{page.name} names files that do not exist: {sorted(set(missing))}"
