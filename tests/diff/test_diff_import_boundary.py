"""The package boundary: what diff imports, what it must never import, and how it is named."""

from __future__ import annotations

import ast
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "src" / "diff"
ALLOWED_REGEN = {
    "dpo.regen.sheet_instrument",
    "dpo.regen.playback",
    "dpo.regen.points",
    "dpo.regen.study_media",
    "dpo.regen.study_store",
}


def imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def test_dpo_imports_stay_inside_the_allowed_regen_modules() -> None:
    for path in PACKAGE.glob("*.py"):
        for name in imported_modules(path):
            if name == "dpo" or name.startswith("dpo."):
                assert name in ALLOWED_REGEN, f"{path.name} imports {name}"


def test_no_dream_path_alias_or_switch_remains() -> None:
    # The researcher replaced Dream with Trida-7B on 2026-10-12. Prose may record history;
    # no module, profile, condition or call of Dream's may remain.
    assert not list(PACKAGE.glob("dream*.py"))
    forbidden = (
        "dream_runner",
        "DREAM_7B",
        "dream_k",
        "diffusion_generate",
        "dream_call_settings",
        "--dream-dir",
    )
    for path in PACKAGE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        for name in forbidden:
            assert name not in text, f"{path.name} still names {name}"


def test_importing_every_module_pulls_in_no_model_or_web_stack() -> None:
    modules = sorted(f"diff.{path.stem}" for path in PACKAGE.glob("*.py") if path.stem != "__init__")
    probe = (
        "import importlib, sys\n"
        f"for name in {modules!r}: importlib.import_module(name)\n"
        "heavy = ('torch', 'transformers', 'fastapi', 'numpy',"
        " 'dpo.models', 'dpo.candidates', 'dpo.contracts')\n"
        "print(sorted(m for m in heavy if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], cwd=ROOT, capture_output=True, text=True, check=True, timeout=120
    )
    assert result.stdout.strip() == "[]", result.stdout + result.stderr


def test_basenames_are_unique_and_shadow_nothing() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split()
    shared = {"__init__.py", "__main__.py", "conftest.py"}
    ours = [
        p for p in [*PACKAGE.glob("*.py"), *(ROOT / "tests" / "diff").glob("*.py")] if p.name not in shared
    ]
    for path in ours:
        others = [
            t for t in tracked if Path(t).name == path.name and not t.startswith(("src/diff/", "tests/diff/"))
        ]
        assert not others, f"{path.name} collides with {others}"
        assert path.stem not in sys.stdlib_module_names, path.name


def test_pyproject_ships_diff_without_a_diff_console_script() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert "src/diff" in project["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "src/diff" in project["tool"]["mypy"]["files"]
    assert "diff" not in project["project"].get("scripts", {})
