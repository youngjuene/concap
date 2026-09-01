"""The two instruments stay detached, so either can be deleted whole.

Two specifications disagree about the surface, and the repository builds both
so one can be adopted and the other archived. That only works while the seam
holds: the shared package must not know about either instrument, and neither
instrument may reach into the other. This file is the seam, asserted.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "src" / "dpo"


def _imported_modules(path: Path) -> set[str]:
    """Every module name this file imports, including inside functions."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
    return found


def _package_files(package: str) -> list[Path]:
    files = sorted((SOURCE / package).rglob("*.py"))
    assert files, f"no sources under {package}"
    return files


def _reaches(package: str, forbidden: str) -> list[str]:
    offenders = []
    for path in _package_files(package):
        for module in _imported_modules(path):
            if module == forbidden or module.startswith(forbidden + "."):
                offenders.append(f"{path.relative_to(SOURCE)} imports {module}")
    return offenders


@pytest.mark.parametrize("forbidden", ["dpo.session", "dpo.console"])
def test_the_shared_package_knows_about_neither_instrument(forbidden: str) -> None:
    """The dependency runs one way: instruments import dpo.caption, never the reverse."""
    assert _reaches("caption", forbidden) == []


def test_the_console_never_reaches_into_the_skeleton_instrument() -> None:
    assert _reaches("console", "dpo.session") == []


def test_the_skeleton_instrument_never_reaches_into_the_console() -> None:
    assert _reaches("session", "dpo.console") == []


def test_each_command_module_touches_only_its_own_instrument() -> None:
    console_cli = _imported_modules(SOURCE / "cli" / "console.py")
    session_cli = _imported_modules(SOURCE / "cli" / "session.py")
    assert not any(module.startswith("dpo.session") for module in console_cli)
    assert not any(module.startswith("dpo.console") for module in session_cli)


def test_both_instruments_do_share_the_caption_machinery() -> None:
    """The seam is worth asserting in the other direction too: it is used."""
    for package in ("session", "console"):
        assert _reaches(package, "dpo.caption"), f"{package} does not use the shared writers"
