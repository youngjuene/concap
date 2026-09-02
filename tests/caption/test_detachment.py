"""The two instruments stay detached, so either can be deleted whole.

Two specifications disagree about the surface, and the repository builds both
so one can be adopted and the other archived. That only works while the seam
holds: the shared package must not know about either instrument, and neither
instrument may reach into the other. This file is the seam, asserted.

It lives under tests/caption because it is a test of the shared package, and
because it has to survive the deletion it describes. It also watches the test
tree: an instrument's tests may not reach into the other's either, or
`rm -rf tests/<loser>` leaves a red suite behind — which is how a test of the
skeleton's neighbours came to live in the console's directory.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src" / "dpo"
TESTS = ROOT / "tests"


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


def _tests_reaching(directory: str, forbidden: str) -> list[str]:
    """Where one instrument's tests import the other's package, at any depth."""
    offenders = []
    for path in sorted((TESTS / directory).rglob("*.py")):
        for module in _imported_modules(path):
            if module == forbidden or module.startswith(forbidden + "."):
                offenders.append(f"{path.relative_to(TESTS)} imports {module}")
    return offenders


@pytest.mark.parametrize(
    ("directory", "forbidden"),
    [("console", "dpo.session"), ("session", "dpo.console")],
)
def test_an_instruments_tests_do_not_reach_into_the_other(directory: str, forbidden: str) -> None:
    """Archiving one instrument deletes its own tests and leaves no red ones behind."""
    assert _tests_reaching(directory, forbidden) == []


def test_every_shared_module_is_tested_outside_both_instruments() -> None:
    """Coverage of what survives must not be deleted with what does not.

    Each module of the shared package has a test file under tests/caption, so
    `rm -rf tests/session` or `rm -rf tests/console` cannot take the last
    tests of the writers, the cache, the background thread or the ontology
    with it. An instrument's own tests may still name a shared type — that is
    not coverage of the shared package, and no import rule can tell the
    difference — so the invariant is stated over the files, not the imports.
    """
    modules = {path.stem for path in (SOURCE / "caption").glob("*.py")} - {"__init__"}
    tested = {path.stem.removeprefix("test_caption_") for path in TESTS.glob("caption/test_caption_*.py")}
    assert modules <= tested, f"no tests beside the shared package for: {sorted(modules - tested)}"
