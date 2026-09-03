"""The instruments stay detached, so any one can be deleted whole.

Three specifications disagree about the surface, and the repository builds all
of them so one can be adopted and the rest archived. That only works while the
seam holds: the shared package must not know about any instrument, and no
instrument may reach into another. This file is the seam, asserted.

Parameterised over :data:`INSTRUMENTS` rather than over a written-out pair, so
adding a fourth instrument extends the seam instead of quietly leaving it
outside — which is what a hand-listed pair did the first time a third arrived.

It lives under tests/caption because it is a test of the shared package, and
because it has to survive the deletion it describes. It also watches the test
tree: an instrument's tests may not reach into another's either, or
`rm -rf tests/<loser>` leaves a red suite behind — which is how a test of the
skeleton's neighbours came to live in the console's directory.
"""

from __future__ import annotations

import ast
import itertools
from pathlib import Path

import pytest

# Every participant instrument, each with its own docs/vN-* specification.
# One list, so every rule below covers all of them.
INSTRUMENTS = ("session", "console", "regen")
PAIRS = list(itertools.permutations(INSTRUMENTS, 2))

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


def _built(package: str) -> bool:
    """Whether this instrument is still in the tree.

    After the archival one of them is not, and this file has to keep passing:
    a seam test that goes red the moment the seam is used for what it was for
    would be the last thing anyone wants on that day.
    """
    return (SOURCE / package).is_dir()


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


@pytest.mark.parametrize("forbidden", [f"dpo.{name}" for name in INSTRUMENTS])
def test_the_shared_package_knows_about_no_instrument(forbidden: str) -> None:
    """The dependency runs one way: instruments import dpo.caption, never the reverse.

    Asserted whether or not that instrument still exists: the shared package
    must not name an archived one either.
    """
    assert _reaches("caption", forbidden) == []


@pytest.mark.parametrize(("package", "other"), PAIRS)
def test_no_instrument_reaches_into_another(package: str, other: str) -> None:
    forbidden = f"dpo.{other}"
    if not _built(package):
        pytest.skip(f"{package} is archived")
    assert _reaches(package, forbidden) == []


@pytest.mark.parametrize(("module", "other"), PAIRS)
def test_each_command_module_touches_only_its_own_instrument(module: str, other: str) -> None:
    forbidden = f"dpo.{other}"
    path = SOURCE / "cli" / f"{module}.py"
    if not path.is_file():
        pytest.skip(f"{module} is archived")
    assert not any(name.startswith(forbidden) for name in _imported_modules(path))


@pytest.mark.parametrize("package", INSTRUMENTS)
def test_an_instrument_that_is_here_does_share_the_caption_machinery(package: str) -> None:
    """The seam is worth asserting in the other direction too: it is used."""
    if not _built(package):
        pytest.skip(f"{package} is archived")
    assert _reaches(package, "dpo.caption"), f"{package} does not use the shared writers"


def test_at_least_one_instrument_is_built() -> None:
    """All of them, until the decision; exactly one after it; never none."""
    assert [package for package in INSTRUMENTS if _built(package)]


def _tests_reaching(directory: str, forbidden: str) -> list[str]:
    """Where one instrument's tests import another's package, at any depth."""
    offenders = []
    for path in sorted((TESTS / directory).rglob("*.py")):
        for module in _imported_modules(path):
            if module == forbidden or module.startswith(forbidden + "."):
                offenders.append(f"{path.relative_to(TESTS)} imports {module}")
    return offenders


@pytest.mark.parametrize(("directory", "other"), PAIRS)
def test_an_instruments_tests_do_not_reach_into_another(directory: str, other: str) -> None:
    """Archiving one instrument deletes its own tests and leaves no red ones behind."""
    forbidden = f"dpo.{other}"
    if not (TESTS / directory).is_dir():
        pytest.skip(f"{directory} is archived")
    assert _tests_reaching(directory, forbidden) == []


def test_every_shared_module_is_tested_outside_both_instruments() -> None:
    """Coverage of what survives must not be deleted with what does not.

    Each module of the shared package has a test file under tests/caption, so
    `rm -rf tests/<instrument>` cannot take the last tests of the writers, the
    cache, the background thread or the ontology with it. An instrument's own
    tests may still name a shared type — that is not coverage of the shared
    package, and no import rule can tell the difference — so the invariant is
    stated over the files, not the imports.
    """
    modules = {path.stem for path in (SOURCE / "caption").glob("*.py")} - {"__init__"}
    tested = {path.stem.removeprefix("test_caption_") for path in TESTS.glob("caption/test_caption_*.py")}
    assert modules <= tested, f"no tests beside the shared package for: {sorted(modules - tested)}"
