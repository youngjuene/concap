"""One offline canary per session, copied for the tests that need a store.

Three tests each built the same store from scratch — about eighteen seconds
apiece, roughly a quarter of the suite — and the landing workflow runs the full
check on every intermediate commit, so the cost is paid many times a day.

The run happens once. Tests that only read the store take its path; tests that
publish into it take a copy, so nothing they write can reach another test and
the order they run in stays irrelevant. The index stores paths relative to the
store root and the workspace sentinel carries only a uuid, so a plain directory
copy opens exactly like the original.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest

from dpo.pipeline.canary import CanaryResult, run_canary
from tests.conftest import CANARY_CONTRACT


@pytest.fixture(scope="session")
def canary_store(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, CanaryResult]:
    """The cold canary run and the store it published into."""
    store = tmp_path_factory.mktemp("canary") / "store"
    return store, run_canary(store, CANARY_CONTRACT)


@pytest.fixture
def canary_copy(canary_store: tuple[Path, CanaryResult], tmp_path: Path) -> Iterator[Path]:
    """A private copy of that store, for a test that publishes into one."""
    destination = tmp_path / "store"
    shutil.copytree(canary_store[0], destination, symlinks=False)
    yield destination
