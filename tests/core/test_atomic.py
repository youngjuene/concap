"""The mutable atomic replace: whole files, and no temporaries left behind."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from dpo.core.atomic import replace_atomically


def test_a_replace_leaves_the_new_bytes_and_no_temporary(tmp_path: Path) -> None:
    target = tmp_path / "snapshot.json"
    replace_atomically(target, b"one")
    replace_atomically(target, b"two")
    assert target.read_bytes() == b"two"
    assert not list(tmp_path.glob(".*.tmp"))


def test_a_failed_replace_leaves_no_temporary_behind(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "snapshot.json"
    target.write_bytes(b"one")

    def refuse(source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> None:
        raise PermissionError("read-only file system")

    monkeypatch.setattr(os, "replace", refuse)
    with pytest.raises(PermissionError):
        replace_atomically(target, b"two")
    assert target.read_bytes() == b"one"
    assert not list(tmp_path.glob(".*.tmp"))
