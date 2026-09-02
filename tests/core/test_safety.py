from __future__ import annotations

import json
from pathlib import Path

import pytest

from dpo.core.safety import (
    DestructivePathError,
    OwnedWorkspace,
)


def test_owned_workspace_deletes_only_claimed_children(tmp_path: Path) -> None:
    root = tmp_path / "owned"
    workspace = OwnedWorkspace.create(root)
    child = workspace.claim_child("staging")
    (child / "payload.json").write_text("{}", encoding="utf-8")
    workspace.safe_delete(child)
    assert not child.exists()


@pytest.mark.parametrize("target_name", ["root", "home", "cwd", "repo", "input", "unowned"])
def test_destructive_paths_fail_before_mutation(tmp_path: Path, target_name: str) -> None:
    workspace = OwnedWorkspace.create(tmp_path / "workspace")
    protected_input = tmp_path / "input"
    protected_input.mkdir()
    unowned = workspace.root / "unowned"
    unowned.mkdir()
    repo = Path(__file__).parents[2]
    targets = {
        "root": Path("/"),
        "home": Path.home(),
        "cwd": Path.cwd(),
        "repo": repo,
        "input": protected_input,
        "unowned": unowned,
    }
    marker = targets[target_name] / "do-not-touch" if target_name in {"input", "unowned"} else None
    if marker is not None:
        marker.write_text("present", encoding="utf-8")
    with pytest.raises(DestructivePathError):
        workspace.safe_delete(targets[target_name], protected_inputs=(protected_input,))
    if marker is not None:
        assert marker.read_text(encoding="utf-8") == "present"


def test_symlink_escape_is_rejected_without_touching_destination(tmp_path: Path) -> None:
    workspace = OwnedWorkspace.create(tmp_path / "workspace")
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "keep"
    marker.write_text("present", encoding="utf-8")
    link = workspace.root / "escape"
    link.symlink_to(outside, target_is_directory=True)
    with pytest.raises(DestructivePathError, match="symlink|outside"):
        workspace.safe_delete(link)
    assert marker.read_text(encoding="utf-8") == "present"


def test_workspace_create_and_open_reject_symlinked_ancestors_before_mutation(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "link"
    link.symlink_to(outside, target_is_directory=True)
    escaped = link / "new-workspace"
    with pytest.raises(DestructivePathError, match="symlink"):
        OwnedWorkspace.create(escaped)
    assert not (outside / "new-workspace").exists()

    real = OwnedWorkspace.create(outside / "owned")
    assert real.root.exists()
    with pytest.raises(DestructivePathError, match="symlink"):
        OwnedWorkspace.open(link / "owned")


def test_workspace_rejects_symlinked_sentinels(tmp_path: Path) -> None:
    root = tmp_path / "owned"
    root.mkdir()
    external = tmp_path / "sentinel.json"
    external.write_text(
        json.dumps({"schema": "dpo.workspace/v1", "workspace_id": "external"}),
        encoding="utf-8",
    )
    (root / ".dpo-workspace.json").symlink_to(external)
    with pytest.raises(DestructivePathError, match="sentinel.*symlink"):
        OwnedWorkspace.open(root)
