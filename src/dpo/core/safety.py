from __future__ import annotations

import json
import os
import shutil
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dpo.core.identity import canonical_bytes, semantic_hash


class DestructivePathError(ValueError):
    """Raised before a destructive operation escapes an owned workspace."""


class CheckpointSafetyError(ValueError):
    """Raised before any unsafe or identity-mismatched checkpoint is deserialized."""


WORKSPACE_SENTINEL = ".dpo-workspace.json"
CHILD_SENTINEL = ".dpo-owned.json"
UNSAFE_CHECKPOINT_SUFFIXES = {".bin", ".pt", ".pth", ".pkl", ".pickle", ".ckpt"}


def _reject_symlink_components(path: Path, *, context: str) -> None:
    absolute = path.expanduser().absolute()
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current /= part
        if current.is_symlink():
            raise DestructivePathError(f"{context} contains a symlink component: {current}")


def _write_new_json(path: Path, document: Mapping[str, object]) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(canonical_bytes(document))
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class OwnedWorkspace:
    root: Path
    workspace_id: str

    @classmethod
    def create(cls, root: str | Path) -> OwnedWorkspace:
        candidate = Path(root).expanduser().absolute()
        _reject_symlink_components(candidate, context="workspace path")
        forbidden = {Path("/"), Path.home().resolve(), Path.cwd().resolve()}
        if candidate.resolve(strict=False) in forbidden:
            raise DestructivePathError(f"refusing unsafe workspace root {candidate}")
        if candidate.exists() and not candidate.is_dir():
            raise DestructivePathError("workspace root must be a directory")
        candidate.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(candidate, context="workspace path")
        sentinel = candidate / WORKSPACE_SENTINEL
        if sentinel.is_symlink():
            raise DestructivePathError("workspace sentinel cannot be a symlink")
        if sentinel.exists():
            try:
                document = json.loads(sentinel.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise DestructivePathError("workspace sentinel is invalid") from exc
            workspace_id = document.get("workspace_id")
            if not isinstance(workspace_id, str) or not workspace_id:
                raise DestructivePathError("workspace sentinel has no owner id")
            return cls(root=candidate.resolve(), workspace_id=workspace_id)
        visible_entries = [entry for entry in candidate.iterdir() if entry.name != WORKSPACE_SENTINEL]
        if visible_entries:
            raise DestructivePathError("refusing to claim a non-empty unowned directory")
        workspace_id = uuid.uuid4().hex
        _write_new_json(sentinel, {"schema": "dpo.workspace/v1", "workspace_id": workspace_id})
        return cls(root=candidate.resolve(), workspace_id=workspace_id)

    @classmethod
    def open(cls, root: str | Path) -> OwnedWorkspace:
        """Open an existing workspace without ever claiming or mutating a path."""
        candidate = Path(root).expanduser().absolute()
        _reject_symlink_components(candidate, context="workspace path")
        if not candidate.is_dir():
            raise DestructivePathError("workspace must be an existing real directory")
        sentinel = candidate / WORKSPACE_SENTINEL
        if sentinel.is_symlink():
            raise DestructivePathError("workspace sentinel cannot be a symlink")
        try:
            document = json.loads(sentinel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DestructivePathError("workspace is not sentinel-owned") from exc
        workspace_id = document.get("workspace_id")
        if document.get("schema") != "dpo.workspace/v1" or not isinstance(workspace_id, str):
            raise DestructivePathError("workspace sentinel is invalid")
        return cls(root=candidate.resolve(), workspace_id=workspace_id)

    def claim_child(self, name: str) -> Path:
        if not name or name in {".", ".."} or Path(name).name != name:
            raise DestructivePathError("owned child name must be one path component")
        child = self.root / name
        _reject_symlink_components(self.root, context="workspace path")
        child.mkdir(mode=0o700, parents=False, exist_ok=False)
        _write_new_json(
            child / CHILD_SENTINEL,
            {"schema": "dpo.owned-child/v1", "workspace_id": self.workspace_id},
        )
        return child

    def _validate_target(self, target: str | Path, protected_inputs: Iterable[str | Path]) -> Path:
        lexical = Path(target).expanduser().absolute()
        _reject_symlink_components(lexical, context="destructive target")
        resolved = lexical.resolve(strict=False)
        forbidden = {Path("/"), Path.home().resolve(), Path.cwd().resolve(), self.root}
        if resolved in forbidden:
            raise DestructivePathError(f"refusing destructive target {resolved}")
        try:
            resolved.relative_to(self.root)
        except ValueError as exc:
            raise DestructivePathError(
                "destructive target is outside the owned workspace or a symlink escape"
            ) from exc
        for protected in protected_inputs:
            protected_path = Path(protected).expanduser().resolve(strict=False)
            if (
                resolved == protected_path
                or protected_path in resolved.parents
                or resolved in protected_path.parents
            ):
                raise DestructivePathError("destructive target overlaps a protected input")
        sentinel = resolved / CHILD_SENTINEL
        if sentinel.is_symlink():
            raise DestructivePathError("destructive target sentinel cannot be a symlink")
        try:
            document = json.loads(sentinel.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DestructivePathError("destructive target is not sentinel-owned") from exc
        if document != {"schema": "dpo.owned-child/v1", "workspace_id": self.workspace_id}:
            raise DestructivePathError("destructive target sentinel belongs to another workspace")
        return resolved

    def safe_delete(
        self,
        target: str | Path,
        *,
        protected_inputs: Iterable[str | Path] = (),
    ) -> None:
        resolved = self._validate_target(target, protected_inputs)
        shutil.rmtree(resolved)

    def gc(self, lock_reachable_names: set[str], *, execute: bool = False) -> list[Path]:
        candidates: list[Path] = []
        for child in sorted(self.root.iterdir(), key=lambda item: item.name):
            if not child.is_dir() or child.name in lock_reachable_names:
                continue
            try:
                self._validate_target(child, ())
            except DestructivePathError:
                continue
            candidates.append(child)
        if execute:
            for candidate in candidates:
                self.safe_delete(candidate)
        return candidates


def validate_safetensors_adapter(
    adapter_dir: str | Path,
    expected_identity: Mapping[str, object],
) -> dict[str, Any]:
    """Validate identity and file types without importing torch or PEFT."""
    root = Path(adapter_dir)
    if not root.is_dir() or root.is_symlink():
        raise CheckpointSafetyError("adapter must be a real directory")
    if any(path.is_symlink() for path in root.rglob("*")):
        raise CheckpointSafetyError("adapter directory must not contain symlinks")
    unsafe = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in UNSAFE_CHECKPOINT_SUFFIXES
    )
    if unsafe:
        raise CheckpointSafetyError(f"legacy or unsafe checkpoint file is forbidden: {unsafe[0]}")
    weights = root / "adapter_model.safetensors"
    if not weights.is_file() or weights.is_symlink():
        raise CheckpointSafetyError("adapter_model.safetensors is required")
    manifest_path = root / "adapter_manifest.json"
    if manifest_path.is_symlink():
        raise CheckpointSafetyError("adapter manifest cannot be a symlink")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckpointSafetyError("adapter manifest is missing or invalid") from exc
    if not isinstance(document, dict):
        raise CheckpointSafetyError("adapter manifest must be an object")
    if semantic_hash(document) != semantic_hash(dict(expected_identity)):
        raise CheckpointSafetyError("adapter identity does not match the resolved contract")
    return document
