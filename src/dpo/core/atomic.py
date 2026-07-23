from __future__ import annotations

import os
import uuid
from contextlib import suppress
from pathlib import Path


class AtomicWriteConflict(ValueError):
    """Raised when an immutable destination already contains different bytes."""


def atomic_write_bytes(path: str | Path, payload: bytes) -> bool:
    """Publish immutable bytes with an idempotent same-content fast path."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.is_file() and destination.read_bytes() == payload:
            return False
        raise AtomicWriteConflict(f"immutable destination already exists: {destination}")
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            if destination.is_file() and destination.read_bytes() == payload:
                return False
            raise AtomicWriteConflict(f"immutable destination won a conflicting race: {destination}") from exc
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        return True
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()
