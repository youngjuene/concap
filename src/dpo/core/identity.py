from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path


class CanonicalEncodingError(ValueError):
    """Raised when a value has no unambiguous scientific encoding."""


def _canonicalize(value: object, location: str) -> object:
    if value is None or type(value) in {bool, int}:
        return value
    if type(value) is float:
        number = float(value)
        if not math.isfinite(number):
            raise CanonicalEncodingError(f"{location}: number must be finite")
        return 0.0 if number == 0 else number
    if isinstance(value, str):
        if unicodedata.normalize("NFC", value) != value:
            raise CanonicalEncodingError(f"{location}: string must use NFC normalization")
        if "${" in value:
            raise CanonicalEncodingError(f"{location}: unresolved value {value!r}")
        return value
    if isinstance(value, Path):
        raise CanonicalEncodingError(f"{location}: filesystem paths are not semantic values")
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise CanonicalEncodingError(f"{location}: mapping keys must be strings")
            result[key] = _canonicalize(value[key], f"{location}.{key}")
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item, f"{location}[{index}]") for index, item in enumerate(value)]
    raise CanonicalEncodingError(f"{location}: unsupported semantic type {type(value).__name__}")


def canonical_bytes(value: object) -> bytes:
    """Encode a semantic value independently of mapping order and runtime metadata."""
    canonical = _canonicalize(value, "value")
    return json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_hash(value: object) -> str:
    return f"sha256:{hashlib.sha256(canonical_bytes(value)).hexdigest()}"


def sha256_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def repo_lock_hash(start: Path | None = None) -> str:
    """Hash of the repository's uv.lock, found by upward search from this file.

    The one shared implementation of the environment-lock identity, so the CLI
    and the canary can never disagree about it; works from a source checkout
    at any package depth.
    """
    origin = start if start is not None else Path(__file__).resolve()
    for parent in [origin, *origin.parents]:
        candidate = parent / "uv.lock"
        if candidate.is_file():
            return sha256_file(candidate)
    raise FileNotFoundError("uv.lock is missing; the environment is not locked")
