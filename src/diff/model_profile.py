"""Pinned model profiles and the verification that runs before any remote code.

``trust_remote_code`` executes Dream's ``configuration_dream.py``,
``modeling_dream.py``, ``generation_utils.py`` and ``tokenization_dream.py``
from the model directory. A profile lists the SHA-256 of those files, the
tokenizer files, the configs and every weight shard at the pinned revision;
:func:`verify_snapshot` refuses to hand a directory to ``from_pretrained``
unless every one matches, and loading then happens offline from that verified
directory. After loading, :func:`verify_loaded_code` hashes the module files
Python actually imported, so a stale copy in the dynamic-module cache cannot
stand in for the verified code.

The text-file hashes were computed from files downloaded at the exact revision
(their git blob IDs matched the Hub's). The weight hashes are the LFS SHA-256
values the Hub API reports; they are re-checked locally whenever a snapshot is
verified with ``weights=True``.
"""

from __future__ import annotations

import hashlib
import inspect
import os
import platform
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from diff.study_protocol import (
    CONFIG_BOS_ID,
    CONFIG_MAX_POSITION_EMBEDDINGS,
    CONTEXT_LIMIT,
    ENDOFTEXT_ID,
    IM_END_ID,
    MASK_ID,
    TOKENIZER_BOS_ID,
)

DREAM_REVISION: Final = "05334cb9faaf763692dcf9d8737c642be2b2a6ae"
QWEN_REVISION: Final = "a09a35458c702b33eeacc393d103063234e8bc28"


def _shards(total: int, *hashes: str) -> dict[str, str]:
    """Weight shard names in the Hub's ``model-0000i-of-0000n`` pattern, with their LFS SHA-256."""
    if len(hashes) != total:
        raise ValueError(f"expected {total} shard hashes, got {len(hashes)}")
    return {f"model-{index:05d}-of-{total:05d}.safetensors": value for index, value in enumerate(hashes, 1)}


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    runner: str
    repo_id: str
    revision: str
    files: Mapping[str, str]
    weights: Mapping[str, str]
    remote_code: tuple[str, ...]
    tokenizer_files: tuple[str, ...]
    transformers_version: str
    token_ids: Mapping[str, int]
    context_limit: int
    facts: tuple[str, ...] = field(default=())

    def record(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "runner": self.runner,
            "repo_id": self.repo_id,
            "revision": self.revision,
            "files": dict(self.files),
            "weights": dict(self.weights),
            "remote_code": list(self.remote_code),
            "tokenizer_files": list(self.tokenizer_files),
            "transformers_version": self.transformers_version,
            "token_ids": dict(self.token_ids),
            "context_limit": self.context_limit,
            "dtype": "bfloat16",
            "facts": list(self.facts),
        }


DREAM_7B: Final = ModelProfile(
    profile_id="dream-v0-instruct-7b@05334cb9",
    runner="dream",
    repo_id="Dream-org/Dream-v0-Instruct-7B",
    revision=DREAM_REVISION,
    files={
        "configuration_dream.py": "24d038962ccf16361595494461000ec0c1b043653c7494d7fc835c96f121f0b9",
        "modeling_dream.py": "3166e789f0d69beb1f4fbdee2317953d1c1c3b9bc21479b109afc5dd059de5b3",
        "generation_utils.py": "7f8ad01484898946b3c9c9d5ebc85e5ac726061d0fa7b4a522c4b8110c13a921",
        "tokenization_dream.py": "1d7f04104fa673831f48256e3cd42cd0bbd1bddd801b4abea5c74cc47d98bdd2",
        "config.json": "11afc7fe7a1881e73dcd70cca6868aaaf8eb89ad45ea5fc6426c3fb82dec9340",
        "generation_config.json": "e4fa42a31ef22740804534395c7a18f335a3c1f9abd9db83a6941b1c8748b2ee",
        "tokenizer_config.json": "b0e9c1a6299d8832a1362d6b24f007de8a2acfe0243d8a3a1d1b57aa8250f7ee",
        "special_tokens_map.json": "fa14641fdc084244b0a530556b1d5bfcb9fa0ed9151a780ffec3b44ba96a0dfe",
        "added_tokens.json": "32ad9d49fafda64e737853737fe3be3e33cce3f719a159ed73f04cbd81a690ce",
        "vocab.json": "87a257b04b17642a0688c98cd1df89c398bda4fee532d6f88b38a659ecb4ac8d",
        "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
        "model.safetensors.index.json": "998a078123ffc97763690de7f2a677eb89168af5eaf8a5e12e6bc24d18e25bdb",
    },
    weights=_shards(
        4,
        "3014d84950f2f3afcce622508b6625e9cd911b9c340f4206e35e6a3177aec993",
        "78fc44a7eae29c3d13bba010ca12584f4220798d41c9e617579a33c47222aa57",
        "eac5afaf9a0e0b4a23c7ac1dbe5f538ccdf9fac5c0db7919fd9625a92c378e48",
        "398da65f1f599ca8d594a9d7a077a3b7c4eb0ddfdff8e7e004c3d259712b3cb7",
    ),
    remote_code=(
        "configuration_dream.py",
        "modeling_dream.py",
        "generation_utils.py",
        "tokenization_dream.py",
    ),
    tokenizer_files=(
        "tokenization_dream.py",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
    ),
    transformers_version="4.46.2",
    token_ids={
        "mask": MASK_ID,
        "endoftext": ENDOFTEXT_ID,
        "im_end": IM_END_ID,
        "config_bos": CONFIG_BOS_ID,
        "tokenizer_bos": TOKENIZER_BOS_ID,
    },
    context_limit=CONTEXT_LIMIT,
    facts=(
        "MC-08: text-only; evidence text is the only grounding",
        "config.json max_position_embeddings 131,072 against the README's 2,048 context; 2,048 is used",
        "loaded with AutoModel (DreamModel), bfloat16, trust_remote_code=True;"
        " the tokenizer is remote code too",
        "without a generation_config the base config comes from config.json, whose legacy"
        " generation defaults (transformers 4.46.2) override Dream's class defaults:"
        " every call keyword is therefore required",
        "the shipped generation_config.json has mask_token_id null, so no generation_config object is passed",
        "generation_config.update drops unknown keys silently,"
        " so any other keyword is refused before the call",
        "the canvas starts as all mask; every step is one full forward with no cache and no early exit",
        "each step commits int(n_mask x (1 - s/t)) positions on linear timesteps from 1 to eps,"
        " all the rest at the last step",
        "predictions are not barred from the mask ID, so a selected position can stay masked",
        "the mask search covers the prompt, so a prompt containing 151666 would be generated into",
    ),
)

QWEN_7B: Final = ModelProfile(
    profile_id="qwen2.5-7b-instruct@a09a3545",
    runner="qwen",
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    revision=QWEN_REVISION,
    files={
        "config.json": "7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c",
        "generation_config.json": "3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f",
        "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
        "tokenizer_config.json": "5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583",
        "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
        "merges.txt": "599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3",
        "model.safetensors.index.json": "624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028",
    },
    weights=_shards(
        4,
        "a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7",
        "f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185",
        "8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5",
        "1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd",
    ),
    remote_code=(),
    tokenizer_files=("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"),
    transformers_version="4.46.2",
    token_ids={"endoftext": ENDOFTEXT_ID, "im_end": IM_END_ID, "config_bos": CONFIG_BOS_ID},
    context_limit=CONTEXT_LIMIT,
    facts=(
        "Qwen2ForCausalLM, no remote code; Dream was initialized from Qwen2.5-7B and uses its tokenizer plus "
        "added tokens, but the comparison supports no architecture claim",
        "generation_config.json turns on sampling (temperature 0.7, top_p 0.8, top_k 20) and "
        "repetition_penalty 1.05, which would still apply under greedy decoding, so repetition_penalty=1.0 "
        "is passed and temperature/top_p/top_k are left unset",
        "the chat template's default system line adds an identity,"
        " so the system message is passed explicitly",
        "the same 2,048-token context limit as Dream keeps the two models' inputs comparable",
    ),
)

PROFILES: Final[dict[str, ModelProfile]] = {item.profile_id: item for item in (DREAM_7B, QWEN_7B)}


def profile(profile_id: str) -> ModelProfile:
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown model profile {profile_id!r}; known: {sorted(PROFILES)}")
    return PROFILES[profile_id]


class ProfileMismatch(RuntimeError):
    """A model directory that does not match its pinned profile, file by file."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_snapshot(directory: Path, model: ModelProfile, *, weights: bool = True) -> dict[str, Any]:
    """Hash every pinned file in ``directory``; raise unless all of them match.

    Returns the verification record attempts carry. Weight shards also record
    size and mtime so a worker can confirm, without re-hashing 15 GB, that they
    have not changed since this verification.
    """
    problems: list[str] = []
    files: dict[str, str] = {}
    for name, expected in sorted(model.files.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        files[name] = actual = file_sha256(path)
        if actual != expected:
            problems.append(f"{name}: sha256 {actual} != pinned {expected}")
    shards: dict[str, dict[str, Any]] = {}
    for name, expected in sorted(model.weights.items()):
        path = directory / name
        if not path.is_file():
            problems.append(f"{name}: missing")
            continue
        stat = path.stat()
        entry: dict[str, Any] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": None}
        if weights:
            entry["sha256"] = actual = file_sha256(path)
            if actual != expected:
                problems.append(f"{name}: sha256 {actual} != pinned {expected}")
        shards[name] = entry
    if problems:
        raise ProfileMismatch(problems)
    return {
        "profile_id": model.profile_id,
        "revision": model.revision,
        "directory": str(directory.resolve()),
        "files": files,
        "weights": shards,
        "weights_hashed": weights,
        "verified_at": time.time(),
    }


def assert_weights_unchanged(directory: Path, record: Mapping[str, Any]) -> None:
    """Refuse weights whose size or mtime moved since ``record`` hashed them."""
    if not record.get("weights_hashed"):
        raise ProfileMismatch(["weights were never hashed for this verification record"])
    problems = []
    for name, entry in record["weights"].items():
        stat = (directory / name).stat()
        if (stat.st_size, stat.st_mtime_ns) != (entry["size"], entry["mtime_ns"]):
            problems.append(f"{name}: changed after verification")
    if problems:
        raise ProfileMismatch(problems)


def verify_loaded_code(objects: list[Any], model: ModelProfile) -> dict[str, str]:
    """Hash every remote-code file the loaded objects' classes were imported from.

    ``objects`` are the loaded model, its config and its tokenizer; every class
    in their method resolution order that came from a remote-code module is
    checked, so ``generation_utils.py`` (the mixin) is covered as well as
    ``modeling_dream.py``.
    """
    if not model.remote_code:
        return {}
    imported: dict[str, Path] = {}
    for item in objects:
        for cls in type(item).__mro__:
            try:
                path = Path(inspect.getfile(cls))
            except (TypeError, OSError):
                continue
            if path.name in model.remote_code:
                imported[path.name] = path
    problems, hashes = [], {}
    for name in model.remote_code:
        if name not in imported:
            problems.append(f"{name}: no loaded class came from it")
            continue
        hashes[name] = actual = file_sha256(imported[name])
        if actual != model.files[name]:
            problems.append(f"{name}: imported copy sha256 {actual} != pinned {model.files[name]}")
    if problems:
        raise ProfileMismatch(problems)
    return hashes


def environment_fingerprint(lock_path: Path | None = None) -> dict[str, Any]:
    """What an attempt ran on. Imports torch/transformers only if already importable."""
    result: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "lock_sha256": file_sha256(lock_path) if lock_path and lock_path.is_file() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "config_max_position_embeddings": CONFIG_MAX_POSITION_EMBEDDINGS,
    }
    try:
        import torch
        import transformers
    except ImportError:
        return result
    result.update(
        torch=str(torch.__version__),
        transformers=str(transformers.__version__),
        cuda=str(torch.version.cuda),
        deterministic_algorithms=bool(torch.are_deterministic_algorithms_enabled()),
    )
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        result.update(gpu=torch.cuda.get_device_name(index), gpu_index=index)
    return result
