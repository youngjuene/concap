"""Pinned model profiles and the verification that runs before any remote code.

``trust_remote_code`` executes Trida's ``configuration_trida.py`` and
``modeling_trida.py`` from the model directory. A profile lists the SHA-256 of those files, the
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
    QWEN35_CONTEXT_LIMIT,
    QWEN35_ENDOFTEXT_ID,
    QWEN35_IM_END_ID,
    QWEN_CONFIG_BOS_ID,
    QWEN_CONTEXT_LIMIT,
    QWEN_ENDOFTEXT_ID,
    QWEN_IM_END_ID,
    TRIDA_BOS_ID,
    TRIDA_CONTEXT_LIMIT,
    TRIDA_IM_END_ID,
    TRIDA_IM_START_ID,
    TRIDA_MASK_ID,
    TRIDA_PAD_ID,
)

QWEN_REVISION: Final = "a09a35458c702b33eeacc393d103063234e8bc28"
QWEN35_REVISION: Final = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TRIDA_REVISION: Final = "c37755e81222de0aff69ee481f5299a99928b105"


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
    # Keywords for the chat template (Qwen3.5: thinking off), part of the pinned rendering.
    chat_options: Mapping[str, Any] = field(default_factory=dict)

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
            "chat_options": dict(self.chat_options),
        }


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
    transformers_version="4.57.3",
    token_ids={"endoftext": QWEN_ENDOFTEXT_ID, "im_end": QWEN_IM_END_ID, "config_bos": QWEN_CONFIG_BOS_ID},
    context_limit=QWEN_CONTEXT_LIMIT,
    facts=(
        "Qwen2ForCausalLM, no remote code;"
        " the baseline beside Trida, never a participant condition by default",
        "generation_config.json turns on sampling (temperature 0.7, top_p 0.8, top_k 20) and "
        "repetition_penalty 1.05, which would still apply under greedy decoding, so repetition_penalty=1.0 "
        "is passed and temperature/top_p/top_k are left unset",
        "the chat template's default system line adds an identity,"
        " so the system message is passed explicitly",
        "a conservative 2,048-token prompt-plus-output limit, far above any prompt here",
    ),
)

# Phase 1's second-caption writer since 2026-10-26 (plan "Second captions: responsive, written live"):
# the researcher's replacement for Trida in Korean caption rewriting. Every hash was computed from the
# files downloaded at the revision and matched the Hub (git blob IDs, and LFS SHA-256 for the shards and
# tokenizer.json).
QWEN35_9B: Final = ModelProfile(
    profile_id="qwen3.5-9b@c2022362",
    runner="qwen35",
    repo_id="Qwen/Qwen3.5-9B",
    revision=QWEN35_REVISION,
    files={
        "config.json": "d0883072e01861ed0b2d47be3c16c36a8e81c224c7ffaa310c6558fb3f932b05",
        "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
        "tokenizer.json": "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
        "tokenizer_config.json": "316230d6a809701f4db5ea8f8fc862bc3a6f3229c937c174e674ff3ca0a64ac8",
        "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
        "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
        "model.safetensors.index.json": "26d3539b516be613f39563617cb9d33b3f83d401298125be392c80cefb8f7fe5",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "video_preprocessor_config.json": "7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13",
    },
    weights={
        f"model.safetensors-{index:05d}-of-00004.safetensors": value
        for index, value in enumerate(
            (
                "db6f444b43d318c92f360a13a25561a6a65b10c0631b8ed305a426dbaa6c380e",
                "31c7d7e2dd5d207840b31cc59083c8f4c4718959149e0358c0364052bb9a0330",
                "7ec36ba3a4176a44c3c0876ad80c56a2f70c84bf008d82e9501df642f17dadec",
                "b62b0c4cd7e44edee103ee8f4fe225f246d5e768e07bfd5f25b63a8aa1fdd0c6",
            ),
            1,
        )
    },
    remote_code=(),
    tokenizer_files=(
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "chat_template.jinja",
    ),
    transformers_version="5.13.1",
    token_ids={"endoftext": QWEN35_ENDOFTEXT_ID, "im_end": QWEN35_IM_END_ID},
    context_limit=QWEN35_CONTEXT_LIMIT,
    facts=(
        "Qwen3_5ForConditionalGeneration (text and vision; only text is passed), no remote code; it needs "
        "transformers 5, so it runs in the repository environment, not the generator environment Trida's "
        "code pins at 4.57.3",
        "no generation_config.json at the revision: greedy is passed explicitly, with repetition_penalty 1.0",
        "thinking is off through the chat template (enable_thinking=False),"
        " which writes an empty think block",
        "chosen on the 2026-10-26 delivery bench over 2B and 4B"
        " (artifacts/diff/bench/candidates-20261026-live)",
    ),
    chat_options={"enable_thinking": False},
)

# The researcher's diffusion model since 2026-10-12 (plan "Model change — 2026-10-12").
# Text-file hashes match plan §2.3; weight hashes are the Hub's LFS SHA-256 at the revision.
TRIDA_7B: Final = ModelProfile(
    profile_id="trida-7b@c37755e8",
    runner="trida",
    repo_id="trillionlabs/Trida-7B",
    revision=TRIDA_REVISION,
    files={
        "configuration_trida.py": "105a4722bc7fd3ed5f3b67e18cda91aae404135835f9e31037435d8c2f78ffe1",
        "modeling_trida.py": "100558ff8e0e7d37e24bb8a3cdfcea508ed9db5c81eeb64a026ba76f4b8ec949",
        "config.json": "ed7d5a45a81e50883e3a67651b3b8bd999715221238770011837744a893566e5",
        "generation_config.json": "12603b65aab9b66affeafb91cbc5c0b3afe4e0c40c71492bce9aa564024efb86",
        "tokenizer.json": "87522bb349f1b34dc65b11fbc841cf6d6499da80c375f142ea4c4a7923011321",
        "tokenizer_config.json": "e9477f4ef3120bc2b001c7b482207690c84cfc4a79a5ad557b114a7549769190",
        "special_tokens_map.json": "ea3dfcfe26148c30aa296b54083a66531a28c58a0f880625a59439fe1d67baa7",
        "chat_template.jinja": "8fe304e54dbef96520a19f0640ddf421c3a2188cf85352b09789c28f4c788238",
        "model.safetensors.index.json": "0ecd88d572d5d263671d0a725369538b5d598e88c6523dc3ddee8f6becb3f461",
    },
    weights=_shards(
        4,
        "e5e9a4589f938a297ead260acedab905c6dc4dc84e190628c58bdc70a560192f",
        "624ad919451158c5bc7229c6346dca79631abfedcd112f9bf128447ca92125c8",
        "ae69bf99459c60fb06138805a4ed8cd8e4ad58a76e2d61220a15634947d23858",
        "5f0a7f453efe5cea64bc17ef717b4177030167f0d724da0d8bda1fd2f89d4a29",
    ),
    remote_code=("configuration_trida.py", "modeling_trida.py"),
    tokenizer_files=(
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ),
    transformers_version="4.57.3",
    token_ids={
        "mask": TRIDA_MASK_ID,
        "im_start": TRIDA_IM_START_ID,
        "im_end": TRIDA_IM_END_ID,
        "pad": TRIDA_PAD_ID,
        "bos": TRIDA_BOS_ID,
    },
    context_limit=TRIDA_CONTEXT_LIMIT,
    facts=(
        "text-only; the evidence or starting caption is the only grounding",
        "TridaForDLM, remote code, bfloat16, loaded with AutoModelForCausalLM;"
        " the tokenizer is not remote code",
        "generate decodes num_blocks = max_new_tokens // 32 blocks of 32 in small blocks of 8,"
        " committing every masked position above the threshold and at least the most confident one per step",
        "temperature 0 takes the argmax and never applies top_p, so a call consumes no randomness",
        "generation stops at <|im_end|> (128001) and pads after it with <|pad|> (128004)",
        "imports einops and transformers modules added after 4.46; saved with transformers 4.57.3",
        "the chat template inserts 'You are Trillion, created by TrillionLabs…'"
        " when no system message is given",
        "transformers 4.57.3 warns about a Mistral regex on loading the tokenizer; its fix cannot apply to"
        " Trida's ByteLevel pre-tokenizer, so the warning is a false positive",
    ),
)

PROFILES: Final[dict[str, ModelProfile]] = {item.profile_id: item for item in (TRIDA_7B, QWEN_7B, QWEN35_9B)}


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
    checked, so a mixin in its own file is covered as well as the model class.
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
