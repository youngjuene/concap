"""Qwen2.5-7B-Instruct, the autoregressive baseline, through transformers' ``generate``.

Review stage only by default (plan §12), reported beside Trida-7B. The pinned
``generation_config.json`` turns on sampling
(temperature 0.7, top_p 0.8, top_k 20) and sets ``repetition_penalty=1.05``,
which would still apply under greedy decoding. So the call passes
``do_sample=False`` and ``repetition_penalty=1.0``, leaves temperature, top_p and
top_k unset, stops at either terminator, and records the resolved config.
Nothing of Trida's contract is ever passed here, and nothing else is accepted.

``max_new_tokens`` is Trida's canvas of the same phase, so both models get the
same room; the output may stop earlier.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from diff.job_plan import FrozenJob, prompt_guard_problems
from diff.model_profile import QWEN_7B, assert_weights_unchanged, environment_fingerprint, verify_snapshot
from diff.model_runtime import (
    LoadedModel,
    RefusedCall,
    chat_token_ids,
    checked_token_ids,
    rng_snapshot,
    rng_unchanged,
)
from diff.output_checks import extract_output
from diff.study_protocol import QWEN_ENDOFTEXT_ID, QWEN_IM_END_ID, QWEN_TERMINATOR_IDS, sha256_tokens

REQUIRED_KEYWORDS: Final = (
    "attention_mask",
    "max_new_tokens",
    "do_sample",
    "repetition_penalty",
    "eos_token_id",
)
RESOLVED_FIELDS: Final = (
    "do_sample",
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
    "max_new_tokens",
    "eos_token_id",
    "pad_token_id",
)
TOKEN_NAMES: Final = {"endoftext": "<|endoftext|>", "im_end": "<|im_end|>"}


def checked_call(arguments: Mapping[str, Any]) -> dict[str, Any]:
    problems = [
        f"missing required keyword {keyword!r}" for keyword in REQUIRED_KEYWORDS if keyword not in arguments
    ]
    extra = sorted(set(arguments) - set(REQUIRED_KEYWORDS))
    if extra:
        problems.append(f"refused keywords {extra}")
    if "do_sample" in arguments and arguments["do_sample"] is not False:
        problems.append("do_sample must be False")
    if "repetition_penalty" in arguments and (
        isinstance(arguments["repetition_penalty"], bool) or arguments["repetition_penalty"] != 1.0
    ):
        problems.append(f"repetition_penalty must be 1.0, got {arguments['repetition_penalty']!r}")
    if "eos_token_id" in arguments and list(arguments["eos_token_id"]) != [QWEN_IM_END_ID, QWEN_ENDOFTEXT_ID]:
        problems.append(f"eos_token_id must be [{QWEN_IM_END_ID}, {QWEN_ENDOFTEXT_ID}]")
    limit = arguments.get("max_new_tokens")
    if "max_new_tokens" in arguments and (type(limit) is not int or limit < 1):
        problems.append(f"max_new_tokens must be a positive integer, got {limit!r}")
    if problems:
        raise RefusedCall("; ".join(problems))
    return dict(arguments)


def load(
    model_dir: Path, verification: Mapping[str, Any], *, device: str, lock_path: Path | None = None
) -> LoadedModel:
    verify_snapshot(model_dir, QWEN_7B, weights=False)
    assert_weights_unchanged(model_dir, verification)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, torch_dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        profile=QWEN_7B,
        environment=environment_fingerprint(lock_path),
        token_ids=checked_token_ids(tokenizer, QWEN_7B.token_ids, TOKEN_NAMES),
    )


def resolved_config(model: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    call = {key: value for key, value in arguments.items() if key != "attention_mask"}
    prepared = model._prepare_generation_config(None, **call)
    config = prepared[0] if isinstance(prepared, tuple) else prepared
    return {key: getattr(config, key, None) for key in RESOLVED_FIELDS}


def run_job(loaded: LoadedModel, frozen: FrozenJob) -> dict[str, Any]:
    import torch

    job = frozen.job
    ids = chat_token_ids(loaded.tokenizer, job.prompt)
    if sha256_tokens(ids) != frozen.token_ids_sha256 or len(ids) != frozen.token_count:
        return {"status": "refused", "error": "prompt token IDs differ from the frozen plan"}
    guards = prompt_guard_problems(job, ids)
    if guards:
        return {"status": "refused", "error": "; ".join(guards)}
    input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    arguments = checked_call({**job.generate, "attention_mask": torch.ones_like(input_ids)})
    resolved = resolved_config(loaded.model, arguments)
    before = rng_snapshot(loaded.device)
    started = time.monotonic()
    output_ids = loaded.model.generate(input_ids, **arguments)
    latency_ms = round((time.monotonic() - started) * 1000)
    unchanged = rng_unchanged(before, rng_snapshot(loaded.device))
    sequence = [int(token) for token in output_ids[0].tolist()]
    output = extract_output(sequence, len(ids), loaded.decode, loaded.special_ids, QWEN_TERMINATOR_IDS)
    status, error = ("completed", None) if unchanged else ("rng_consumed", "the call changed an RNG state")
    return {
        "status": status,
        "error": error,
        "prompt_tokens": len(ids),
        "output": output.record(),
        "output_token_ids_sha256": sha256_tokens(sequence),
        "resolved_config": resolved,
        "rng_unchanged": unchanged,
        "latency_ms": latency_ms,
        "trace": {"events": [], "level": "none", "summary": {}, "problems": []},
    }


def warm_up(loaded: LoadedModel) -> dict[str, Any]:
    return {"wall_ms": 0}
