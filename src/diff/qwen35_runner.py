"""Qwen3.5-9B, phase 1's second-caption writer since 2026-10-26, through transformers' ``generate``.

The researcher replaced Trida with Qwen3.5-9B for rewriting Korean captions, phase 1 first
(plan "Second captions: responsive, written live"). Like the Qwen2.5 baseline it is greedy:
the call passes ``do_sample=False`` and ``repetition_penalty=1.0``, leaves temperature, top_p and
top_k unset, and stops at either terminator. The revision ships no generation config, so
nothing hidden applies. The chat template runs with the profile's options (thinking off).

The checkpoint is a vision-language model (``Qwen3_5ForConditionalGeneration``); only text is
passed. It needs transformers 5, so this runner runs in the repository environment, not the
generator environment, whose transformers 4.57.3 Trida's remote code pins. Live (tweak) jobs are
frozen inside the worker, as Trida's are.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

from diff.job_plan import FrozenJob, Job, PlanError, prompt_guard_problems
from diff.model_profile import QWEN35_9B, assert_weights_unchanged, environment_fingerprint, verify_snapshot
from diff.model_runtime import (
    LoadedModel,
    RefusedCall,
    chat_token_ids,
    checked_token_ids,
    rng_snapshot,
    rng_unchanged,
)
from diff.output_checks import extract_output
from diff.study_protocol import QWEN35_ENDOFTEXT_ID, QWEN35_IM_END_ID, QWEN35_TERMINATOR_IDS, sha256_tokens

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
    if "eos_token_id" in arguments and list(arguments["eos_token_id"]) != [
        QWEN35_IM_END_ID,
        QWEN35_ENDOFTEXT_ID,
    ]:
        problems.append(f"eos_token_id must be [{QWEN35_IM_END_ID}, {QWEN35_ENDOFTEXT_ID}]")
    limit = arguments.get("max_new_tokens")
    if "max_new_tokens" in arguments and (type(limit) is not int or limit < 1):
        problems.append(f"max_new_tokens must be a positive integer, got {limit!r}")
    if problems:
        raise RefusedCall("; ".join(problems))
    return dict(arguments)


def load(
    model_dir: Path, verification: Mapping[str, Any], *, device: str, lock_path: Path | None = None
) -> LoadedModel:
    verify_snapshot(model_dir, QWEN35_9B, weights=False)
    assert_weights_unchanged(model_dir, verification)
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model: Any = AutoModelForImageTextToText.from_pretrained(
        str(model_dir), local_files_only=True, dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        profile=QWEN35_9B,
        environment=environment_fingerprint(lock_path),
        token_ids=checked_token_ids(tokenizer, QWEN35_9B.token_ids, TOKEN_NAMES),
    )


def token_ids(loaded: LoadedModel, job: Job) -> list[int]:
    return chat_token_ids(loaded.tokenizer, job.prompt, loaded.profile.chat_options)


def freeze_live(loaded: LoadedModel, job: Job) -> FrozenJob:
    """A live (tweak) job frozen inside the worker: the pinned template with thinking off, the guards."""
    ids = token_ids(loaded, job)
    problems = prompt_guard_problems(job, ids)
    if problems:
        raise PlanError(problems)
    return FrozenJob(job, sha256_tokens(ids), len(ids))


def resolved_config(model: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    call = {key: value for key, value in arguments.items() if key != "attention_mask"}
    prepared = model._prepare_generation_config(None, **call)
    config = prepared[0] if isinstance(prepared, tuple) else prepared
    return {key: getattr(config, key, None) for key in RESOLVED_FIELDS}


def run_job(loaded: LoadedModel, frozen: FrozenJob) -> dict[str, Any]:
    import torch

    job = frozen.job
    ids = token_ids(loaded, job)
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
    with torch.no_grad():
        output_ids = loaded.model.generate(input_ids=input_ids, **arguments)
    latency_ms = round((time.monotonic() - started) * 1000)
    unchanged = rng_unchanged(before, rng_snapshot(loaded.device))
    sequence = [int(token) for token in output_ids[0].tolist()]
    output = extract_output(sequence, len(ids), loaded.decode, loaded.special_ids, QWEN35_TERMINATOR_IDS)
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
    """One short greedy call, so the first participant's caption does not pay for kernel set-up."""
    import torch

    started = time.monotonic()
    ids = chat_token_ids(loaded.tokenizer, _WARM_UP, loaded.profile.chat_options)
    input_ids = torch.tensor([ids], dtype=torch.long, device=loaded.device)
    with torch.no_grad():
        loaded.model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=4,
            do_sample=False,
            repetition_penalty=1.0,
            eos_token_id=[QWEN35_IM_END_ID, QWEN35_ENDOFTEXT_ID],
        )
    return {"wall_ms": round((time.monotonic() - started) * 1000)}


def _warm_up_prompt() -> Any:
    from diff.rewrite_prompts import RenderedPrompt

    return RenderedPrompt("warm-up", 0, "", 0, None, "안녕하세요.")


_WARM_UP: Final = _warm_up_prompt()
