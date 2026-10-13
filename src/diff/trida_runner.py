"""Trida-7B through its native block-diffusion ``generate`` (plan §4.4).

The researcher's model since 2026-10-12, replacing Dream. The call is exactly
the pinned remote code's: batch 1, no attention mask, ``threshold`` τ from the
protocol's conditions, 32-token blocks decoded in small blocks of 8,
``temperature=0`` (argmax; ``top_p`` is then never applied, so it is not
passed), no block cache, and the pinned mask, stop and pad IDs.
:func:`checked_call` refuses any other keyword and any other value before the
model is touched.

What the pinned ``generate`` does, and what the guards answer:

* ``num_blocks = max_new_tokens // 32``, so the canvas must be a whole number
  of blocks; a value below 32 would decode nothing.
* A prompt longer than 32 tokens is prefilled; one of exactly 32 tokens is not,
  and the first block would then be decoded without seeing the prompt. That
  length is refused.
* Every step commits each masked position above τ and at least the most
  confident one. Generation stops at ``<|im_end|>`` (128001) and pads after it.
* At temperature 0 nothing is sampled, so a call must leave the CPU and CUDA RNG
  states unchanged (AC-07); a call that does not is a technical failure.

Forward calls vary with τ, so they are recorded, not capped. The output is cut
at the first ``<|im_end|>`` by :func:`diff.output_checks.extract_output`; the
tail is kept and nothing is ever trimmed.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from diff.job_plan import FrozenJob, Job, PlanError, prompt_guard_problems
from diff.model_profile import (
    TRIDA_7B,
    assert_weights_unchanged,
    environment_fingerprint,
    verify_loaded_code,
    verify_snapshot,
)
from diff.model_runtime import (
    LoadedModel,
    RefusedCall,
    chat_token_ids,
    checked_token_ids,
    rng_snapshot,
    rng_unchanged,
)
from diff.output_checks import extract_output
from diff.study_protocol import (
    SYSTEM_MESSAGE,
    TAU_CONDITIONS,
    TRIDA_BLOCK_SIZE,
    TRIDA_CONTEXT_LIMIT,
    TRIDA_IM_END_ID,
    TRIDA_MASK_ID,
    TRIDA_PAD_ID,
    TRIDA_SMALL_BLOCK_SIZE,
    TRIDA_TEMPERATURE,
    TRIDA_TERMINATOR_IDS,
    TRIDA_USE_BLOCK_CACHE,
    sha256_tokens,
)

REQUIRED_KEYWORDS: Final = (
    "max_new_tokens",
    "threshold",
    "block_size",
    "small_block_size",
    "temperature",
    "use_block_cache",
    "mask_id",
    "stop_token",
    "pad_token_id",
)
FIXED_VALUES: Final[dict[str, Any]] = {
    "block_size": TRIDA_BLOCK_SIZE,
    "small_block_size": TRIDA_SMALL_BLOCK_SIZE,
    "use_block_cache": TRIDA_USE_BLOCK_CACHE,
    "mask_id": TRIDA_MASK_ID,
    "stop_token": TRIDA_IM_END_ID,
    "pad_token_id": TRIDA_PAD_ID,
}
TOKEN_NAMES: Final = {
    "mask": "<|mask|>",
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "pad": "<|pad|>",
    "bos": "<|endoftext|>",
}


def _same(value: Any, wanted: Any) -> bool:
    """Equal and of the same kind (``True`` is not 1, ``0.0`` is not ``False``)."""
    return type(value) is type(wanted) and value == wanted


def checked_call(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Exactly the contract's keywords and values, or :class:`RefusedCall` naming every problem."""
    problems = []
    for keyword in REQUIRED_KEYWORDS:
        if keyword not in arguments:
            problems.append(f"missing required keyword {keyword!r}")
    extra = sorted(set(arguments) - set(REQUIRED_KEYWORDS))
    if extra:
        problems.append(f"refused keywords {extra}: the contract passes nothing else")
    for keyword, wanted in FIXED_VALUES.items():
        if keyword in arguments and not _same(arguments[keyword], wanted):
            problems.append(f"{keyword} must be {wanted!r}, got {arguments[keyword]!r}")
    temperature = arguments.get("temperature")
    if "temperature" in arguments and (isinstance(temperature, bool) or temperature != TRIDA_TEMPERATURE):
        problems.append(f"temperature must be {TRIDA_TEMPERATURE}, got {temperature!r}")
    threshold = arguments.get("threshold")
    if "threshold" in arguments and (
        isinstance(threshold, bool) or threshold not in set(TAU_CONDITIONS.values())
    ):
        problems.append(f"threshold must be one of {sorted(TAU_CONDITIONS.values())}, got {threshold!r}")
    canvas = arguments.get("max_new_tokens")
    if "max_new_tokens" in arguments and (
        type(canvas) is not int or canvas < TRIDA_BLOCK_SIZE or canvas % TRIDA_BLOCK_SIZE
    ):
        problems.append(
            f"max_new_tokens must be a whole number of {TRIDA_BLOCK_SIZE}-token blocks, got {canvas!r}"
        )
    if problems:
        raise RefusedCall("; ".join(problems))
    return dict(arguments)


def context_problems(prompt_ids: Sequence[int], canvas: int) -> list[str]:
    """Prompts the pinned ``generate`` would mishandle (AC-05)."""
    problems = []
    if len(prompt_ids) + canvas + 1 > TRIDA_CONTEXT_LIMIT:
        problems.append(
            f"{len(prompt_ids)} prompt tokens + a {canvas}-token canvas + 1 exceed the "
            f"{TRIDA_CONTEXT_LIMIT}-token context"
        )
    if len(prompt_ids) == TRIDA_BLOCK_SIZE:
        problems.append(f"a prompt of exactly {TRIDA_BLOCK_SIZE} tokens is not prefilled and would go unseen")
    for name, token in (("mask", TRIDA_MASK_ID), ("pad", TRIDA_PAD_ID)):
        if token in prompt_ids:
            problems.append(f"the prompt contains the {name} token {token}")
    return problems


def load(
    model_dir: Path, verification: Mapping[str, Any], *, device: str, lock_path: Path | None = None
) -> LoadedModel:
    """Re-verify the small files, confirm the weights are the verified ones, then load offline."""
    verify_snapshot(model_dir, TRIDA_7B, weights=False)
    assert_weights_unchanged(model_dir, verification)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model: Any = AutoModelForCausalLM.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        profile=TRIDA_7B,
        loaded_code=verify_loaded_code([model, model.config], TRIDA_7B),
        environment=environment_fingerprint(lock_path),
        token_ids=checked_token_ids(tokenizer, TRIDA_7B.token_ids, TOKEN_NAMES),
    )


def call_with_caps(
    loaded: LoadedModel, prompt_ids: Sequence[int], generate: Mapping[str, Any]
) -> dict[str, Any]:
    """One generate call with its forward count and RNG check; raises RefusedCall before the call."""
    import torch

    canvas = int(generate["max_new_tokens"])
    refusals = context_problems(prompt_ids, canvas)
    if refusals:
        raise RefusedCall("; ".join(refusals))
    arguments = checked_call(generate)
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=loaded.device)
    forward_calls: list[int] = []

    def count_forward(module: Any, args: Any, output: Any) -> None:
        forward_calls.append(1)

    handle = loaded.model.register_forward_hook(count_forward)
    before = rng_snapshot(loaded.device)
    started = time.monotonic()
    try:
        with torch.no_grad():
            output = loaded.model.generate(input_ids, **arguments)
    finally:
        handle.remove()
    latency_ms = round((time.monotonic() - started) * 1000)
    sequences = output if isinstance(output, torch.Tensor) else output.sequences
    return {
        "sequence": [int(token) for token in sequences[0].tolist()],
        "forward_calls": len(forward_calls),
        "rng_unchanged": rng_unchanged(before, rng_snapshot(loaded.device)),
        "latency_ms": latency_ms,
    }


def warm_up(loaded: LoadedModel) -> dict[str, Any]:
    """One tiny untimed call so CUDA initialisation is not charged to the first job's deadline.

    The prompt carries the protocol's system message and avoids the one prompt
    length the guards refuse (exactly one block).
    """
    for text in ("안녕하세요.", "안녕하세요, 반갑습니다."):
        rendered = loaded.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_MESSAGE}, {"role": "user", "content": text}],
            add_generation_prompt=True,
            tokenize=True,
        )
        if len(rendered) != TRIDA_BLOCK_SIZE:
            break
    generate = {
        "max_new_tokens": TRIDA_BLOCK_SIZE,
        "threshold": TAU_CONDITIONS["trida_tau_090"],
        "temperature": TRIDA_TEMPERATURE,
        **FIXED_VALUES,
    }
    started = time.monotonic()
    call = call_with_caps(loaded, [int(token) for token in rendered], generate)
    return {"wall_ms": round((time.monotonic() - started) * 1000), "rng_unchanged": call["rng_unchanged"]}


def freeze_live(loaded: LoadedModel, job: Job) -> FrozenJob:
    """A live (phase-2) job frozen inside the worker: Trida's own chat template, the same guards."""
    ids = chat_token_ids(loaded.tokenizer, job.prompt)
    problems = prompt_guard_problems(job, ids)
    if problems:
        raise PlanError(problems)
    return FrozenJob(job, sha256_tokens(ids), len(ids))


def run_job(loaded: LoadedModel, frozen: FrozenJob) -> dict[str, Any]:
    """One attempt's model-side result: status, output, forward count and RNG check."""
    job = frozen.job
    ids = chat_token_ids(loaded.tokenizer, job.prompt)
    if sha256_tokens(ids) != frozen.token_ids_sha256 or len(ids) != frozen.token_count:
        return {"status": "refused", "error": "prompt token IDs differ from the frozen plan"}
    guards = prompt_guard_problems(job, ids)
    if guards:
        return {"status": "refused", "error": "; ".join(guards)}
    call = call_with_caps(loaded, ids, job.generate)
    output = extract_output(
        call["sequence"],
        len(ids),
        loaded.decode,
        loaded.special_ids,
        TRIDA_TERMINATOR_IDS,
        mask_id=TRIDA_MASK_ID,
    )
    status, problems = "completed", []
    if not call["rng_unchanged"]:
        status, problems = "rng_consumed", ["the call changed the CPU or CUDA RNG state"]
    elif output.leftover_mask_count:
        status, problems = "leftover_mask", [f"{output.leftover_mask_count} mask tokens left before the stop"]
    return {
        "status": status,
        "error": "; ".join(problems) or None,
        "prompt_tokens": len(ids),
        "output": output.record(),
        "output_token_ids_sha256": sha256_tokens(call["sequence"]),
        "caps": {"forward_calls": call["forward_calls"], "canvas": int(job.generate["max_new_tokens"])},
        "rng_unchanged": call["rng_unchanged"],
        "latency_ms": call["latency_ms"],
        "trace": {"events": [], "level": "forward-count", "summary": {}, "problems": problems},
    }
