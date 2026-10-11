"""Dream-v0-Instruct-7B through its native ``diffusion_generate`` (plan §4.4).

Every keyword of the call is required and explicit. Without a
``generation_config`` argument Dream builds its base config from
``config.json``, and in transformers 4.46.2 every model config carries legacy
generation defaults (temperature 1.0, top_k 50, top_p 1.0) that override
Dream's class defaults; an omitted ``temperature`` would sample, an omitted
``top_k`` would filter, and omitted ``alg`` and ``steps`` fall back to
``"origin"`` and 512. ``generation_config.update`` drops unknown keys
silently, so :func:`checked_call` refuses any keyword outside the contract, and
the resolved config is recorded with every attempt. No ``generation_config``
object is ever passed: with one, missing special tokens are filled from the
shipped ``generation_config.json``, whose ``mask_token_id`` is null.

Each call is also held to the §6.2 caps: forward calls must equal ``steps``
(a read-only forward hook counts them, AC-08) and new positions must equal the
canvas. Since protocol v4 (2026-10-11) Dream samples at the README's
temperature 0.2 and ``top_p`` 0.95, so a call does consume randomness: it is
seeded with its job's seed immediately before it runs, and the worker refuses
to start on a device where two seeded calls differ (AC-07, revised).
Realized per-step commit counts come from ``output_history``. The full decode
trace — the logits hook that recomputes confidences and selections — follows a
passing G0 (follow-up §7) and is not part of this module yet.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from diff.job_plan import FrozenJob, prompt_guard_problems
from diff.model_profile import (
    DREAM_7B,
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
)
from diff.output_checks import extract_output
from diff.study_protocol import (
    ALGORITHMS,
    CONTEXT_LIMIT,
    EPS,
    MASK_ID,
    TEMPERATURE,
    TERMINATOR_IDS,
    TOP_P,
    WARMUP_SEED,
    sha256_tokens,
)

REQUIRED_KEYWORDS: Final = (
    "attention_mask",
    "max_new_tokens",
    "steps",
    "temperature",
    "top_p",
    "top_k",
    "alg",
    "alg_temp",
    "eps",
    "mask_token_id",
    "output_history",
    "return_dict_in_generate",
)
HOOK_KEYWORDS: Final = ("generation_tokens_hook_func", "generation_logits_hook_func")
RESOLVED_FIELDS: Final = (
    "max_new_tokens",
    "steps",
    "temperature",
    "top_p",
    "top_k",
    "alg",
    "alg_temp",
    "eps",
    "mask_token_id",
    "output_history",
    "return_dict_in_generate",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
)
TOKEN_NAMES: Final = {
    "mask": "<|mask|>",
    "endoftext": "<|endoftext|>",
    "im_end": "<|im_end|>",
    "tokenizer_bos": "<|beginoftext|>",
}


def _is_zero(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and value == 0


def checked_call(arguments: Mapping[str, Any]) -> dict[str, Any]:
    """Exactly the contract's keywords and values, or :class:`RefusedCall` naming every problem."""
    problems = []
    for keyword in REQUIRED_KEYWORDS:
        if keyword not in arguments:
            problems.append(f"missing required keyword {keyword!r}")
    extra = sorted(set(arguments) - set(REQUIRED_KEYWORDS) - set(HOOK_KEYWORDS))
    if extra:
        problems.append(f"refused keywords {extra}: generation_config.update would drop them silently")
    alg = arguments.get("alg")
    if "alg" in arguments and alg == "origin":
        problems.append("alg='origin' is refused: it orders unmasking by torch.rand")
    elif "alg" in arguments and alg not in ALGORITHMS:
        problems.append(f"alg must be one of {list(ALGORITHMS)}, got {alg!r}")
    for keyword, wanted in (("temperature", TEMPERATURE), ("top_p", TOP_P)):
        value = arguments.get(keyword)
        if keyword in arguments and (isinstance(value, bool) or value != wanted):
            problems.append(f"{keyword} must be {wanted} (protocol v4), got {value!r}")
    if "alg_temp" in arguments and not _is_zero(arguments["alg_temp"]):
        problems.append(f"alg_temp must be 0, got {arguments['alg_temp']!r}")
    if "top_k" in arguments and arguments["top_k"] is not None:
        problems.append(
            f"top_k must be None (filtering runs before the confidence), got {arguments['top_k']!r}"
        )
    if "eps" in arguments and arguments["eps"] != EPS:
        problems.append(f"eps must be {EPS}, got {arguments['eps']!r}")
    if "mask_token_id" in arguments and arguments["mask_token_id"] != MASK_ID:
        problems.append(f"mask_token_id must be {MASK_ID}, got {arguments['mask_token_id']!r}")
    for keyword in ("output_history", "return_dict_in_generate"):
        if keyword in arguments and arguments[keyword] is not True:
            problems.append(f"{keyword} must be True")
    canvas, steps = arguments.get("max_new_tokens"), arguments.get("steps")
    if "max_new_tokens" in arguments and (type(canvas) is not int or canvas < 1):
        problems.append(f"max_new_tokens (the canvas) must be a positive integer, got {canvas!r}")
    if "steps" in arguments and (type(steps) is not int or steps < 1):
        problems.append(f"steps must be a positive integer, got {steps!r}")
    elif type(canvas) is int and type(steps) is int and (steps > canvas or canvas % steps):
        problems.append(f"steps {steps} must not exceed the canvas {canvas} and must divide it")
    for keyword in HOOK_KEYWORDS:
        if keyword in arguments and not callable(arguments[keyword]):
            problems.append(f"{keyword} must be callable")
    if problems:
        raise RefusedCall("; ".join(problems))
    return dict(arguments)


def context_problems(prompt_ids: Sequence[int], canvas: int) -> list[str]:
    """Prompt-side refusals: the 2,048-token context and no mask token in the prompt."""
    problems = []
    if len(prompt_ids) + canvas > CONTEXT_LIMIT:
        problems.append(f"{len(prompt_ids)} prompt tokens + a {canvas}-token canvas exceed {CONTEXT_LIMIT}")
    if MASK_ID in prompt_ids:
        problems.append(
            f"the prompt contains the mask token {MASK_ID}; the mask search would generate into it"
        )
    return problems


def load(
    model_dir: Path, verification: Mapping[str, Any], *, device: str, lock_path: Path | None = None
) -> LoadedModel:
    """Re-verify the small files, confirm the weights are the verified ones, then load offline."""
    verify_snapshot(model_dir, DREAM_7B, weights=False)
    assert_weights_unchanged(model_dir, verification)
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    model: Any = AutoModel.from_pretrained(
        str(model_dir), trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16
    )
    model = model.to(device).eval()
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        device=device,
        profile=DREAM_7B,
        loaded_code=verify_loaded_code([model, model.config, tokenizer], DREAM_7B),
        environment=environment_fingerprint(lock_path),
        token_ids=checked_token_ids(tokenizer, DREAM_7B.token_ids, TOKEN_NAMES),
    )


def resolved_config(model: Any, arguments: Mapping[str, Any]) -> dict[str, Any]:
    """What the generation config resolves to for these keywords, from a copy (no side effects)."""
    call = {key: value for key, value in arguments.items() if key not in ("attention_mask", *HOOK_KEYWORDS)}
    config = model._prepare_generation_config(None, **call)
    return {key: getattr(config, key, None) for key in RESOLVED_FIELDS}


def realized_counts(
    initial: Sequence[int], history: Sequence[Sequence[int]], prompt_tokens: int
) -> list[int]:
    """Positions that left the mask ID at each step (zero allowed; the last step takes the rest)."""
    counts, previous = [], list(initial)
    for state in history:
        counts.append(
            sum(1 for p in range(prompt_tokens, len(state)) if previous[p] == MASK_ID and state[p] != MASK_ID)
        )
        previous = list(state)
    return counts


def seed_all(seed: int) -> None:
    """Seed the CPU and every CUDA generator; called immediately before each Dream call."""
    import torch

    torch.manual_seed(seed)


def warm_up(loaded: LoadedModel) -> dict[str, Any]:
    """Two tiny untimed calls with the same seed, which must agree; refuses the device otherwise.

    They also keep CUDA initialisation out of the first cue's deadline.
    """
    import torch

    rendered = loaded.tokenizer.apply_chat_template(
        [{"role": "user", "content": "안녕하세요."}], add_generation_prompt=True, tokenize=True
    )
    input_ids = torch.tensor([[int(token) for token in rendered]], dtype=torch.long, device=loaded.device)
    arguments = checked_call(
        {
            "attention_mask": torch.ones_like(input_ids),
            "max_new_tokens": 8,
            "steps": 8,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": None,
            "alg": "entropy",
            "alg_temp": 0.0,
            "eps": EPS,
            "mask_token_id": MASK_ID,
            "output_history": True,
            "return_dict_in_generate": True,
        }
    )
    started = time.monotonic()
    sequences = []
    for _ in range(2):
        seed_all(WARMUP_SEED)
        output = loaded.model.diffusion_generate(input_ids, **arguments)
        sequences.append([int(token) for token in output.sequences[0].tolist()])
    if sequences[0] != sequences[1]:
        raise RuntimeError(
            "two calls with the same seed differ on this device; seeded sampling is not reproducible"
        )
    return {
        "wall_ms": round((time.monotonic() - started) * 1000),
        "seed": WARMUP_SEED,
        "seeded_calls_identical": True,
    }


def call_with_caps(
    loaded: LoadedModel,
    prompt_ids: Sequence[int],
    generate: Mapping[str, Any],
    seed: int,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """One seeded diffusion_generate call with its caps measured; raises RefusedCall before the call."""
    import torch

    canvas = int(generate["max_new_tokens"])
    refusals = context_problems(prompt_ids, canvas)
    if refusals:
        raise RefusedCall("; ".join(refusals))
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=loaded.device)
    arguments = checked_call({**generate, **(extra or {}), "attention_mask": torch.ones_like(input_ids)})
    resolved = resolved_config(loaded.model, arguments)
    forward_calls: list[int] = []

    def count_forward(module: Any, args: Any, output: Any) -> None:
        forward_calls.append(1)

    handle = loaded.model.register_forward_hook(count_forward)
    seed_all(seed)
    started = time.monotonic()
    try:
        output = loaded.model.diffusion_generate(input_ids, **arguments)
    finally:
        handle.remove()
    latency_ms = round((time.monotonic() - started) * 1000)
    sequence = [int(token) for token in output.sequences[0].tolist()]
    history = [[int(token) for token in state[0].tolist()] for state in output.history]
    initial = [*prompt_ids, *([MASK_ID] * canvas)]
    return {
        "sequence": sequence,
        "history_steps": len(history),
        "realized_per_step": realized_counts(initial, history, len(prompt_ids)),
        "forward_calls": len(forward_calls),
        "seed": seed,
        "latency_ms": latency_ms,
        "resolved_config": resolved,
    }


def run_job(loaded: LoadedModel, frozen: FrozenJob) -> dict[str, Any]:
    """One attempt's model-side result: status, output, caps and per-step counts."""
    job = frozen.job
    ids = chat_token_ids(loaded.tokenizer, job.prompt)
    if sha256_tokens(ids) != frozen.token_ids_sha256 or len(ids) != frozen.token_count:
        return {"status": "refused", "error": "prompt token IDs differ from the frozen plan"}
    guards = prompt_guard_problems(job, ids)
    if guards:
        return {"status": "refused", "error": "; ".join(guards)}
    seed = job.seed
    if seed is None:
        return {"status": "refused", "error": "a Dream job has no seed"}
    call = call_with_caps(loaded, ids, job.generate, seed)
    canvas, steps = int(job.generate["max_new_tokens"]), int(job.generate["steps"])
    output = extract_output(
        call["sequence"], len(ids), loaded.decode, loaded.special_ids, TERMINATOR_IDS, mask_id=MASK_ID
    )
    problems = []
    if call["forward_calls"] != steps or call["history_steps"] != steps:
        problems.append(
            f"{call['forward_calls']} forward calls and {call['history_steps']} history steps, not {steps}"
        )
    if output.new_token_count != canvas:
        problems.append(f"{output.new_token_count} new positions, not the {canvas}-position canvas")
    status = "completed"
    if problems:
        status = "count_mismatch"
    elif output.leftover_mask_count:
        status, problems = "leftover_mask", [f"{output.leftover_mask_count} mask tokens left in the canvas"]
    counts = call["realized_per_step"]
    return {
        "status": status,
        "error": "; ".join(problems) or None,
        "prompt_tokens": len(ids),
        "output": output.record(),
        "output_token_ids_sha256": sha256_tokens(call["sequence"]),
        "resolved_config": call["resolved_config"],
        "caps": {"forward_calls": call["forward_calls"], "steps": steps, "canvas": canvas},
        "seed": call["seed"],
        "latency_ms": call["latency_ms"],
        "trace": {
            "events": [],
            "level": "history-counts",
            "summary": {
                "realized_per_step": counts,
                "zero_commit_steps": sum(1 for count in counts if count == 0),
                "max_commit": max(counts, default=0),
            },
            "problems": problems,
        },
    }
