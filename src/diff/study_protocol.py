"""The diff study's protocol as hashed data (plan §3.1, §4.3, §12).

Everything that makes two runs "the same study" is a plain value here: the
Trida call contract, the canvas per phase, the decoding conditions, the
lattice, output limits, gate thresholds and the open-decision defaults.
``PROTOCOL_SHA256`` names the whole. Job identity deliberately covers only the
values a job actually uses (see :mod:`diff.job_plan`), so editing, say, a gate
threshold never renames a generation job.

The decoding factor is Trida's unmasking threshold τ (the researcher switched
back from Dream to Trida-7B on 2026-10-12). Within each 32-token block Trida
commits every masked position whose confidence exceeds τ, and at least the
most confident one per step, so a lower τ commits more tokens in parallel. The
canvas is ``max_new_tokens``: a whole number of 32-token blocks. Generation
stops at ``<|im_end|>``, so the canvas caps the output rather than being filled.

The canonical hashing helpers live here because every other module already
depends on this one: canonical JSON is sorted, compact, UTF-8 and refuses NaN.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Any, Final


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_of(value: Any) -> str:
    """SHA-256 of a value's canonical JSON."""
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_text(text: str) -> str:
    """SHA-256 of a string's UTF-8 bytes, for templates and rendered prompts."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_tokens(token_ids: Sequence[int]) -> str:
    """SHA-256 of a token-ID list, as the canonical JSON list of integers."""
    return sha256_of([int(token) for token in token_ids])


PROTOCOL_SCHEMA: Final = "diff.study-protocol/v6"

# Qwen2.5-7B-Instruct's token IDs at a09a3545 (the baseline), checked against the
# pinned tokenizer: <|endoftext|> is EOS and PAD, <|im_end|> ends a chat turn.
# config.json's BOS (151643) is never used (the chat template has none), so a
# rendered prompt containing it is refused.
QWEN_ENDOFTEXT_ID: Final = 151643
QWEN_IM_END_ID: Final = 151645
QWEN_CONFIG_BOS_ID: Final = 151643
QWEN_TERMINATOR_IDS: Final[dict[int, str]] = {QWEN_ENDOFTEXT_ID: "endoftext", QWEN_IM_END_ID: "im_end"}
QWEN_BOS_IDS: Final = (QWEN_CONFIG_BOS_ID,)
# A conservative prompt-plus-output limit for the baseline, far above any prompt here.
QWEN_CONTEXT_LIMIT: Final = 2048

# Trida-7B's token IDs at c37755e8 (researcher's model since 2026-10-12), checked
# against the pinned tokenizer: <|im_end|> ends a turn and is generate's stop
# token; <|endoftext|> is BOS/unk; <|pad|> fills after the stop; <|mask|> is the
# diffusion mask. There is no separate end-of-turn token.
TRIDA_MASK_ID: Final = 128012
TRIDA_IM_START_ID: Final = 128000
TRIDA_IM_END_ID: Final = 128001
TRIDA_PAD_ID: Final = 128004
TRIDA_BOS_ID: Final = 0
TRIDA_TERMINATOR_IDS: Final[dict[int, str]] = {TRIDA_IM_END_ID: "im_end"}
TRIDA_BOS_IDS: Final = (TRIDA_BOS_ID,)
# config.json max_position_embeddings (the model card says 8,192; the config is what runs).
TRIDA_CONTEXT_LIMIT: Final = 4096

# The Trida call contract (plan §4.4): batch 1, no attention mask, temperature 0
# (argmax; top_p is then never applied, so it is not passed), no block cache,
# the pinned block sizes and token IDs. Nothing in a call consumes randomness.
TRIDA_BLOCK_SIZE: Final = 32
TRIDA_SMALL_BLOCK_SIZE: Final = 8
TRIDA_TEMPERATURE: Final = 0
TRIDA_USE_BLOCK_CACHE: Final = False
SYSTEM_MESSAGE: Final = "You are a helpful assistant."

# CV-08 for Trida (2026-10-12): whole 32-token blocks covering each phase's
# character limit. Trida's tokenizer takes 0.43 tokens per Korean character on
# average and 0.75 at most on the c2, development and synthetic captions, so 96
# characters need at most about 72 tokens (three blocks) and 160 about 120 (four).
CANVAS_TOKENS: Final[dict[int, int]] = {1: 96, 2: 128}
# Set at M0 on 2026-10-10 (then for Dream); Trida takes about 0.4 s a caption,
# so the deadline only catches a stuck worker.
TIMEOUT_SECONDS: Final = 60.0
MAX_CHARS: Final[dict[int, int]] = {1: 96, 2: 160}
# Live phase 2: a starting caption and its rewrite fit phase 1's 96 characters
# (researcher, 2026-10-12), so live captions use phase 1's three-block canvas.
# Trida writes one in about 0.4 s at τ 0.9, its README setting.
LIVE_CONDITION: Final = "trida_tau_090"
LIVE_CANVAS_TOKENS: Final = 96
LIVE_MAX_CHARS: Final = 96
# The texture slider spans steps 0-2 (researcher, 2026-10-12). On held-out
# captions with Dream, the 0-4 range gave 25-30 broken, looping or copied
# captions in 55, against 22 with the cap; the cap stays for Trida.
LIVE_TEXTURE_MAX: Final = 2
MAX_LINES: Final[dict[int, int]] = {1: 2}
CUES_PER_CLIP: Final = 4
PHASE2_CUE_MS: Final = 5000

CONTROL_CONDITION: Final = "original"
QWEN_CONDITION: Final = "qwen"
# The brief's original decoding conditions, restored with Trida on 2026-10-12.
TAU_CONDITIONS: Final[dict[str, float]] = {"trida_tau_070": 0.7, "trida_tau_090": 0.9, "trida_tau_100": 1.0}
DEFAULT_PHASE1_CONDITIONS: Final = (CONTROL_CONDITION, *TAU_CONDITIONS)
PHASE1_CONDITIONS: Final = (*DEFAULT_PHASE1_CONDITIONS, QWEN_CONDITION)
DEFAULT_PHASE2_CONDITION: Final = "trida_tau_090"

CONTEXT_STEPS: Final = ("broad", "specific", "relation")
# The texture axis is the template's {k}: how many closed-vocabulary qualities a
# cell may use. It is called quality_limit here so it is never confused with a
# decoding setting, and cell IDs use t, not k.
LATTICES: Final[dict[str, dict[str, tuple[Any, ...]]]] = {
    "3x3": {"quality_limit": (0, 2, 4), "context": CONTEXT_STEPS},
    "5x3": {"quality_limit": (0, 1, 2, 3, 4), "context": CONTEXT_STEPS},
}
DEFAULT_LATTICE: Final = "3x3"

GATES: Final[dict[str, Any]] = {
    "status": "provisional operating thresholds, not validated cut-offs",
    "G0": {
        "condition": DEFAULT_PHASE2_CONDITION,
        "terminal": "every job terminal with no unresolved technical failure",
        "min_script_pass_share": 0.95,
        "incomplete_counts_as_failure": True,
        # Decided by the researcher on 2026-10-10. The brief's script check is
        # Han/kana and stray Latin plus incomplete. Output that cannot be shown
        # as a caption fails too, and the share is over every job, so a job with
        # no output (a timeout) counts against it.
        "script_check_failures": (
            "non_korean_script",
            "undecodable_bytes",
            "incomplete",
            "empty_text",
            "leftover_mask",
        ),
        "script_share_denominator": "every job of the rated condition, including jobs without output",
        # Decided by the researcher on 2026-10-11 (then for Dream, which copied the
        # source caption under every setting): a rater shown a human caption
        # copied back could pass the gate on copies. An exact copy fails the
        # check; near-copies stay flags.
        "script_check_failing_flags": ("identical_to_source",),
        "min_naturalness": 3,
        "min_naturalness_share": 0.75,
        "minimum_phase2_cell_prompts": 12,
        "baseline_reported_beside": "qwen",
    },
    "G1": {
        "technical": "every planned development job terminal, zero unresolved technical failures",
        "readability": {"min_naturalness": 3, "min_clip_share": 0.75},
        "difference": {"min_naturalness": 4, "min_differing_decoding_tracks": 2, "min_clip_share": 0.5},
        "on_difference_failure": "remove the decoding factor from the participant conditions (option C)",
    },
    "G2": {"min_monotone_cue_share": 0.8, "max_identical_neighbour_rate": 0.2},
}

OPEN_DECISIONS: Final[dict[str, dict[str, Any]]] = {
    "phase1_manipulation": {
        "default": "A: trida_tau_070/090/100 with original as the control",
        "alternatives": ["C: steering tracks at one decoding condition"],
    },
    "decoding_factor": {
        "default": "Trida's unmasking threshold τ in {0.7, 0.9, 1.0}",
        "alternatives": ["one τ with steering tracks (option C)"],
    },
    "phase2_decoding": {
        "default": "live, fixed at τ 0.9 (the README's setting)",
        "alternatives": ["assigned per video"],
    },
    "lattice": {"default": "3x3", "alternatives": ["5x3 after G2"]},
    "baseline": {
        "default": "Qwen2.5-7B-Instruct, review stage only",
        "alternatives": ["participant condition"],
    },
    "phase2_sources": {
        "default": "reviewed Gemma drafts with provenance",
        "alternatives": ["re-authored by people"],
    },
    "missing_lattice_cell": {
        "default": "nearest-cell substitute, recorded with the original cell",
        "alternatives": ["exclude the video"],
    },
    "abandoned_slot": {
        "default": "reissue only after researcher release, and only if no formal response exists",
        "alternatives": ["reissue after release regardless", "auto-release after inactivity"],
    },
}

GENERATION: Final[dict[str, Any]] = {
    "trida": {
        "call": "generate (remote code, block diffusion)",
        "batch_size": 1,
        "block_size": TRIDA_BLOCK_SIZE,
        "small_block_size": TRIDA_SMALL_BLOCK_SIZE,
        "temperature": TRIDA_TEMPERATURE,
        "top_p": "not passed: temperature 0 never applies it",
        "use_block_cache": TRIDA_USE_BLOCK_CACHE,
        "mask_id": TRIDA_MASK_ID,
        "stop_token": TRIDA_IM_END_ID,
        "pad_token_id": TRIDA_PAD_ID,
        "rng": "none consumed",
        "context_limit": TRIDA_CONTEXT_LIMIT,
    },
    "qwen": {
        "call": "generate",
        "batch_size": 1,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "eos_token_id": [QWEN_IM_END_ID, QWEN_ENDOFTEXT_ID],
        "unset": ["temperature", "top_p", "top_k"],
        "context_limit": QWEN_CONTEXT_LIMIT,
    },
    "live": {
        "condition": LIVE_CONDITION,
        "canvas_tokens": LIVE_CANVAS_TOKENS,
        "max_chars": LIVE_MAX_CHARS,
        "texture_max_step": LIVE_TEXTURE_MAX,
    },
    "system_message": SYSTEM_MESSAGE,
    "canvas_tokens": CANVAS_TOKENS,
    "tau_conditions": TAU_CONDITIONS,
    "timeout_seconds": TIMEOUT_SECONDS,
    "caps": "a canvas of whole 32-token blocks; no call may consume randomness",
    "retry": "one identical retry only after a transient runner or device failure",
}

PROTOCOL: Final[dict[str, Any]] = {
    "schema": PROTOCOL_SCHEMA,
    "generation": GENERATION,
    "token_ids": {
        "trida": {
            "mask": TRIDA_MASK_ID,
            "im_start": TRIDA_IM_START_ID,
            "im_end": TRIDA_IM_END_ID,
            "pad": TRIDA_PAD_ID,
            "bos": TRIDA_BOS_ID,
        },
        "qwen": {"endoftext": QWEN_ENDOFTEXT_ID, "im_end": QWEN_IM_END_ID, "config_bos": QWEN_CONFIG_BOS_ID},
    },
    "phase1": {
        "cues_per_clip": CUES_PER_CLIP,
        "max_chars": MAX_CHARS[1],
        "max_lines": MAX_LINES[1],
        "conditions": DEFAULT_PHASE1_CONDITIONS,
        "control": CONTROL_CONDITION,
    },
    "phase2": {
        "cue_ms": PHASE2_CUE_MS,
        "max_chars": MAX_CHARS[2],
        "lattice": DEFAULT_LATTICE,
        "decoding": DEFAULT_PHASE2_CONDITION,
    },
    "lattices": LATTICES,
    "gates": GATES,
    "open_decisions": OPEN_DECISIONS,
}
PROTOCOL_SHA256: Final = sha256_of(PROTOCOL)


def canvas_tokens(phase: int) -> int:
    if phase not in CANVAS_TOKENS:
        raise ValueError(f"Unknown phase: {phase}")
    canvas = CANVAS_TOKENS[phase]
    if canvas < TRIDA_BLOCK_SIZE or canvas % TRIDA_BLOCK_SIZE:
        raise ValueError(
            f"The phase-{phase} canvas {canvas} is not a whole number of {TRIDA_BLOCK_SIZE}-token blocks"
        )
    return canvas


def lattice_cells(name: str = DEFAULT_LATTICE) -> list[dict[str, Any]]:
    """Cells in row-major order (context outer, texture inner), with internal IDs."""
    if name not in LATTICES:
        raise ValueError(f"Unknown lattice: {name}")
    lattice = LATTICES[name]
    textures, contexts = lattice["quality_limit"], lattice["context"]
    centre = (len(textures) // 2, len(contexts) // 2)
    return [
        {
            "cell_id": f"t{limit}-{context}",
            "texture_index": t_index,
            "quality_limit": limit,
            "context_index": c_index,
            "context": context,
            "is_centre": (t_index, c_index) == centre,
        }
        for c_index, context in enumerate(contexts)
        for t_index, limit in enumerate(textures)
    ]


def lattice_corners(name: str = DEFAULT_LATTICE) -> list[dict[str, Any]]:
    """The four corner cells — the presets — in a fixed order."""
    cells = lattice_cells(name)
    textures = LATTICES[name]["quality_limit"]
    wanted = [
        (textures[0], CONTEXT_STEPS[0]),
        (textures[-1], CONTEXT_STEPS[0]),
        (textures[0], CONTEXT_STEPS[-1]),
        (textures[-1], CONTEXT_STEPS[-1]),
    ]
    return [next(c for c in cells if (c["quality_limit"], c["context"]) == pair) for pair in wanted]
