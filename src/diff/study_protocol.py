"""The diff study's protocol as hashed data (plan §3.1, §4.3, §12).

Everything that makes two runs "the same study" is a plain value here: the
Dream call contract, the canvas per phase, the decoding conditions, the
lattice, output limits, gate thresholds and the open-decision defaults.
``PROTOCOL_SHA256`` names the whole. Job identity deliberately covers only the
values a job actually uses (see :mod:`diff.job_plan`), so editing, say, a gate
threshold never renames a generation job.

The decoding factor is parallelism k = canvas ÷ steps, the nominal number of
tokens Dream commits per step at a fixed canvas and ``alg``. Dream has no
unmasking threshold; emulating one with hooks would be a different sampler,
not the native path. The canvas is the block of mask positions appended after
the prompt: Dream fills all of it, so its size changes the output and is a
fixed protocol value per phase, the same for every condition, never a cap.

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


PROTOCOL_SCHEMA: Final = "diff.study-protocol/v5"

# CV-15: Dream's token IDs at 05334cb9, checked against the pinned tokenizer.
# <|endoftext|> is both EOS and PAD; <|im_end|> ends a chat turn. BOS differs
# between config.json (151643) and the tokenizer (<|beginoftext|>, 151665);
# neither is used (add_bos_token is false and the chat template has no BOS), so
# both are recorded and a rendered prompt containing either is refused.
MASK_ID: Final = 151666
ENDOFTEXT_ID: Final = 151643
IM_END_ID: Final = 151645
CONFIG_BOS_ID: Final = 151643
TOKENIZER_BOS_ID: Final = 151665
TERMINATOR_IDS: Final[dict[int, str]] = {ENDOFTEXT_ID: "endoftext", IM_END_ID: "im_end"}
BOS_IDS: Final = (CONFIG_BOS_ID, TOKENIZER_BOS_ID)

# Trida-7B's token IDs at c37755e8 (researcher's model since 2026-10-12), checked
# against the pinned tokenizer: <|im_end|> ends a turn and is generate's stop
# token; <|endoftext|> is BOS/unk; <|pad|> fills after the stop; <|mask|> is the
# diffusion mask. There is no separate end-of-turn token.
TRIDA_MASK_ID: Final = 128012
TRIDA_IM_START_ID: Final = 128000
TRIDA_IM_END_ID: Final = 128001
TRIDA_PAD_ID: Final = 128004
TRIDA_BOS_ID: Final = 0
# config.json max_position_embeddings (the model card says 8,192; the config is what runs).
TRIDA_CONTEXT_LIMIT: Final = 4096

# CV-09 / MC-08: prompt plus canvas must fit Dream's 2,048-token context (README).
# config.json's max_position_embeddings of 131,072 is not a usable limit.
CONTEXT_LIMIT: Final = 2048
CONFIG_MAX_POSITION_EMBEDDINGS: Final = 131072

# The Dream call contract (CV-05, CV-10–CV-12). Set by the researcher on
# 2026-10-11 (protocol v4), after the k1 diagnostic: under greedy decoding
# Dream commits end-of-text first and returns empty captions (in English too),
# and no deterministic setting tried avoided that. Only the README's sampling
# settings did, so Dream samples at temperature 0.2 with top_p 0.95. Each call
# is seeded from its job's identity immediately before it runs, which keeps a
# job reproducible and a retry identical. alg_temp stays 0: the positions
# committed at each step are still chosen by topk, not sampled.
ALG: Final = "entropy"
ALGORITHMS: Final = ("entropy", "topk_margin", "maskgit_plus")
ALG_TEMP: Final = 0.0
EPS: Final = 1e-3
TEMPERATURE: Final = 0.2
TOP_P: Final = 0.95
# The worker's start-up check calls twice with this seed and needs identical output.
WARMUP_SEED: Final = 0
SYSTEM_MESSAGE: Final = "You are a helpful assistant."

# CV-08, set at M0 on 2026-10-10 by the researcher. The canvas covers each
# phase's character limit: 96 characters is about 72 tokens and 160 about 120,
# at the roughly 0.75 tokens per character measured on the development
# captions. At the former 128/192, two-thirds or more of every canvas had to
# be end-of-text, and k1/k2 often filled all of it. It must be divisible by
# every k.
CANVAS_TOKENS: Final[dict[int, int]] = {1: 80, 2: 128}
# Finalised at M0 on 2026-10-10: about twice the slowest job measured, which was
# dream_k1 on the former 192 canvas at about 32 s on an RTX 3090. Generation is
# offline, so the deadline only catches a stuck worker.
TIMEOUT_SECONDS: Final = 60.0
MAX_CHARS: Final[dict[int, int]] = {1: 96, 2: 160}
# Live phase 2 (researcher, 2026-10-12): live captions are written on an
# 80-token canvas, about 5.7 s at k2 on an RTX 3090 against 8.9 s at 128, so a
# tweak shows sooner. A starting caption, and the rewrite, must therefore fit
# phase 1's 96 characters.
LIVE_CONDITION: Final = "dream_k2"
LIVE_CANVAS_TOKENS: Final = 80
LIVE_MAX_CHARS: Final = 96
# The texture slider spans steps 0-2 (researcher, 2026-10-12). On held-out
# captions at the live canvas, the 0-4 range gave 25-30 broken, looping or
# copied captions in 55, against 22 with the cap.
LIVE_TEXTURE_MAX: Final = 2
MAX_LINES: Final[dict[int, int]] = {1: 2}
CUES_PER_CLIP: Final = 4
PHASE2_CUE_MS: Final = 5000

CONTROL_CONDITION: Final = "original"
QWEN_CONDITION: Final = "qwen"
K_CONDITIONS: Final[dict[str, int]] = {"dream_k1": 1, "dream_k2": 2, "dream_k4": 4}
DEFAULT_PHASE1_CONDITIONS: Final = ("original", "dream_k1", "dream_k2", "dream_k4")
PHASE1_CONDITIONS: Final = (*DEFAULT_PHASE1_CONDITIONS, QWEN_CONDITION)
DEFAULT_PHASE2_CONDITION: Final = "dream_k1"

CONTEXT_STEPS: Final = ("broad", "specific", "relation")
# The texture axis is the template's {k}: how many closed-vocabulary qualities a
# cell may use. It is called quality_limit here so it is never confused with
# the decoding factor k (parallelism), and cell IDs use t, not k.
LATTICES: Final[dict[str, dict[str, tuple[Any, ...]]]] = {
    "3x3": {"quality_limit": (0, 2, 4), "context": CONTEXT_STEPS},
    "5x3": {"quality_limit": (0, 1, 2, 3, 4), "context": CONTEXT_STEPS},
}
DEFAULT_LATTICE: Final = "3x3"

GATES: Final[dict[str, Any]] = {
    "status": "provisional operating thresholds, not validated cut-offs",
    "G0": {
        "condition": "dream_k1",
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
        "script_share_denominator": "every dream_k1 job, including jobs without output",
        # Decided by the researcher on 2026-10-11: Dream copied the source caption
        # under every setting that produced text, and a rater shown a human
        # caption copied back could pass the gate on copies. An exact copy fails
        # the check; near-copies stay flags.
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
        "default": "A: dream_k1/k2/k4 with original as the control",
        "alternatives": ["C: steering tracks at one decoding condition"],
    },
    "decoding_factor": {
        "default": "parallelism k at alg=entropy",
        "alternatives": ["remasking order (entropy, topk_margin, maskgit_plus) at k = 1"],
    },
    "phase2_decoding": {
        "default": "fixed, chosen after G1 (dream_k1, the README's setting)",
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
    "dream": {
        "call": "diffusion_generate",
        "batch_size": 1,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": None,
        "alg": ALG,
        "alg_temp": ALG_TEMP,
        "eps": EPS,
        "mask_token_id": MASK_ID,
        "output_history": True,
        "return_dict_in_generate": True,
        "generation_config": "never passed",
        "rng": "torch.manual_seed(job seed) immediately before each call",
        "seed": "the first 8 hex digits of the job ID, as an integer",
        "warmup_seed": WARMUP_SEED,
        "reproducibility": "worker start-up refuses a device where two seeded calls differ",
    },
    "qwen": {
        "call": "generate",
        "batch_size": 1,
        "do_sample": False,
        "repetition_penalty": 1.0,
        "eos_token_id": [IM_END_ID, ENDOFTEXT_ID],
        "unset": ["temperature", "top_p", "top_k"],
    },
    "live": {
        "condition": LIVE_CONDITION,
        "canvas_tokens": LIVE_CANVAS_TOKENS,
        "max_chars": LIVE_MAX_CHARS,
        "texture_max_step": LIVE_TEXTURE_MAX,
    },
    "system_message": SYSTEM_MESSAGE,
    "canvas_tokens": CANVAS_TOKENS,
    "k_conditions": K_CONDITIONS,
    "timeout_seconds": TIMEOUT_SECONDS,
    "context_limit": CONTEXT_LIMIT,
    "config_max_position_embeddings": CONFIG_MAX_POSITION_EMBEDDINGS,
    "caps": "forward calls must equal steps and new positions must equal the canvas",
    "retry": "one identical retry only after a transient runner or device failure",
}

PROTOCOL: Final[dict[str, Any]] = {
    "schema": PROTOCOL_SCHEMA,
    "generation": GENERATION,
    "token_ids": {
        "mask": MASK_ID,
        "endoftext": ENDOFTEXT_ID,
        "im_end": IM_END_ID,
        "config_bos": CONFIG_BOS_ID,
        "tokenizer_bos": TOKENIZER_BOS_ID,
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
    if any(canvas % k for k in K_CONDITIONS.values()):
        raise ValueError(f"The phase-{phase} canvas {canvas} is not divisible by every k")
    return canvas


def steps_for(phase: int, k: int) -> int:
    """Dream's ``steps`` for parallelism k: canvas ÷ k, which must divide exactly."""
    canvas = canvas_tokens(phase)
    if type(k) is not int or k < 1 or canvas % k:
        raise ValueError(f"k must be a positive integer dividing the canvas {canvas}, got {k!r}")
    return canvas // k


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
