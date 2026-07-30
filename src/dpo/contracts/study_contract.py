"""Normative vocabularies, the nine-condition experiment matrix, and the study-contract validator.

The vocabularies and the experiment matrix are code-owned: a study contract can
configure thresholds and hyperparameters but can never redefine what a split,
an annotation outcome, or an experiment's initialization is. That keeps the
comparison constraints of the PRD (DPO, IPO, CDPO, RDPO, DRDPO, and WDPO start
from SEED, SFT_DPO starts from SFT, one frozen preference dataset feeds every
training view) out of reach of config drift.
"""

from __future__ import annotations

import copy
import hashlib
import math
import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeGuard

from dpo.core.identity import semantic_hash


class ContractError(ValueError):
    """Raised when the executable study contract is incomplete or ambiguous."""


HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|sha256:[0-9a-f]{64})\Z")

# ---------------------------------------------------------------------------
# Shared normative vocabularies.
# ---------------------------------------------------------------------------

TRACKS = ("visual", "audio")
SPLITS = ("train", "validation", "test", "study")
EXECUTION_CLASSES = ("synthetic_canary", "live")
TERMINAL_STATES = ("offline", "release")
TERMINAL_VALUES = ("pending", "blocked_pending_external_operation", "complete")

# How a clip is presented to a PREFERENCE annotator — someone choosing between
# two candidate captions. Every judgment records which presentation it saw, so
# any cross-modal contamination stays measurable.
#
# This vocabulary is not the downstream user study's condition set. That study
# measures the effect of a trained model's captions on participants and is a
# different instrument with a different response schema; a caption condition
# there does not have to be expressible here.
#
# The visual track is always the muted video, and the audio track must be able
# to hear something — a preference judgment about an audio caption made in
# silence is unverifiable, not a condition.
PRESENTATIONS = ("muted_video", "audio_only", "unmuted_video")
AUDIO_PRESENTATIONS = ("audio_only", "unmuted_video")

# Annotation response options (binary forced choice alone is insufficient).
CHOICES = ("a_better", "b_better", "tie", "both_unacceptable")
TIE_SUBTYPES = ("both_good", "both_acceptable", "both_bad", "indistinguishable")
REASON_TAGS = (
    "factual_support",
    "coverage",
    "temporal",
    "specificity",
    "concision",
    "modality_purity",
    "language_quality",
    "accessibility",
)

PAIR_CATEGORIES = (
    "quality_contrast",
    "factuality_contrast",
    "specificity_contrast",
    "style_contrast",
    "ambiguous_near_tie",
)

CANDIDATE_SOURCES = (
    "greedy",
    "sample",
    "alt_decoding",
    "controlled_error",
)
CHALLENGE_SOURCES = frozenset({"controlled_error"})

OBJECTIVE_NAMES = ("sft", "dpo", "ipo", "cdpo", "rdpo", "drdpo", "wdpo")
TRAINING_VIEWS = ("sft", "pair_strict", "pair_all")

# Per-objective loss knobs that a contract may sweep by giving a list of
# values (e.g. beta = [0.1, 0.3]). Each list expands to one trained variant
# per value; validation-time selection picks one winner per experiment and
# track before anything is locked or studied. Structural knobs (views, modes,
# stage toggles) are never sweepable.
SWEEPABLE_HYPERPARAMETERS = ("beta", "epsilon", "beta_prime")

# ---------------------------------------------------------------------------
# The nine-condition experiment matrix (PRD section 10). Code-owned and immutable.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExperimentSpec:
    """Structural definition of one matrix entry: what may never vary by config."""

    experiment_id: str
    policy_init: str | None  # None (frozen seed), "SEED", or "SFT"
    reference: str | None  # None, "SEED", or "SFT"
    allowed_views: tuple[str, ...]
    objective: str | None
    purpose: str

    @property
    def is_preference(self) -> bool:
        return self.objective is not None and self.objective != "sft"


EXPERIMENT_MATRIX: dict[str, ExperimentSpec] = {
    "SEED": ExperimentSpec("SEED", None, None, (), None, "unaligned task-capable seed"),
    "SFT": ExperimentSpec("SFT", "SEED", None, ("sft",), "sft", "chosen-only SFT"),
    "DPO": ExperimentSpec("DPO", "SEED", "SEED", ("pair_strict",), "dpo", "direct DPO"),
    "IPO": ExperimentSpec("IPO", "SEED", "SEED", ("pair_strict",), "ipo", "direct IPO"),
    "CDPO": ExperimentSpec("CDPO", "SEED", "SEED", ("pair_strict",), "cdpo", "direct cDPO"),
    "RDPO": ExperimentSpec("RDPO", "SEED", "SEED", ("pair_strict",), "rdpo", "direct rDPO"),
    "DRDPO": ExperimentSpec("DRDPO", "SEED", "SEED", ("pair_strict",), "drdpo", "direct Dr.DPO"),
    "WDPO": ExperimentSpec("WDPO", "SEED", "SEED", ("pair_strict", "pair_all"), "wdpo", "direct wDPO"),
    "SFT_DPO": ExperimentSpec("SFT_DPO", "SFT", "SFT", ("pair_strict",), "dpo", "SFT-to-DPO warm start"),
}
EXPERIMENT_IDS = tuple(sorted(EXPERIMENT_MATRIX))


# ---------------------------------------------------------------------------
# Validator kit.
# ---------------------------------------------------------------------------


def _is_table(value: object) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(value, Mapping)


def _table(
    value: object,
    location: str,
    required: set[str],
    optional: set[str] | None = None,
) -> Mapping[str, Any]:
    if not _is_table(value):
        raise ContractError(f"{location} must be a table")
    allowed = required | (optional or set())
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContractError(f"{location}: unknown key {unknown[0]!r}")
    missing = sorted(required - set(value))
    if missing:
        raise ContractError(f"{location}: missing required key {missing[0]!r}")
    return value


def _string(value: object, location: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value.strip()):
        raise ContractError(f"{location} must be non-empty text")
    if "${" in value:
        raise ContractError(f"{location} contains an unresolved value")
    return value


def _boolean(value: object, location: str) -> bool:
    if type(value) is not bool:
        raise ContractError(f"{location} must be an explicit boolean")
    return bool(value)


def _integer(value: object, location: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise ContractError(f"{location} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ContractError(f"{location} must be >= {minimum}")
    return result


def _number(
    value: object,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ContractError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{location} must be finite")
    if minimum is not None and result < minimum:
        raise ContractError(f"{location} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ContractError(f"{location} must be <= {maximum}")
    if exclusive_minimum is not None and result <= exclusive_minimum:
        raise ContractError(f"{location} must be > {exclusive_minimum}")
    return result


def _hash(value: object, location: str) -> str:
    result = _string(value, location)
    if not HASH_RE.fullmatch(result):
        raise ContractError(f"{location} must be a sha256 hash")
    return result


def _revision(value: object, location: str) -> str:
    if not isinstance(value, str) or "${" in value or not REVISION_RE.fullmatch(value):
        raise ContractError(f"{location} must be an immutable revision")
    return value


def _integers(value: object, location: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{location} must be a non-empty integer array")
    return [_integer(item, f"{location}[{index}]") for index, item in enumerate(value)]


def _number_or_axis(
    value: object,
    location: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: float | None = None,
) -> list[float]:
    """A scalar hyperparameter or a sweep axis of unique values, each in bounds."""
    if isinstance(value, list):
        if not value:
            raise ContractError(f"{location} sweep axis must be a non-empty array")
        values = [
            _number(
                item,
                f"{location}[{index}]",
                minimum=minimum,
                maximum=maximum,
                exclusive_minimum=exclusive_minimum,
            )
            for index, item in enumerate(value)
        ]
        if len(set(values)) != len(values):
            raise ContractError(f"{location} sweep axis values must be unique")
        return values
    return [_number(value, location, minimum=minimum, maximum=maximum, exclusive_minimum=exclusive_minimum)]


def _model(value: object, location: str, *, with_init_seed: bool = False) -> None:
    keys = {"model_id", "revision", "implementation", "lock_hash"}
    if with_init_seed:
        keys.add("init_seed")
    table = _table(value, location, keys)
    _string(table["model_id"], f"{location}.model_id")
    _revision(table["revision"], f"{location}.revision")
    _string(table["implementation"], f"{location}.implementation")
    _hash(table["lock_hash"], f"{location}.lock_hash")
    if with_init_seed:
        _integer(table["init_seed"], f"{location}.init_seed", minimum=0)


def _validate_backends(value: object) -> None:
    """Optional pins of the live backend runtime configs (provenance only)."""
    table = _table(value, "backends", set(), {"visual", "audio"})
    for track in sorted(table):
        entry = _table(table[track], f"backends.{track}", {"config_hash"})
        _hash(entry["config_hash"], f"backends.{track}.config_hash")


# ---------------------------------------------------------------------------
# Typed contract views.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaptionContract:
    """One track's caption contract: the output rules every stage reads."""

    track: str
    contract_version: str
    language: str
    min_words: int
    max_words: int
    min_clip_ms: int
    max_clip_ms: int
    prompt: str
    prompt_hash: str
    # Frames sampled per clip on the visual track (0 on the audio track). It
    # decides how much of a clip the model sees, so it is contract-owned like
    # any other result-affecting knob — and it dominates the training
    # sequence length, so it is also what makes a visual cell fit or not.
    video_frames: int
    include_ocr: bool
    include_camera_motion: bool
    transcribe_speech: bool


@dataclass(frozen=True)
class DecodingMixture:
    name: str
    temperature: float
    top_p: float
    count: int


@dataclass(frozen=True)
class StudyContract:
    raw: dict[str, Any]
    contract_hash: str
    execution_class: str
    tracks: dict[str, CaptionContract]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.raw[name]
        assert _is_table(value)
        return value

    @property
    def corpus(self) -> Mapping[str, Any]:
        return self.section("corpus")

    @property
    def split_fractions(self) -> dict[str, float]:
        fractions = self.corpus["split_fractions"]
        return {split: float(fractions[split]) for split in SPLITS}

    @property
    def candidates(self) -> Mapping[str, Any]:
        return self.section("candidates")

    @property
    def decoding_mixture(self) -> tuple[DecodingMixture, ...]:
        return tuple(
            DecodingMixture(
                name=str(entry["name"]),
                temperature=float(entry["temperature"]),
                top_p=float(entry["top_p"]),
                count=int(entry["count"]),
            )
            for entry in self.candidates["decoding"]
        )

    @property
    def pairs(self) -> Mapping[str, Any]:
        return self.section("pairs")

    @property
    def annotation(self) -> Mapping[str, Any]:
        return self.section("annotation")

    @property
    def views(self) -> Mapping[str, Any]:
        return self.section("views")

    @property
    def training(self) -> Mapping[str, Any]:
        return self.section("training")

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(int(seed) for seed in self.training["seeds"])

    @property
    def experiments(self) -> Mapping[str, Mapping[str, Any]]:
        table = self.section("experiments")
        return {name: table[name] for name in EXPERIMENT_IDS}

    def experiment(self, experiment_id: str) -> Mapping[str, Any]:
        if experiment_id not in EXPERIMENT_MATRIX:
            raise ContractError(f"unknown experiment {experiment_id!r}")
        return self.experiments[experiment_id]

    @property
    def validation(self) -> Mapping[str, Any]:
        return self.section("validation")

    @property
    def robustness(self) -> Mapping[str, Any]:
        return self.section("robustness")


# ---------------------------------------------------------------------------
# Section validators.
# ---------------------------------------------------------------------------


def _validate_corpus(value: object) -> None:
    corpus = _table(
        value,
        "corpus",
        {"group_key", "split_seed", "adjacency_guard_ms", "split_fractions"},
    )
    if _string(corpus["group_key"], "corpus.group_key") != "source_video_id":
        raise ContractError("corpus.group_key must be 'source_video_id'")
    _integer(corpus["split_seed"], "corpus.split_seed", minimum=0)
    _integer(corpus["adjacency_guard_ms"], "corpus.adjacency_guard_ms", minimum=1)
    fractions = _table(corpus["split_fractions"], "corpus.split_fractions", set(SPLITS))
    total = 0.0
    for split in SPLITS:
        total += _number(
            fractions[split], f"corpus.split_fractions.{split}", exclusive_minimum=0.0, maximum=1.0
        )
    if abs(total - 1.0) > 1e-9:
        raise ContractError("corpus.split_fractions must sum to 1.0")


def _validate_track(track: str, value: object) -> CaptionContract:
    keys = {
        "contract_version",
        "language",
        "min_words",
        "max_words",
        "min_clip_ms",
        "max_clip_ms",
        "prompt",
        "prompt_hash",
    }
    if track == "visual":
        keys.add("video_frames")
        optional = {"include_ocr", "include_camera_motion"}
    else:
        optional = {"transcribe_speech"}
    table = _table(value, f"tracks.{track}", keys, optional)
    language = _string(table["language"], f"tracks.{track}.language")
    if language != "en":
        raise ContractError(f"tracks.{track}.language must be 'en' (multilingual output is out of scope)")
    min_words = _integer(table["min_words"], f"tracks.{track}.min_words", minimum=1)
    max_words = _integer(table["max_words"], f"tracks.{track}.max_words", minimum=min_words)
    min_clip = _integer(table["min_clip_ms"], f"tracks.{track}.min_clip_ms", minimum=1)
    max_clip = _integer(table["max_clip_ms"], f"tracks.{track}.max_clip_ms", minimum=min_clip)
    prompt = _string(table["prompt"], f"tracks.{track}.prompt")
    prompt_hash = _hash(table["prompt_hash"], f"tracks.{track}.prompt_hash")
    if prompt_hash != f"sha256:{hashlib.sha256(prompt.encode('utf-8')).hexdigest()}":
        raise ContractError(f"tracks.{track}.prompt_hash does not match tracks.{track}.prompt")
    contract_version = _string(table["contract_version"], f"tracks.{track}.contract_version")
    if not contract_version.startswith(f"{track}-caption/"):
        raise ContractError(f"tracks.{track}.contract_version must start with '{track}-caption/'")
    if track == "visual":
        lowered = prompt.casefold()
        for token in ("audio", "sound", "hear"):
            if token in lowered:
                raise ContractError(f"tracks.visual.prompt must not reference audio evidence ({token!r})")
    else:
        lowered = prompt.casefold()
        for token in ("visual", "frame", "image", "see", "watch"):
            if token in lowered:
                raise ContractError(f"tracks.audio.prompt must not reference visual evidence ({token!r})")
    video_frames = (
        _integer(table["video_frames"], f"tracks.{track}.video_frames", minimum=1) if track == "visual" else 0
    )
    return CaptionContract(
        track=track,
        contract_version=contract_version,
        language=language,
        min_words=min_words,
        max_words=max_words,
        min_clip_ms=min_clip,
        max_clip_ms=max_clip,
        prompt=prompt,
        prompt_hash=prompt_hash,
        video_frames=video_frames,
        include_ocr=bool(table.get("include_ocr", False)),
        include_camera_motion=bool(table.get("include_camera_motion", False)),
        transcribe_speech=bool(table.get("transcribe_speech", False)),
    )


def _validate_candidates(value: object) -> None:
    table = _table(
        value,
        "candidates",
        {
            "source_policy",
            "per_clip",
            "challenge_fraction_min",
            "challenge_fraction_max",
            "controlled_error_rate",
            "max_new_tokens",
            "generation_seed",
            "decoding",
        },
    )
    if _string(table["source_policy"], "candidates.source_policy") != "C0":
        raise ContractError("candidates.source_policy must be the frozen collection policy 'C0'")
    _integer(table["generation_seed"], "candidates.generation_seed", minimum=0)
    per_clip = _integer(table["per_clip"], "candidates.per_clip", minimum=4)
    low = _number(table["challenge_fraction_min"], "candidates.challenge_fraction_min", minimum=0.0)
    high = _number(
        table["challenge_fraction_max"], "candidates.challenge_fraction_max", minimum=low, maximum=0.5
    )
    _number(table["controlled_error_rate"], "candidates.controlled_error_rate", minimum=0.0, maximum=0.5)
    _integer(table["max_new_tokens"], "candidates.max_new_tokens", minimum=1)
    mixture = table["decoding"]
    if not isinstance(mixture, list) or not mixture:
        raise ContractError("candidates.decoding must be a non-empty array of tables")
    names: set[str] = set()
    total = 0
    for index, entry in enumerate(mixture):
        location = f"candidates.decoding[{index}]"
        item = _table(entry, location, {"name", "temperature", "top_p", "count"})
        name = _string(item["name"], f"{location}.name")
        if name not in CANDIDATE_SOURCES:
            raise ContractError(f"{location}.name must be one of {sorted(CANDIDATE_SOURCES)}")
        if name in names:
            raise ContractError(f"{location}.name is duplicated")
        names.add(name)
        _number(item["temperature"], f"{location}.temperature", minimum=0.0)
        _number(item["top_p"], f"{location}.top_p", exclusive_minimum=0.0, maximum=1.0)
        total += _integer(item["count"], f"{location}.count", minimum=1)
    if total != per_clip:
        raise ContractError(
            f"candidates.decoding counts sum to {total} but candidates.per_clip is {per_clip}"
        )
    challenge = sum(int(entry["count"]) for entry in mixture if str(entry["name"]) in CHALLENGE_SOURCES)
    fraction = challenge / per_clip
    if not low <= fraction <= high:
        raise ContractError(
            f"challenge candidates are {fraction:.2f} of the pool; contract requires [{low}, {high}]"
        )


def _validate_pairs(value: object) -> None:
    table = _table(
        value,
        "pairs",
        {"per_clip_min", "per_clip_max", "seed", "near_duplicate_jaccard"},
    )
    minimum = _integer(table["per_clip_min"], "pairs.per_clip_min", minimum=1)
    _integer(table["per_clip_max"], "pairs.per_clip_max", minimum=minimum)
    _integer(table["seed"], "pairs.seed", minimum=0)
    _number(table["near_duplicate_jaccard"], "pairs.near_duplicate_jaccard", minimum=0.0, maximum=1.0)


def _validate_annotation(value: object) -> None:
    table = _table(
        value,
        "annotation",
        {
            "judgments_per_pair",
            "repeat_fraction",
            "attention_fraction",
            "collection_version",
            "min_response_ms",
            "max_position_bias",
            "min_attention_pass",
        },
    )
    # Three or more gives a majority that resolves a split; two does not, and
    # `agreement` (modal/total) can then only be 0.5 or 1.0, so any
    # min_agreement above 0.5 means unanimity and every disagreed pair is
    # dropped rather than adjudicated. That is a defensible rule for two expert
    # annotators — a pair they split on is genuinely ambiguous and would train
    # noise — but it makes inter-rater agreement the retention rate exactly, so
    # a contract choosing 2 is accepting a smaller, cleaner view.
    _integer(table["judgments_per_pair"], "annotation.judgments_per_pair", minimum=2)
    _number(table["repeat_fraction"], "annotation.repeat_fraction", minimum=0.0, maximum=0.5)
    _number(table["attention_fraction"], "annotation.attention_fraction", minimum=0.0, maximum=0.5)
    _string(table["collection_version"], "annotation.collection_version")
    _integer(table["min_response_ms"], "annotation.min_response_ms", minimum=0)
    _number(table["max_position_bias"], "annotation.max_position_bias", minimum=0.0, maximum=1.0)
    _number(table["min_attention_pass"], "annotation.min_attention_pass", minimum=0.0, maximum=1.0)


def _validate_views(value: object) -> None:
    table = _table(value, "views", {"sft", "pair_strict", "weighting"})
    sft = _table(table["sft"], "views.sft", {"min_agreement", "min_confidence"})
    _number(sft["min_agreement"], "views.sft.min_agreement", minimum=0.0, maximum=1.0)
    _number(sft["min_confidence"], "views.sft.min_confidence", minimum=1.0, maximum=5.0)
    strict = _table(table["pair_strict"], "views.pair_strict", {"min_agreement", "min_strength"})
    _number(strict["min_agreement"], "views.pair_strict.min_agreement", minimum=0.0, maximum=1.0)
    _number(strict["min_strength"], "views.pair_strict.min_strength", minimum=1.0, maximum=5.0)
    weighting = _table(table["weighting"], "views.weighting", {"strategy", "cap_per_clip"})
    strategy = _string(weighting["strategy"], "views.weighting.strategy")
    if strategy not in {"inverse_pair_count", "cap", "none"}:
        raise ContractError("views.weighting.strategy must be inverse_pair_count, cap, or none")
    _integer(weighting["cap_per_clip"], "views.weighting.cap_per_clip", minimum=1)


def _validate_training(value: object) -> None:
    """The contract is the sole authority for every result-affecting knob.

    The offline runner consumes seeds, precision, learning rates, epochs,
    batch geometry, max_completion_tokens, and gradient_accumulation_steps.
    ``lora`` and ``precision="bf16"`` bind the live QLoRA backend: validated
    and hash-bearing now so no live run can start without a pinned topology,
    consumed when the real training gate is wired.
    """
    table = _table(
        value,
        "training",
        {
            "seeds",
            "canonical_seed",
            "world_size",
            "precision",
            "max_completion_tokens",
            "epochs",
            "learning_rate",
            "sft_learning_rate",
            "batch_size",
            "gradient_accumulation_steps",
            "max_grad_norm",
            "lora",
        },
    )
    seeds = _integers(table["seeds"], "training.seeds")
    if len(seeds) < 3 or len(set(seeds)) != len(seeds):
        raise ContractError("training.seeds must contain at least 3 unique seeds")
    canonical = _integer(table["canonical_seed"], "training.canonical_seed", minimum=0)
    if canonical not in seeds:
        raise ContractError("training.canonical_seed must be one of training.seeds")
    if _integer(table["world_size"], "training.world_size", minimum=1) != 1:
        raise ContractError("training.world_size must be 1 (one process per GPU; DDP is out of scope)")
    if _string(table["precision"], "training.precision") not in {"bf16", "fp32"}:
        raise ContractError("training.precision must be bf16 or fp32")
    _integer(table["max_completion_tokens"], "training.max_completion_tokens", minimum=1)
    _number(table["epochs"], "training.epochs", exclusive_minimum=0.0)
    _number(table["learning_rate"], "training.learning_rate", exclusive_minimum=0.0)
    _number(table["sft_learning_rate"], "training.sft_learning_rate", exclusive_minimum=0.0)
    _integer(table["batch_size"], "training.batch_size", minimum=1)
    _integer(table["gradient_accumulation_steps"], "training.gradient_accumulation_steps", minimum=1)
    _number(table["max_grad_norm"], "training.max_grad_norm", exclusive_minimum=0.0)
    lora = _table(table["lora"], "training.lora", {"rank", "alpha", "dropout", "targets"})
    _integer(lora["rank"], "training.lora.rank", minimum=1)
    _integer(lora["alpha"], "training.lora.alpha", minimum=1)
    _number(lora["dropout"], "training.lora.dropout", minimum=0.0, maximum=1.0)
    targets = lora["targets"]
    if not isinstance(targets, list) or not targets:
        raise ContractError("training.lora.targets must be a non-empty array")
    for index, target in enumerate(targets):
        _string(target, f"training.lora.targets[{index}]")


_OBJECTIVE_KEYS: dict[str, set[str]] = {
    "dpo": {"objective", "beta"},
    "ipo": {"objective", "beta"},
    "cdpo": {"objective", "beta", "epsilon"},
    "rdpo": {"objective", "beta", "epsilon", "epsilon_mode"},
    "drdpo": {"objective", "beta", "beta_prime"},
    "wdpo": {
        "objective",
        "beta",
        "flip_warmup_ratio",
        "flip_budget",
        "tail_quantile",
        "max_tail_cap",
        "cap_floor",
        "enable_correction",
        "enable_winsorization",
        "paper_revision",
        "code_commit",
        "view",
    },
}
_RDPO_EPSILON_MODES = ("estimated", "validation_tuned", "known_synthetic")


def _validate_experiment(experiment_id: str, value: object) -> None:
    spec = EXPERIMENT_MATRIX[experiment_id]
    location = f"experiments.{experiment_id}"
    if spec.objective is None:
        gate = _table(
            value,
            location,
            {"objective", "gate"},
        )
        if _string(gate["objective"], f"{location}.objective") != "none":
            raise ContractError(f"{location}.objective must be 'none' (SEED is never trained)")
        gate_table = _table(
            gate["gate"],
            f"{location}.gate",
            {"max_invalid_output_rate", "min_event_hit_rate", "require_logprob_support"},
        )
        _number(
            gate_table["max_invalid_output_rate"],
            f"{location}.gate.max_invalid_output_rate",
            minimum=0.0,
            maximum=1.0,
        )
        _number(
            gate_table["min_event_hit_rate"], f"{location}.gate.min_event_hit_rate", minimum=0.0, maximum=1.0
        )
        _boolean(gate_table["require_logprob_support"], f"{location}.gate.require_logprob_support")
        return
    if spec.objective == "sft":
        table = _table(value, location, {"objective"})
        if _string(table["objective"], f"{location}.objective") != "sft":
            raise ContractError(f"{location}.objective must be 'sft'")
        return
    keys = _OBJECTIVE_KEYS[spec.objective]
    table = _table(value, location, keys)
    objective = _string(table["objective"], f"{location}.objective")
    if objective != spec.objective:
        raise ContractError(
            f"{location}.objective must be {spec.objective!r}; the matrix assignment is code-owned"
        )
    _number_or_axis(table["beta"], f"{location}.beta", exclusive_minimum=0.0)
    if spec.objective in {"cdpo", "rdpo"}:
        for epsilon in _number_or_axis(table["epsilon"], f"{location}.epsilon", minimum=0.0):
            if epsilon >= 0.5:
                raise ContractError(f"{location}.epsilon must be < 0.5")
    if spec.objective == "rdpo":
        mode = _string(table["epsilon_mode"], f"{location}.epsilon_mode")
        if mode not in _RDPO_EPSILON_MODES:
            raise ContractError(f"{location}.epsilon_mode must be one of {sorted(_RDPO_EPSILON_MODES)}")
    if spec.objective == "drdpo":
        _number_or_axis(table["beta_prime"], f"{location}.beta_prime", exclusive_minimum=0.0)
    if spec.objective == "wdpo":
        _number(table["flip_warmup_ratio"], f"{location}.flip_warmup_ratio", minimum=0.0, maximum=1.0)
        _number(table["flip_budget"], f"{location}.flip_budget", minimum=0.0, maximum=1.0)
        _number(table["tail_quantile"], f"{location}.tail_quantile", exclusive_minimum=0.0, maximum=1.0)
        cap_floor = _number(table["cap_floor"], f"{location}.cap_floor", minimum=0.0, maximum=1.0)
        _number(table["max_tail_cap"], f"{location}.max_tail_cap", minimum=cap_floor, maximum=1.0)
        _boolean(table["enable_correction"], f"{location}.enable_correction")
        _boolean(table["enable_winsorization"], f"{location}.enable_winsorization")
        _string(table["paper_revision"], f"{location}.paper_revision")
        _string(table["code_commit"], f"{location}.code_commit", nonempty=False)
        view = _string(table["view"], f"{location}.view")
        if view not in spec.allowed_views:
            raise ContractError(f"{location}.view must be one of {sorted(spec.allowed_views)}")


def _validate_validation(value: object) -> None:
    table = _table(
        value,
        "validation",
        {"temperature", "top_p", "max_new_tokens", "bootstrap_samples"},
    )
    _number(table["temperature"], "validation.temperature", minimum=0.0)
    _number(table["top_p"], "validation.top_p", exclusive_minimum=0.0, maximum=1.0)
    _integer(table["max_new_tokens"], "validation.max_new_tokens", minimum=1)
    _integer(table["bootstrap_samples"], "validation.bootstrap_samples", minimum=1)


def _validate_robustness(value: object) -> None:
    table = _table(value, "robustness", {"flip_rates", "flip_seed"})
    rates = table["flip_rates"]
    if not isinstance(rates, list) or not rates:
        raise ContractError("robustness.flip_rates must be a non-empty array")
    values = [
        _number(rate, f"robustness.flip_rates[{index}]", minimum=0.0, maximum=0.5)
        for index, rate in enumerate(rates)
    ]
    if len(set(values)) != len(values) or 0.0 not in values:
        raise ContractError("robustness.flip_rates must be unique and include 0.0")
    _integer(table["flip_seed"], "robustness.flip_seed", minimum=0)


def _validate_terminal_states(value: object, execution_class: str) -> None:
    table = _table(value, "terminal_states", set(TERMINAL_STATES))
    for key in TERMINAL_STATES:
        state = _string(table[key], f"terminal_states.{key}")
        if state not in TERMINAL_VALUES:
            raise ContractError(f"terminal_states.{key} must be one of {sorted(TERMINAL_VALUES)}")
    if execution_class == "synthetic_canary" and table["release"] != "blocked_pending_external_operation":
        raise ContractError(
            "terminal_states.release must stay blocked_pending_external_operation"
            " in a synthetic_canary contract"
        )


# ---------------------------------------------------------------------------
# Entry points.
# ---------------------------------------------------------------------------


def validate_contract(document: Mapping[str, Any]) -> StudyContract:
    raw = copy.deepcopy(dict(document))
    root = _table(
        raw,
        "contract",
        {
            "schema_version",
            "study_id",
            "execution_class",
            "corpus",
            "tracks",
            "models",
            "candidates",
            "pairs",
            "annotation",
            "views",
            "training",
            "experiments",
            "validation",
            "robustness",
            "terminal_states",
        },
        {"backends"},
    )
    if _integer(root["schema_version"], "schema_version") != 1:
        raise ContractError("schema_version must be 1")
    _string(root["study_id"], "study_id")
    execution_class = _string(root["execution_class"], "execution_class")
    if execution_class not in EXECUTION_CLASSES:
        raise ContractError(f"execution_class must be one of {sorted(EXECUTION_CLASSES)}")
    _validate_corpus(root["corpus"])
    # A study may collect and train one caption track. Requiring both forced a
    # single-track study to generate, annotate, and train a track it does not
    # report — and for an audio-caption study that means producing captions of
    # the visual scene, which is exactly what such a study excludes. At least
    # one track is still mandatory: a contract with none trains nothing.
    tracks_table = _table(root["tracks"], "tracks", set(), set(TRACKS))
    declared = [track for track in TRACKS if track in tracks_table]
    if not declared:
        raise ContractError(f"tracks: declare at least one of {sorted(TRACKS)}")
    tracks = {track: _validate_track(track, tracks_table[track]) for track in declared}
    models = _table(root["models"], "models", {"seed"})
    _model(models["seed"], "models.seed", with_init_seed=True)
    if "backends" in root:
        _validate_backends(root["backends"])
    _validate_candidates(root["candidates"])
    _validate_pairs(root["pairs"])
    _validate_annotation(root["annotation"])
    _validate_views(root["views"])
    _validate_training(root["training"])
    experiments = _table(root["experiments"], "experiments", set(EXPERIMENT_IDS))
    for experiment_id in EXPERIMENT_IDS:
        _validate_experiment(experiment_id, experiments[experiment_id])
    _validate_validation(root["validation"])
    _validate_robustness(root["robustness"])
    _validate_terminal_states(root["terminal_states"], execution_class)
    return StudyContract(
        raw=raw,
        contract_hash=semantic_hash(raw),
        execution_class=execution_class,
        tracks=tracks,
    )


def load_contract(path: str | Path) -> StudyContract:
    source = Path(path)
    if source.is_symlink():
        raise ContractError("contract path cannot be a symlink")
    try:
        with source.open("rb") as handle:
            document = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ContractError(f"contract is unreadable or invalid TOML: {exc}") from exc
    return validate_contract(document)
