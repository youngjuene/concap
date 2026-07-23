"""Candidate records and the frozen collection policy C0."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dpo.contracts.study_contract import CANDIDATE_SOURCES, TRACKS
from dpo.core.identity import semantic_hash
from dpo.core.textsafety import UntrustedTextError, validate_untrusted_text

HASH_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")


class CandidateError(ValueError):
    """Raised when a candidate record or pool violates the construction contract."""


@dataclass(frozen=True)
class GenerationConfig:
    temperature: float
    top_p: float
    max_new_tokens: int
    seed: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise CandidateError("generation temperature must be a finite non-negative number")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise CandidateError("generation top_p must be in (0, 1]")
        if self.max_new_tokens < 1:
            raise CandidateError("generation max_new_tokens must be positive")
        if self.seed < 0:
            raise CandidateError("generation seed must be non-negative")

    def document(self) -> dict[str, object]:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_new_tokens": self.max_new_tokens,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class CollectionPolicy:
    """The frozen candidate-generation policy C0.

    C0 is distinct from the experimental training role. It is normally the
    task-capable seed model SEED; if SEED was bootstrapped, the bootstrap
    corpus must not contain preference-derived chosen responses from this
    experiment (that provenance lives in the checkpoint's own artifact
    lineage).
    """

    policy_id: str
    checkpoint_hash: str

    def __post_init__(self) -> None:
        if self.policy_id != "C0":
            raise CandidateError("the collection policy id must be 'C0'")
        if not HASH_RE.fullmatch(self.checkpoint_hash):
            raise CandidateError("collection policy checkpoint hash must be a sha256 hash")


@dataclass(frozen=True)
class CandidateRecord:
    candidate_id: str
    clip_id: str
    track: str
    text: str
    source_policy: str
    source_kind: str
    checkpoint_hash: str
    generation_config: GenerationConfig

    def __post_init__(self) -> None:
        if self.track not in TRACKS:
            raise CandidateError(f"candidate track must be one of {sorted(TRACKS)}")
        if not self.candidate_id.strip() or not self.clip_id.strip():
            raise CandidateError("candidate requires candidate_id and clip_id")
        if self.source_policy != "C0":
            raise CandidateError("candidates must come from the frozen collection policy C0")
        if self.source_kind not in CANDIDATE_SOURCES:
            raise CandidateError(f"candidate source kind must be one of {sorted(CANDIDATE_SOURCES)}")
        if not HASH_RE.fullmatch(self.checkpoint_hash):
            raise CandidateError("candidate checkpoint hash must be a sha256 hash")
        try:
            validate_untrusted_text(self.text, field=f"candidate.{self.candidate_id}.text")
        except UntrustedTextError as exc:
            raise CandidateError(str(exc)) from exc

    @property
    def candidate_hash(self) -> str:
        return semantic_hash(
            {
                "candidate_id": self.candidate_id,
                "clip_id": self.clip_id,
                "track": self.track,
                "text": self.text,
                "source_policy": self.source_policy,
                "source_kind": self.source_kind,
                "checkpoint_hash": self.checkpoint_hash,
                "generation_config": self.generation_config.document(),
            }
        )

    def document(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "clip_id": self.clip_id,
            "track": self.track,
            "text": self.text,
            "source_policy": self.source_policy,
            "source_kind": self.source_kind,
            "checkpoint_hash": self.checkpoint_hash,
            "generation_config": self.generation_config.document(),
            "candidate_hash": self.candidate_hash,
        }


def build_candidate_records(
    *,
    clip_id: str,
    track: str,
    policy: CollectionPolicy,
    generations: Sequence[tuple[str, str, GenerationConfig]],
) -> tuple[CandidateRecord, ...]:
    """Assemble candidate records for one clip from (source_kind, text, config) tuples.

    Candidate ids are content-derived so a regenerated pool with identical text
    keeps identical ids, and any drift changes them.
    """
    if len(generations) < 4:
        raise CandidateError("candidate construction requires at least four candidates per clip")
    records = []
    seen_texts: set[str] = set()
    for source_kind, text, config in generations:
        normalized = text.strip()
        if normalized in seen_texts:
            raise CandidateError(f"clip {clip_id!r}: duplicate candidate text {normalized!r}")
        seen_texts.add(normalized)
        suffix = semantic_hash(
            {
                "clip_id": clip_id,
                "track": track,
                "text": normalized,
                "source_kind": source_kind,
                "config": config.document(),
            }
        ).removeprefix("sha256:")[:12]
        records.append(
            CandidateRecord(
                candidate_id=f"cand-{suffix}",
                clip_id=clip_id,
                track=track,
                text=normalized,
                source_policy=policy.policy_id,
                source_kind=source_kind,
                checkpoint_hash=policy.checkpoint_hash,
                generation_config=config,
            )
        )
    return tuple(records)


def validate_pool_mixture(
    records: Sequence[CandidateRecord],
    *,
    per_clip: int,
    challenge_fraction_min: float,
    challenge_fraction_max: float,
) -> None:
    """Enforce the per-clip count and natural/challenge composition of the pool."""
    from dpo.contracts.study_contract import CHALLENGE_SOURCES

    by_clip: dict[str, list[CandidateRecord]] = {}
    for record in records:
        by_clip.setdefault(record.clip_id, []).append(record)
    for clip_id, clip_records in sorted(by_clip.items()):
        if len(clip_records) != per_clip:
            raise CandidateError(
                f"clip {clip_id!r} has {len(clip_records)} candidates; the contract requires {per_clip}"
            )
        challenge = sum(1 for record in clip_records if record.source_kind in CHALLENGE_SOURCES)
        fraction = challenge / len(clip_records)
        if not challenge_fraction_min <= fraction <= challenge_fraction_max:
            raise CandidateError(
                f"clip {clip_id!r}: challenge candidates are {fraction:.2f} of the pool;"
                f" the contract requires [{challenge_fraction_min}, {challenge_fraction_max}]"
            )


def parse_candidate_record(value: Mapping[str, object]) -> CandidateRecord:
    expected = {
        "candidate_id",
        "clip_id",
        "track",
        "text",
        "source_policy",
        "source_kind",
        "checkpoint_hash",
        "generation_config",
        "candidate_hash",
    }
    if set(value) != expected:
        raise CandidateError("candidate record has an unexpected field set")
    config_value = value["generation_config"]
    if not isinstance(config_value, Mapping) or set(config_value) != {
        "temperature",
        "top_p",
        "max_new_tokens",
        "seed",
    }:
        raise CandidateError("candidate generation_config has an unexpected field set")
    temperature = config_value["temperature"]
    top_p = config_value["top_p"]
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise CandidateError("candidate generation temperature must be a number")
    if not isinstance(top_p, (int, float)) or isinstance(top_p, bool):
        raise CandidateError("candidate generation top_p must be a number")
    if type(config_value["max_new_tokens"]) is not int or type(config_value["seed"]) is not int:
        raise CandidateError("candidate generation token/seed values must be integers")
    record = CandidateRecord(
        candidate_id=str(value["candidate_id"]),
        clip_id=str(value["clip_id"]),
        track=str(value["track"]),
        text=str(value["text"]),
        source_policy=str(value["source_policy"]),
        source_kind=str(value["source_kind"]),
        checkpoint_hash=str(value["checkpoint_hash"]),
        generation_config=GenerationConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(config_value["max_new_tokens"]),
            seed=int(config_value["seed"]),
        ),
    )
    if value["candidate_hash"] != record.candidate_hash:
        raise CandidateError(f"candidate {record.candidate_id!r} hash does not match its content")
    return record
