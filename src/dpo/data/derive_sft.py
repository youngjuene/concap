"""The chosen-only SFT view D_sft.

A candidate becomes an SFT target only when a human outcome positively
endorses it: a clear win, or an explicit both-good/both-acceptable tie. A
"less bad" candidate from a both-unacceptable pair never qualifies, `chosen`
never automatically implies `sft_eligible`, and every gate (confidence,
agreement, audits) is a contract parameter. Candidates endorsed by several
pairs are deduplicated into one example with the strongest consensus weight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from dpo.annotation.aggregate import PairAggregate
from dpo.annotation.raw_annotations import RawAnnotation
from dpo.candidates.audit import ResolvedCandidateAudit
from dpo.candidates.freeze import FrozenCandidatePool
from dpo.core.identity import semantic_hash
from dpo.data.derive_pairs import ViewError
from dpo.data.split import SplitManifest

_ACCEPTABLE_TIE_SUBTYPES = frozenset({"both_good", "both_acceptable"})


@dataclass(frozen=True)
class SftExample:
    example_id: str
    clip_id: str
    track: str
    candidate_id: str
    completion: str
    consensus_weight: float

    def document(self) -> dict[str, object]:
        return {
            "example_id": self.example_id,
            "clip_id": self.clip_id,
            "track": self.track,
            "candidate_id": self.candidate_id,
            "completion": self.completion,
            "consensus_weight": self.consensus_weight,
        }


def _acceptable_tie_support(annotations: Sequence[RawAnnotation], aggregate: PairAggregate) -> bool:
    """True when the pair's modal outcome is an explicitly acceptable tie."""
    if aggregate.modal_choice != "tie":
        return False
    tie_rows = [
        annotation
        for annotation in annotations
        if annotation.pair_id == aggregate.pair_id
        and annotation.repeat_of is None
        and not annotation.is_attention_check
        and annotation.canonical_choice() == "tie"
    ]
    subtyped = [row for row in tie_rows if row.tie_subtype is not None]
    if not subtyped:
        return False
    return all(row.tie_subtype in _ACCEPTABLE_TIE_SUBTYPES for row in subtyped)


def derive_sft(
    aggregates: Sequence[PairAggregate],
    annotations: Sequence[RawAnnotation],
    pool: FrozenCandidatePool,
    manifest: SplitManifest,
    audits: Mapping[str, ResolvedCandidateAudit],
    *,
    split: str,
    min_agreement: float,
    min_confidence: float,
) -> tuple[SftExample, ...]:
    support: dict[str, float] = {}
    for aggregate in aggregates:
        if manifest.role_of(aggregate.clip_id) != split:
            raise ViewError(
                f"pair {aggregate.pair_id!r} belongs to split"
                f" {manifest.role_of(aggregate.clip_id)!r}, not {split!r}"
            )
        if aggregate.modal_choice == "both_unacceptable" or aggregate.both_bad_count > 0:
            # A less-bad output from a both-unacceptable pair is never an SFT target.
            continue
        if aggregate.agreement < min_agreement or aggregate.mean_confidence < min_confidence:
            continue
        weight = aggregate.agreement * (aggregate.mean_confidence / 5.0)
        endorsed: tuple[str, ...]
        winner = aggregate.winner_id
        if winner is not None:
            endorsed = (winner,)
        elif _acceptable_tie_support(annotations, aggregate):
            endorsed = (aggregate.candidate_a, aggregate.candidate_b)
        else:
            continue
        for candidate_id in endorsed:
            audit = audits.get(candidate_id)
            if audit is None:
                raise ViewError(f"candidate {candidate_id!r} has no resolved audit")
            if not audit.acceptable or audit.critical_hallucination or audit.modality_violation:
                continue
            support[candidate_id] = max(support.get(candidate_id, 0.0), weight)
    examples = []
    for candidate_id in sorted(support):
        candidate = pool.candidate(candidate_id)
        suffix = semantic_hash({"candidate_id": candidate_id}).removeprefix("sha256:")[:12]
        examples.append(
            SftExample(
                example_id=f"sft-{suffix}",
                clip_id=candidate.clip_id,
                track=candidate.track,
                candidate_id=candidate_id,
                completion=candidate.text,
                consensus_weight=support[candidate_id],
            )
        )
    return tuple(examples)
