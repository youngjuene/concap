"""The memorization gate: a validation caption must not be a training candidate."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from dpo.candidates.freeze import FrozenCandidatePool


@dataclass(frozen=True)
class GeneratedCaption:
    clip_id: str
    text: str


def training_candidate_reuse_rate(captions: Sequence[GeneratedCaption], pool: FrozenCandidatePool) -> float:
    """Fraction of generated captions that byte-match a frozen training candidate.

    Generated validation/test captions must be fresh model outputs; reuse of a
    training candidate indicates memorization or an evaluation wiring bug.
    """
    if not captions:
        raise ValueError("reuse rate requires at least one caption")
    frozen_texts = {candidate.text.strip().casefold() for candidate in pool.candidates}
    reused = sum(1 for caption in captions if caption.text.strip().casefold() in frozen_texts)
    return reused / len(captions)
