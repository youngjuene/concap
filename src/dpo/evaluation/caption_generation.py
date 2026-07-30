"""Generated-caption validation under the frozen decoding configuration."""

from __future__ import annotations

from collections.abc import Sequence

from dpo.candidates.freeze import FrozenCandidatePool
from dpo.evaluation.compliance import GeneratedCaption
from dpo.models.base import MediaBatch, ModelAdapter


def generate_captions(
    adapter: ModelAdapter,
    clip_ids: Sequence[str],
    media: MediaBatch,
    *,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    seed: int,
) -> tuple[GeneratedCaption, ...]:
    """Generate one caption per clip with one frozen decoding configuration.

    Every model in the comparison must be called with the same decoding budget;
    the caller passes the contract's validation decoding verbatim.
    """
    if tuple(media.clip_ids) != tuple(clip_ids):
        raise ValueError("media batch clips do not match the requested clip list")
    # Eval mode: a validation caption must come from the weights as they will be
    # shipped, so dropout must not perturb the decode. The comparison across
    # variants is only fair if every one of them is sampled the same way.
    with adapter.module_mode(training=False):
        texts = adapter.generate(
            media,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            seed=seed,
        )
    return tuple(
        GeneratedCaption(clip_id=clip_id, text=text) for clip_id, text in zip(clip_ids, texts, strict=True)
    )


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
