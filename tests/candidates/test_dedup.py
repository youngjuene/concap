"""Cross-pool dedup: the leakage gate enforced before annotation, not after."""

from __future__ import annotations

import pytest

from dpo.candidates.candidate_records import (
    CandidateError,
    CollectionPolicy,
    GenerationConfig,
    build_candidate_records,
)
from dpo.candidates.dedup import cross_pool_dedup
from dpo.candidates.freeze import freeze_pool
from dpo.core.identity import sha256_bytes

_POLICY = CollectionPolicy(policy_id="C0", checkpoint_hash=sha256_bytes(b"c0"))
_CONFIG = GenerationConfig(temperature=0.7, top_p=0.9, max_new_tokens=48, seed=1)


def _pool(clip_texts: dict[str, list[str]], *, version: str):
    records = []
    for clip_id, texts in clip_texts.items():
        records.extend(
            build_candidate_records(
                clip_id=clip_id,
                track="visual",
                policy=_POLICY,
                generations=[
                    ("greedy" if index == 0 else "sample", text, _CONFIG)
                    for index, text in enumerate(texts)
                ],
            )
        )
    return freeze_pool(
        dataset_version=version,
        track="visual",
        evidence_audit_version="audit/v1",
        candidates=tuple(records),
        pairs=(),
    )


def test_cross_pool_dedup_drops_both_sides_of_a_collision() -> None:
    leak = "A person is speaking into a microphone in a quiet room."
    near = "A person is speaking into a microphone in a room."
    pool_a = _pool(
        {
            "clip-a1": [near, "Traffic hums along a wet boulevard.", "Rain patters on the awnings.", "Extra one."],
            "clip-a2": ["Bicycles rattle over bricks.", "A tram bell rings twice.", "Gulls cry overhead.", "Extra two."],
        },
        version="train/v1",
    )
    pool_b = _pool(
        {"clip-b1": [leak, "A vendor calls out prices.", "Coins clink on a metal tray.", "Extra three."]},
        version="validation/v1",
    )
    outcome = cross_pool_dedup(pool_a, pool_b, per_clip_min=3)
    dropped_texts = {row[3] for row in outcome.dropped}
    assert dropped_texts == {leak, near}, "both sides of the collision must be cut"
    assert len(outcome.kept_a) == 7
    assert len(outcome.kept_b) == 3


def test_cross_pool_dedup_refuses_to_gut_a_clip() -> None:
    # clip-a loses TWO of four candidates to two different collisions, landing
    # at 2 < per_clip_min: the transform must refuse rather than publish it.
    same_one = "The very same generic caption text."
    same_two = "Another equally generic caption text."
    pool_a = _pool(
        {"clip-a": [same_one, same_two, "One alternative caption here.", "A fourth caption text."]},
        version="t/v1",
    )
    pool_b = _pool(
        {
            "clip-b1": [same_one, "A vendor calls out prices.", "Coins clink on a tray.", "Filler b1."],
            "clip-b2": [same_two, "A dog barks at a scooter.", "Wind rattles the shutters.", "Filler b2."],
        },
        version="v/v1",
    )
    with pytest.raises(CandidateError, match="below per_clip_min"):
        cross_pool_dedup(pool_a, pool_b, per_clip_min=3)


def test_cross_pool_dedup_rejects_pools_sharing_a_clip() -> None:
    pool_a = _pool(
        {"clip-x": ["Alpha text one.", "Beta text two.", "Gamma text three.", "Delta text extra."]},
        version="t/v1",
    )
    pool_b = _pool(
        {"clip-x": ["Epsilon text four.", "Zeta text five.", "Eta text six.", "Theta text extra."]},
        version="v/v1",
    )
    with pytest.raises(CandidateError, match="appears in both pools"):
        cross_pool_dedup(pool_a, pool_b, per_clip_min=3)
