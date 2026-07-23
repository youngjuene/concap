"""Clip-level weighting so pair-rich clips cannot dominate training."""

from __future__ import annotations

from collections.abc import Sequence

from dpo.core.identity import semantic_hash
from dpo.data.derive_pairs import StrictPair, ViewError


def pair_weights(
    rows: Sequence[StrictPair],
    *,
    strategy: str,
    cap_per_clip: int,
    seed: int = 0,
) -> dict[str, float]:
    """Map pair_id -> weight. Pairs absent from the result are dropped (cap strategy)."""
    by_clip: dict[str, list[StrictPair]] = {}
    for row in rows:
        by_clip.setdefault(row.clip_id, []).append(row)
    weights: dict[str, float] = {}
    if strategy == "none":
        for row in rows:
            weights[row.pair_id] = 1.0
        return weights
    if strategy == "inverse_pair_count":
        for clip_rows in by_clip.values():
            weight = 1.0 / len(clip_rows)
            for row in clip_rows:
                weights[row.pair_id] = weight
        return weights
    if strategy == "cap":
        if cap_per_clip < 1:
            raise ViewError("cap_per_clip must be at least 1")
        for clip_id, clip_rows in by_clip.items():
            kept = sorted(
                clip_rows,
                key=lambda row: semantic_hash({"seed": seed, "clip": clip_id, "pair": row.pair_id}),
            )[:cap_per_clip]
            for row in kept:
                weights[row.pair_id] = 1.0
        return weights
    raise ViewError(f"unknown weighting strategy {strategy!r}")
