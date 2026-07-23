"""Clip-clustered bootstrap and multiple-comparison correction."""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import TypeVar

from dpo.analysis.bradley_terry import AnalysisError

Row = TypeVar("Row")


def clip_cluster_bootstrap(
    rows: Sequence[Row],
    clip_of: Callable[[Row], str],
    statistic: Callable[[Sequence[Row]], float],
    *,
    samples: int,
    seed: int,
    confidence: float = 0.95,
) -> tuple[float, tuple[float, float]]:
    """Percentile CI for a statistic, resampling whole clips with replacement."""
    if not rows:
        raise AnalysisError("bootstrap requires at least one row")
    if not 0.5 < confidence < 1.0:
        raise AnalysisError("confidence must be in (0.5, 1)")
    if samples < 1:
        raise AnalysisError("bootstrap requires at least one sample")
    by_clip: dict[str, list[Row]] = {}
    for row in rows:
        by_clip.setdefault(clip_of(row), []).append(row)
    clips = sorted(by_clip)
    point = statistic(rows)
    generator = random.Random(seed)
    replicates = []
    for _ in range(samples):
        resampled: list[Row] = []
        for _ in clips:
            resampled.extend(by_clip[generator.choice(clips)])
        replicates.append(statistic(resampled))
    replicates.sort()
    lower_index = int(((1.0 - confidence) / 2.0) * (samples - 1))
    upper_index = int((1.0 - (1.0 - confidence) / 2.0) * (samples - 1))
    return point, (replicates[lower_index], replicates[upper_index])


def benjamini_hochberg(p_values: dict[str, float], *, alpha: float = 0.05) -> dict[str, bool]:
    """BH step-up procedure: name -> rejected (significant after correction)."""
    if not p_values:
        raise AnalysisError("BH correction requires at least one p-value")
    for name, value in p_values.items():
        if not 0.0 <= value <= 1.0:
            raise AnalysisError(f"p-value {name!r} is outside [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    count = len(ordered)
    threshold_rank = 0
    for rank, (_, value) in enumerate(ordered, start=1):
        if value <= alpha * rank / count:
            threshold_rank = rank
    return {name: rank <= threshold_rank for rank, (name, _) in enumerate(ordered, start=1)}
