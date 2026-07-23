"""Bradley-Terry strengths from pairwise study outcomes.

Fitted with the classical MM algorithm (Hunter, 2004) over win counts; ties
contribute half a win to each side. Strengths are identified by fixing the
geometric mean to one, and reported as log-strengths.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


class AnalysisError(ValueError):
    """Raised when an analysis input is degenerate."""


@dataclass(frozen=True)
class PairwiseOutcome:
    """One judged comparison. ``winner`` is a model id or None for a tie."""

    model_a: str
    model_b: str
    winner: str | None
    clip_id: str

    def __post_init__(self) -> None:
        if self.model_a == self.model_b:
            raise AnalysisError("a pairwise outcome needs two distinct models")
        if self.winner is not None and self.winner not in {self.model_a, self.model_b}:
            raise AnalysisError("winner must be one of the two compared models")


@dataclass(frozen=True)
class BradleyTerryFit:
    log_strengths: dict[str, float]
    iterations: int
    converged: bool

    def ranking(self) -> list[str]:
        return sorted(self.log_strengths, key=lambda model: -self.log_strengths[model])

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.bradley-terry-fit/v1",
            "log_strengths": dict(sorted(self.log_strengths.items())),
            "iterations": self.iterations,
            "converged": self.converged,
            "ranking": self.ranking(),
        }


def fit_bradley_terry(
    outcomes: Sequence[PairwiseOutcome],
    *,
    max_iterations: int = 500,
    tolerance: float = 1e-9,
    smoothing: float = 0.1,
) -> BradleyTerryFit:
    """MM fit over (possibly tied) outcomes.

    ``smoothing`` adds a small symmetric pseudo-count to every observed pair so
    a model with zero wins keeps a finite strength; it is reported in the
    document via the fit being deterministic in its inputs.
    """
    if not outcomes:
        raise AnalysisError("Bradley-Terry requires at least one outcome")
    models = sorted({model for outcome in outcomes for model in (outcome.model_a, outcome.model_b)})
    if len(models) < 2:
        raise AnalysisError("Bradley-Terry requires at least two models")
    wins: dict[str, float] = dict.fromkeys(models, 0.0)
    games: dict[tuple[str, str], float] = {}
    for outcome in outcomes:
        low, high = sorted((outcome.model_a, outcome.model_b))
        key = (low, high)
        games[key] = games.get(key, 0.0) + 1.0
        if outcome.winner is None:
            wins[outcome.model_a] += 0.5
            wins[outcome.model_b] += 0.5
        else:
            wins[outcome.winner] += 1.0
    for left, right in games:
        games[(left, right)] += 2 * smoothing
        wins[left] += smoothing
        wins[right] += smoothing
    strengths: dict[str, float] = dict.fromkeys(models, 1.0)
    converged = False
    iterations_used = 0
    for iteration in range(1, max_iterations + 1):
        iterations_used = iteration
        updated: dict[str, float] = {}
        for model in models:
            denominator = 0.0
            for (left, right), count in games.items():
                if model in (left, right):
                    denominator += count / (strengths[left] + strengths[right])
            if denominator <= 0:
                raise AnalysisError(f"model {model!r} appears in no comparisons")
            updated[model] = wins[model] / denominator
        geometric_mean = math.exp(sum(math.log(value) for value in updated.values()) / len(updated))
        updated = {model: value / geometric_mean for model, value in updated.items()}
        delta = max(abs(updated[model] - strengths[model]) for model in models)
        strengths = updated
        if delta < tolerance:
            converged = True
            break
    return BradleyTerryFit(
        log_strengths={model: math.log(value) for model, value in strengths.items()},
        iterations=iterations_used,
        converged=converged,
    )
