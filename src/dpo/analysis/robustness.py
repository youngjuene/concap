"""Robustness slicing and synthetic flip-rate curves (PRD section 30)."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from dpo.analysis.bradley_terry import AnalysisError
from dpo.evaluation.preference_accuracy import ScoredPair, evaluate_preferences

SliceRule = Callable[[ScoredPair], bool]


def natural_noise_slices() -> dict[str, SliceRule]:
    """The preregistered natural-noise slices computable from pair metadata."""
    return {
        "low_agreement": lambda pair: pair.agreement < 0.7,
        "high_agreement": lambda pair: pair.agreement >= 0.9,
        "near_tie": lambda pair: pair.difficulty >= 0.67,
        "easy": lambda pair: pair.difficulty < 0.33,
    }


def sliced_preference_reports(
    pairs: Sequence[ScoredPair],
    *,
    beta: float,
    rules: Mapping[str, SliceRule] | None = None,
) -> dict[str, dict[str, object]]:
    if not pairs:
        raise AnalysisError("slicing requires at least one scored pair")
    resolved_rules = dict(rules) if rules is not None else natural_noise_slices()
    reports: dict[str, dict[str, object]] = {}
    for name, rule in sorted(resolved_rules.items()):
        members = [pair for pair in pairs if rule(pair)]
        if not members:
            reports[name] = {"pair_count": 0}
            continue
        reports[name] = evaluate_preferences(members, beta=beta).document()
    return reports


@dataclass(frozen=True)
class FlipCurvePoint:
    flip_rate: float
    epsilon_mode: str
    accuracy: float
    log_loss: float

    def document(self) -> dict[str, object]:
        return {
            "flip_rate": self.flip_rate,
            "epsilon_mode": self.epsilon_mode,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
        }


def flip_curve(
    points: Sequence[tuple[float, str, Sequence[ScoredPair]]], *, beta: float
) -> tuple[FlipCurvePoint, ...]:
    """Assemble the flip-rate robustness curve from per-rate validation scorings.

    Each entry is (train-time flip rate, epsilon mode, scored validation pairs).
    Validation labels are never flipped — only the training runs behind each
    point differ — and known-epsilon versus estimated-epsilon conditions are
    reported as separate curve series.
    """
    curve = []
    for flip_rate, epsilon_mode, pairs in points:
        if not 0.0 <= flip_rate <= 0.5:
            raise AnalysisError("flip rates must be in [0, 0.5]")
        report = evaluate_preferences(pairs, beta=beta)
        curve.append(
            FlipCurvePoint(
                flip_rate=flip_rate,
                epsilon_mode=epsilon_mode,
                accuracy=report.accuracy,
                log_loss=report.log_loss,
            )
        )
    return tuple(sorted(curve, key=lambda point: (point.epsilon_mode, point.flip_rate)))
