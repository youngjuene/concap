"""Offline pairwise preference evaluation (PRD section 27.1).

For each held-out pair, p_hat = sigmoid(beta * h_theta). Reports accuracy,
pairwise log loss, calibration error, the margin distribution, and accuracy
sliced by pair difficulty, agreement bucket, and track.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoredPair:
    """One evaluation pair with completion logps from policy and reference."""

    pair_id: str
    clip_id: str
    track: str
    policy_chosen_logp: float
    policy_rejected_logp: float
    ref_chosen_logp: float
    ref_rejected_logp: float
    difficulty: float
    agreement: float

    @property
    def margin(self) -> float:
        return (self.policy_chosen_logp - self.ref_chosen_logp) - (
            self.policy_rejected_logp - self.ref_rejected_logp
        )


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponent = math.exp(value)
    return exponent / (1.0 + exponent)


def _difficulty_bucket(difficulty: float) -> str:
    if difficulty >= 0.67:
        return "hard"
    if difficulty >= 0.33:
        return "medium"
    return "easy"


def _agreement_bucket(agreement: float) -> str:
    if agreement >= 0.9:
        return "high"
    if agreement >= 0.7:
        return "medium"
    return "low"


@dataclass(frozen=True)
class PreferenceReport:
    pair_count: int
    accuracy: float
    log_loss: float
    calibration_error: float
    margin_mean: float
    margin_std: float
    accuracy_by_difficulty: dict[str, float]
    accuracy_by_agreement: dict[str, float]
    accuracy_by_track: dict[str, float]

    def document(self) -> dict[str, object]:
        return {
            "schema": "dpo.preference-report/v1",
            "pair_count": self.pair_count,
            "accuracy": self.accuracy,
            "log_loss": self.log_loss,
            "calibration_error": self.calibration_error,
            "margin_mean": self.margin_mean,
            "margin_std": self.margin_std,
            "accuracy_by_difficulty": dict(self.accuracy_by_difficulty),
            "accuracy_by_agreement": dict(self.accuracy_by_agreement),
            "accuracy_by_track": dict(self.accuracy_by_track),
        }


def evaluate_preferences(pairs: Sequence[ScoredPair], *, beta: float, bins: int = 10) -> PreferenceReport:
    if not pairs:
        raise ValueError("preference evaluation requires at least one pair")
    if beta <= 0:
        raise ValueError("beta must be positive")
    probabilities = []
    margins = []
    correct_flags = []
    grouped: dict[str, dict[str, list[bool]]] = {
        "difficulty": {},
        "agreement": {},
        "track": {},
    }
    log_loss = 0.0
    for pair in pairs:
        margin = pair.margin
        probability = _sigmoid(beta * margin)
        # Guard the log against exact 0/1 saturation.
        clamped = min(max(probability, 1e-12), 1.0 - 1e-12)
        log_loss -= math.log(clamped)
        correct = margin > 0
        probabilities.append(probability)
        margins.append(margin)
        correct_flags.append(correct)
        grouped["difficulty"].setdefault(_difficulty_bucket(pair.difficulty), []).append(correct)
        grouped["agreement"].setdefault(_agreement_bucket(pair.agreement), []).append(correct)
        grouped["track"].setdefault(pair.track, []).append(correct)
    count = len(pairs)
    mean_margin = sum(margins) / count
    variance = sum((margin - mean_margin) ** 2 for margin in margins) / count
    # Expected calibration error over equal-width probability bins: the model
    # "predicts" the chosen response with probability p_hat, and the observed
    # outcome is whether the margin actually favored the chosen response.
    calibration = 0.0
    for index in range(bins):
        low = index / bins
        high = (index + 1) / bins
        members = [
            (probability, correct)
            for probability, correct in zip(probabilities, correct_flags, strict=True)
            if low <= probability < high or (index == bins - 1 and probability == 1.0)
        ]
        if not members:
            continue
        mean_probability = sum(probability for probability, _ in members) / len(members)
        mean_outcome = sum(1.0 for _, correct in members if correct) / len(members)
        calibration += (len(members) / count) * abs(mean_probability - mean_outcome)
    return PreferenceReport(
        pair_count=count,
        accuracy=sum(correct_flags) / count,
        log_loss=log_loss / count,
        calibration_error=calibration,
        margin_mean=mean_margin,
        margin_std=math.sqrt(variance),
        accuracy_by_difficulty={
            bucket: sum(flags) / len(flags) for bucket, flags in sorted(grouped["difficulty"].items())
        },
        accuracy_by_agreement={
            bucket: sum(flags) / len(flags) for bucket, flags in sorted(grouped["agreement"].items())
        },
        accuracy_by_track={
            bucket: sum(flags) / len(flags) for bucket, flags in sorted(grouped["track"].items())
        },
    )
