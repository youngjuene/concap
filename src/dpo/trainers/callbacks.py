"""Step-level diagnostics collection with a stable, append-only schema."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

REQUIRED_PREFERENCE_KEYS = frozenset(
    {
        "policy_chosen_logp_mean",
        "policy_rejected_logp_mean",
        "ref_chosen_logp_mean",
        "ref_rejected_logp_mean",
        "margin_mean",
        "preference_accuracy",
        "per_example_loss_mean",
        "kl_proxy_chosen",
        "kl_proxy_rejected",
        "sample_weight_sum",
        "nan_count",
        "inf_count",
        "batch_size",
        "grad_norm",
        "loss",
        "caption_length_chosen_mean",
        "caption_length_rejected_mean",
        "pair_difficulty_mean",
        "agreement_bucket",
        "track",
        "global_step",
    }
)


class DiagnosticsError(ValueError):
    """Raised when a trainer step omits required diagnostics."""


class DiagnosticsLog:
    """Collects per-step diagnostics; optionally appends JSONL to disk."""

    def __init__(self, *, path: str | Path | None = None, required: frozenset[str] = frozenset()) -> None:
        self.path = None if path is None else Path(path)
        self.required = required
        self.steps: list[dict[str, object]] = []
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, step: Mapping[str, object]) -> None:
        missing = sorted(self.required - set(step))
        if missing:
            raise DiagnosticsError(f"diagnostics step is missing required key {missing[0]!r}")
        for key, value in step.items():
            if isinstance(value, float) and not math.isfinite(value):
                raise DiagnosticsError(f"diagnostics key {key!r} is non-finite")
        document = dict(step)
        self.steps.append(document)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(document, ensure_ascii=False, sort_keys=True) + "\n")

    def summary(self) -> dict[str, object]:
        if not self.steps:
            return {"steps": 0}
        numeric: dict[str, list[float]] = {}
        for step in self.steps:
            for key, value in step.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    numeric.setdefault(key, []).append(float(value))
        return {
            "steps": len(self.steps),
            "means": {key: sum(values) / len(values) for key, values in sorted(numeric.items())},
            "last": dict(self.steps[-1]),
        }
