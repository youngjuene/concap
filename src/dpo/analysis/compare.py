"""The inferential comparison over a published validation report.

``dpo report show`` prints raw per-variant accuracy; this module is the layer a
paper can cite: clip-clustered bootstrap CIs per experiment, exact paired tests
against SEED with Benjamini-Hochberg correction across experiments, a
Bradley-Terry fit over per-pair contests between the selected variants, and the
preregistered natural-noise slices for the ranked winner.

Everything is computed from the per-pair scores the validation report persists;
nothing here re-scores a model, so the analysis is exactly reproducible from
the published artifacts alone.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from dpo.analysis.bootstrap import benjamini_hochberg, clip_cluster_bootstrap
from dpo.analysis.bradley_terry import AnalysisError, PairwiseOutcome, fit_bradley_terry
from dpo.analysis.robustness import sliced_preference_reports
from dpo.evaluation.preference_accuracy import ScoredPair
from dpo.pipeline.experiments import compute_report_fields

DEFAULT_BOOTSTRAP_SAMPLES = 200


def _scored(track: str, rows: Sequence[Mapping[str, Any]]) -> list[ScoredPair]:
    return [
        ScoredPair(
            pair_id=str(row["pair_id"]),
            clip_id=str(row["clip_id"]),
            track=track,
            policy_chosen_logp=float(row["policy_chosen_logp"]),
            policy_rejected_logp=float(row["policy_rejected_logp"]),
            ref_chosen_logp=float(row["ref_chosen_logp"]),
            ref_rejected_logp=float(row["ref_rejected_logp"]),
            difficulty=float(row["difficulty"]),
            agreement=float(row["agreement"]),
        )
        for row in rows
    ]


def _exact_binomial_two_sided(successes: int, trials: int) -> float:
    """Exact two-sided sign-test p-value at p=0.5 (the paired McNemar case)."""
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, k) for k in range(0, min(successes, trials - successes) + 1))
    return min(1.0, 2.0 * tail / 2.0**trials)


def compare_experiments(
    validation: Mapping[str, Any],
    selection: Mapping[str, Any],
    *,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, object]:
    """The comparison document, per track, from published payloads alone."""
    scores = validation.get("scores")
    if not isinstance(scores, Mapping):
        raise AnalysisError(
            "this validation report carries no per-pair scores; it predates score"
            " persistence — re-run `dpo select run` to publish one that does"
        )
    document: dict[str, object] = {"schema": "dpo.analysis-report/v1", "tracks": {}}
    for track, ranking in sorted(selection["ranking"].items()):
        selected = selection["selected_variants"][track]
        hyper = selection["selected_hyperparameters"][track]
        rows_of: dict[str, list[ScoredPair]] = {}
        for experiment_id in ranking:
            variant_id = selected[experiment_id]
            rows_of[experiment_id] = _scored(track, scores[track][experiment_id][variant_id])

        experiments: dict[str, dict[str, object]] = {}
        p_values: dict[str, float] = {}
        seed_correct = {pair.pair_id: pair.margin > 0 for pair in rows_of.get("SEED", [])}
        for experiment_id in ranking:
            rows = rows_of[experiment_id]
            point, interval = clip_cluster_bootstrap(
                rows,
                lambda pair: pair.clip_id,
                lambda members: sum(1 for pair in members if pair.margin > 0) / len(members),
                samples=bootstrap_samples,
                seed=seed,
            )
            entry: dict[str, object] = {
                "variant_id": selected[experiment_id],
                "accuracy": point,
                "accuracy_ci95": list(interval),
                **compute_report_fields(experiment_id),
            }
            if experiment_id != "SEED" and seed_correct:
                paired = [
                    (seed_correct[pair.pair_id], pair.margin > 0)
                    for pair in rows
                    if pair.pair_id in seed_correct
                ]
                wins = sum(1 for was, now in paired if now and not was)
                losses = sum(1 for was, now in paired if was and not now)
                p_value = _exact_binomial_two_sided(wins, wins + losses)
                entry["vs_seed"] = {"wins": wins, "losses": losses, "p_value": p_value}
                p_values[experiment_id] = p_value
            experiments[experiment_id] = entry
        if p_values:
            rejected = benjamini_hochberg(p_values, alpha=alpha)
            for experiment_id, significant in rejected.items():
                entry = experiments[experiment_id]
                assert isinstance(entry["vs_seed"], dict)
                entry["vs_seed"]["bh_significant"] = significant

        outcomes: list[PairwiseOutcome] = []
        for index, left_id in enumerate(ranking):
            for right_id in ranking[index + 1 :]:
                left_by_pair = {pair.pair_id: pair for pair in rows_of[left_id]}
                for right in rows_of[right_id]:
                    left = left_by_pair.get(right.pair_id)
                    if left is None:
                        continue
                    left_ok, right_ok = left.margin > 0, right.margin > 0
                    winner = (
                        left_id
                        if left_ok and not right_ok
                        else right_id
                        if right_ok and not left_ok
                        else None
                    )
                    outcomes.append(
                        PairwiseOutcome(
                            model_a=left_id, model_b=right_id, winner=winner, clip_id=right.clip_id
                        )
                    )
        bradley_terry = fit_bradley_terry(outcomes)

        top = ranking[0]
        top_beta = float(hyper[top].get("beta", 1.0)) if isinstance(hyper.get(top), Mapping) else 1.0
        document["tracks"][track] = {  # type: ignore[index]
            "experiments": experiments,
            "bradley_terry": bradley_terry.document(),
            "top_experiment": top,
            "top_slices": sliced_preference_reports(rows_of[top], beta=top_beta),
            "bootstrap_samples": bootstrap_samples,
            "alpha": alpha,
        }
    return document
