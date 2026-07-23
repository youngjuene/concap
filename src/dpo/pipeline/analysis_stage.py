"""Shared analysis-stage implementation: Bradley-Terry and the clustered CI.

Pairwise study outcomes become one analysis report: the Bradley-Terry fit,
the top model's win rate, and its clip-clustered bootstrap interval under the
contract's statistical plan. Where the outcomes come from is the caller's
concern — the offline canary synthesizes them, a live runner reads the real
study results.
"""

from __future__ import annotations

from collections.abc import Sequence

from dpo.analysis.bootstrap import clip_cluster_bootstrap
from dpo.analysis.bradley_terry import PairwiseOutcome, fit_bradley_terry
from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.pipeline.publishing import ArtifactPublisher


def publish_analysis_report(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    outcomes: Sequence[PairwiseOutcome],
    study_export_artifact_id: str,
    study_clips: Sequence[str],
) -> str:
    """Fit, bootstrap, and publish the analysis report over study outcomes."""
    fit = fit_bradley_terry(outcomes)
    top_model = fit.ranking()[0]
    point, interval = clip_cluster_bootstrap(
        outcomes,
        lambda outcome: outcome.clip_id,
        lambda rows: sum(1.0 for row in rows if row.winner == top_model) / len(rows),
        samples=int(str(contract.analysis["bootstrap_samples"])),
        seed=int(str(contract.analysis["seed"])),
    )
    return publisher.publish(
        "dpo.analysis-report/v1",
        {
            "schema": "dpo.analysis-report/v1",
            "bradley_terry": fit.document(),
            "top_model_win_rate": {"point": point, "interval_95": list(interval)},
        },
        parents=(ParentEdge(study_export_artifact_id, "study-export"),),
        stage="analysis",
        parameters={"operation": "analysis"},
        clips=set(study_clips),
        role_exposure={"study"},
    )
