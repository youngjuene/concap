"""Shared validate/lock-stage implementation: score, select, and freeze.

Every experiment variant is scored identically on the validation pairs
through the one shared ``completion_logps`` interface; the validation report,
the per-experiment winner selection (max validation accuracy, lexical
variant-id tie-break), the ranking, the selection report, and the lock
manifest all have one home. Only selected variants reach the lock.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch

from dpo.contracts.study_contract import EXPERIMENT_IDS, StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.data.derive_pairs import StrictPair
from dpo.evaluation.preference_accuracy import ScoredPair, evaluate_preferences
from dpo.models.base import CompletionBatch, MediaBatch, ModelAdapter
from dpo.pipeline.experiments import ExperimentVariant
from dpo.pipeline.lock import LockManifest, create_lock_manifest
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.run_matrix import CellResult


@dataclass(frozen=True)
class SelectionOutcome:
    """What downstream stages need from validation, selection, and the lock."""

    validation_reports: dict[str, dict[str, dict[str, float]]]
    selected_variants: dict[str, dict[str, str]]
    ranking: dict[str, list[str]]
    lock_manifest: LockManifest
    validation_artifact_id: str
    selection_artifact_id: str
    lock_artifact_id: str


def publish_selection(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    variants_by_experiment: Mapping[str, tuple[ExperimentVariant, ...]],
    canonical_seed: int,
    validation_pairs: Mapping[str, tuple[StrictPair, ...]],
    strict_pairs: Mapping[str, tuple[StrictPair, ...]],
    policies: Mapping[tuple[str, str, str, int], ModelAdapter],
    seed_adapters: Mapping[str, ModelAdapter],
    media_provider: Callable[[str, Sequence[str]], MediaBatch],
    view_artifacts: Mapping[str, Mapping[str, str]],
    cells: Mapping[tuple[str, str, str], CellResult],
    cell_artifacts: Mapping[tuple[str, str, str], str],
    processor_hash: str,
    preprocessing_hash: str,
    evaluation_version: str,
    metric_versions: Mapping[str, str],
    selection_note: str,
) -> SelectionOutcome:
    """Score every variant, publish validation/selection reports, and lock."""
    # Common validation scoring: every experiment variant is scored identically.
    validation_reports: dict[str, dict[str, dict[str, float]]] = {}
    validation_scores: dict[str, dict[str, dict[str, list[dict[str, object]]]]] = {}
    for track in contract.tracks:
        seed_adapter = seed_adapters[track]
        sft_adapter = policies[("SFT", "base", track, canonical_seed)]
        report_by_experiment: dict[str, dict[str, float]] = {}
        scores_by_experiment: dict[str, dict[str, list[dict[str, object]]]] = {}
        for experiment_id in EXPERIMENT_IDS:
            report_by_variant: dict[str, float] = {}
            scores_by_variant: dict[str, list[dict[str, object]]] = {}
            for variant in variants_by_experiment[experiment_id]:
                policy_adapter = policies[(experiment_id, variant.variant_id, track, canonical_seed)]
                reference_adapter = sft_adapter if experiment_id == "SFT_DPO" else seed_adapter
                scored = []
                for validation_pair in validation_pairs[track]:
                    media = media_provider(track, [validation_pair.clip_id])
                    chosen_batch = CompletionBatch(texts=(validation_pair.chosen_text,))
                    rejected_batch = CompletionBatch(texts=(validation_pair.rejected_text,))
                    with torch.no_grad():
                        policy_chosen = float(policy_adapter.completion_logps(media, chosen_batch)[0])
                        policy_rejected = float(policy_adapter.completion_logps(media, rejected_batch)[0])
                        ref_chosen = float(reference_adapter.completion_logps(media, chosen_batch)[0])
                        ref_rejected = float(reference_adapter.completion_logps(media, rejected_batch)[0])
                    scored.append(
                        ScoredPair(
                            pair_id=validation_pair.pair_id,
                            clip_id=validation_pair.clip_id,
                            track=track,
                            policy_chosen_logp=policy_chosen,
                            policy_rejected_logp=policy_rejected,
                            ref_chosen_logp=ref_chosen,
                            ref_rejected_logp=ref_rejected,
                            difficulty=validation_pair.difficulty,
                            agreement=validation_pair.agreement,
                        )
                    )
                # SEED and SFT carry no trained beta; score them at beta=1.0
                # (accuracy is beta-invariant; beta only scales the probability fields).
                hyper = variant.hyperparameters
                beta = float(str(hyper["beta"])) if "beta" in hyper else 1.0
                preference_report = evaluate_preferences(scored, beta=beta)
                report_by_variant[variant.variant_id] = preference_report.accuracy
                # The per-pair scores are what any inferential comparison needs
                # (clip-clustered CIs, paired tests, Bradley-Terry); dropping
                # them here would force the expensive scoring pass to be redone.
                scores_by_variant[variant.variant_id] = [
                    {
                        "pair_id": pair.pair_id,
                        "clip_id": pair.clip_id,
                        "policy_chosen_logp": pair.policy_chosen_logp,
                        "policy_rejected_logp": pair.policy_rejected_logp,
                        "ref_chosen_logp": pair.ref_chosen_logp,
                        "ref_rejected_logp": pair.ref_rejected_logp,
                        "difficulty": pair.difficulty,
                        "agreement": pair.agreement,
                    }
                    for pair in scored
                ]
            report_by_experiment[experiment_id] = report_by_variant
            scores_by_experiment[experiment_id] = scores_by_variant
        validation_reports[track] = report_by_experiment
        validation_scores[track] = scores_by_experiment
    validation_artifact = publisher.publish(
        "dpo.validation-report/v1",
        {"schema": "dpo.validation-report/v1", "accuracy": validation_reports, "scores": validation_scores},
        parents=tuple(
            ParentEdge(view_artifacts[track]["validation_pairs"], "validation-pairs")
            for track in contract.tracks
        )
        + tuple(ParentEdge(artifact_id, "matrix-cell") for artifact_id in cell_artifacts.values()),
        stage="validate",
        parameters={"operation": "validate"},
        clips={row.clip_id for track in contract.tracks for row in validation_pairs[track]}
        | {row.clip_id for track in contract.tracks for row in strict_pairs[track]},
        role_exposure={"train", "validation"},
        selection_exposure={row.clip_id for track in contract.tracks for row in validation_pairs[track]},
    )
    # Selection: one winning variant per experiment and track (max validation
    # accuracy, lexical variant-id tie-break), then experiments ranked by their
    # winner. Only selected variants reach the lock.
    selected_variants: dict[str, dict[str, str]] = {}
    for track in contract.tracks:
        selected_variants[track] = {}
        for experiment_id in EXPERIMENT_IDS:
            per_variant = validation_reports[track][experiment_id]
            selected_variants[track][experiment_id] = sorted(
                per_variant, key=lambda variant_id: (-per_variant[variant_id], variant_id)
            )[0]
    ranking = {
        track: sorted(
            EXPERIMENT_IDS,
            key=lambda experiment_id: (
                -validation_reports[track][experiment_id][selected_variants[track][experiment_id]],
                experiment_id,
            ),
        )
        for track in contract.tracks
    }
    selection_artifact = publisher.publish(
        "dpo.selection-report/v1",
        {
            "schema": "dpo.selection-report/v1",
            "ranking": ranking,
            "selected_variants": selected_variants,
            "selected_hyperparameters": {
                track: {
                    experiment_id: dict(
                        cells[(experiment_id, selected_variants[track][experiment_id], track)].hyperparameters
                    )
                    for experiment_id in EXPERIMENT_IDS
                }
                for track in contract.tracks
            },
            "canonical_seed": canonical_seed,
            "note": selection_note,
        },
        parents=(ParentEdge(validation_artifact, "validation-report"),),
        stage="validate",
        parameters={"operation": "selection"},
    )

    lock_manifest = create_lock_manifest(
        contract,
        checkpoint_hashes={
            experiment_id: {
                track: cells[
                    (experiment_id, selected_variants[track][experiment_id], track)
                ].checkpoint_signature
                for track in contract.tracks
            }
            for experiment_id in EXPERIMENT_IDS
        },
        processor_hash=processor_hash,
        preprocessing_hash=preprocessing_hash,
        evaluation_version=evaluation_version,
        metric_versions=metric_versions,
        selection_report_hash=selection_artifact,
    )
    lock_artifact = publisher.publish(
        "dpo.lock-manifest/v1",
        lock_manifest.document,
        parents=(ParentEdge(selection_artifact, "selection-report"),),
        stage="lock",
        parameters={"operation": "lock"},
        attributes={"lock_id": lock_manifest.lock_id},
    )
    return SelectionOutcome(
        validation_reports=validation_reports,
        selected_variants=selected_variants,
        ranking=ranking,
        lock_manifest=lock_manifest,
        validation_artifact_id=validation_artifact,
        selection_artifact_id=selection_artifact,
        lock_artifact_id=lock_artifact,
    )
