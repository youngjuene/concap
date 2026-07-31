"""The stage registry: every pipeline stage's artifact-type and contract interface.

Stages declare the artifact types they consume and produce and the contract
sections they read. The registry is load-bearing: the CLI validates command
inputs against the declared artifact types, and every published artifact's
``contract_id`` is the semantic hash of exactly the declared contract slice
(``contract_slice_hash``), so an artifact's cache identity changes only when
a section its stage actually reads changes. A stage that needed an
undeclared section would produce wrong reuse — declare a superset, never a
subset.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dpo.core.identity import semantic_hash

FULL_CONTRACT = ("*",)


class StageError(ValueError):
    """Raised when a stage's declared interface is violated."""


@dataclass(frozen=True)
class PipelineStage:
    name: str
    input_artifact_types: tuple[str, ...]
    output_artifact_types: tuple[str, ...]
    contract_sections: tuple[str, ...]
    description: str


STAGES: dict[str, PipelineStage] = {
    stage.name: stage
    for stage in (
        PipelineStage(
            "contract",
            (),
            ("dpo.contract-lock/v1",),
            FULL_CONTRACT,
            "Validate and lock the study contract",
        ),
        PipelineStage(
            "corpus-ingest",
            (),
            ("dpo.corpus-ingest/v1",),
            ("corpus",),
            "Register clip rows with media hashes",
        ),
        PipelineStage(
            "lock-splits",
            ("dpo.corpus-ingest/v1",),
            ("dpo.clip-registry/v1", "dpo.clip-registry-shard/v1"),
            ("corpus",),
            "Compute and freeze the immutable group-level splits",
        ),
        PipelineStage(
            "candidates",
            ("dpo.clip-registry-shard/v1",),
            ("dpo.frozen-candidate-pool/v1",),
            ("candidates", "pairs", "tracks", "models"),
            "Generate, audit, pair, and freeze the candidate pool",
        ),
        PipelineStage(
            "annotation",
            ("dpo.frozen-candidate-pool/v1",),
            ("dpo.raw-annotations/v1", "dpo.reliability-report/v1"),
            ("annotation",),
            "Collect human preferences against the frozen pool",
        ),
        PipelineStage(
            "views",
            ("dpo.raw-annotations/v1", "dpo.frozen-candidate-pool/v1"),
            (
                "dpo.sft-view/v1",
                "dpo.pair-strict-view/v1",
                "dpo.pair-all-view/v1",
                "dpo.noise-calibration/v1",
                "dpo.flip-manifest/v1",
            ),
            ("views", "robustness"),
            "Derive every training view from the one frozen preference dataset",
        ),
        PipelineStage(
            "train",
            ("dpo.sft-view/v1", "dpo.pair-strict-view/v1", "dpo.pair-all-view/v1"),
            ("dpo.matrix-cell/v1",),
            ("training", "experiments", "models", "tracks"),
            "Run one experiment-matrix cell",
        ),
        PipelineStage(
            "validate",
            ("dpo.matrix-cell/v1",),
            ("dpo.validation-report/v1", "dpo.selection-report/v1"),
            ("validation", "training", "experiments", "models", "tracks"),
            "Common validation and model selection over all cells",
        ),
        PipelineStage(
            "lock",
            ("dpo.selection-report/v1",),
            ("dpo.lock-manifest/v1",),
            FULL_CONTRACT,
            "Freeze configuration before any test access",
        ),
        PipelineStage(
            "study-export",
            ("dpo.lock-manifest/v1",),
            ("dpo.study-export/v1",),
            ("validation", "tracks", "models"),
            "Caption the held-out study split with the locked winner, for the human study",
        ),
    )
}


def stage(name: str) -> PipelineStage:
    found = STAGES.get(name)
    if found is None:
        raise StageError(f"unknown pipeline stage {name!r}")
    return found


def contract_slice_hash(raw: Mapping[str, Any], contract_hash: str, sections: tuple[str, ...]) -> str:
    """Cache identity of the contract slice a stage reads.

    ``FULL_CONTRACT`` stages key on the whole-contract hash; every other stage
    keys on exactly its declared sections, so tweaking an unrelated section
    (e.g. an experiment's beta) leaves the stage's artifacts cache-valid.
    """
    if sections == FULL_CONTRACT:
        return contract_hash
    missing = sorted(set(sections) - set(raw))
    if missing:
        raise StageError(f"declared contract section {missing[0]!r} is absent from the contract")
    return semantic_hash({name: raw[name] for name in sorted(sections)})


def allowed_contract_ids(raw: Mapping[str, Any], contract_hash: str) -> frozenset[str]:
    """Every contract_id an artifact of this contract may legitimately carry."""
    ids = {contract_hash}
    for entry in STAGES.values():
        ids.add(contract_slice_hash(raw, contract_hash, entry.contract_sections))
    return frozenset(ids)


def lineage_edges() -> list[tuple[str, str]]:
    """(producer stage, consumer stage) pairs implied by artifact types."""
    producers: dict[str, str] = {}
    for name, entry in STAGES.items():
        for artifact_type in entry.output_artifact_types:
            producers[artifact_type] = name
    edges = []
    for name, entry in STAGES.items():
        for artifact_type in entry.input_artifact_types:
            producer = producers.get(artifact_type)
            if producer is not None and producer != name:
                edges.append((producer, name))
    return sorted(set(edges))
