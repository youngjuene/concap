"""Rebuilding a stage's in-memory inputs from published artifacts.

The offline canary holds every derived row in memory and hands it straight to
the next stage. The CLI cannot: each command receives content-addressed
artifact ids and must reconstruct exactly the rows the shared stage functions
consume. These parsers are the inverse of the ``document()`` methods the view
and matrix-cell payloads are published from — strict about the field set, so a
schema drift fails loudly here instead of producing a subtly different view.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from dpo.annotation.raw_annotations import RawAnnotation, parse_annotation
from dpo.contracts.study_contract import TRACKS
from dpo.core.artifacts import ArtifactError, ArtifactManifest, ArtifactStore
from dpo.data.derive_pairs import MetadataPair, StrictPair, ViewError
from dpo.data.derive_sft import SftExample
from dpo.pipeline.run_matrix import CellResult

SFT_VIEW_TYPE = "dpo.sft-view/v1"
PAIR_STRICT_VIEW_TYPE = "dpo.pair-strict-view/v1"
PAIR_ALL_VIEW_TYPE = "dpo.pair-all-view/v1"
VALIDATION_PAIRS_TYPE = "dpo.validation-pairs/v1"
MATRIX_CELL_TYPE = "dpo.matrix-cell/v1"

# Published view type -> the key the stage functions index it by.
VIEW_KEYS: dict[str, str] = {
    SFT_VIEW_TYPE: "sft",
    PAIR_STRICT_VIEW_TYPE: "pair_strict",
    PAIR_ALL_VIEW_TYPE: "pair_all",
    VALIDATION_PAIRS_TYPE: "validation_pairs",
}

_STRICT_PAIR_FIELDS = {
    "pair_id",
    "clip_id",
    "track",
    "chosen_id",
    "rejected_id",
    "chosen_text",
    "rejected_text",
    "category",
    "agreement",
    "mean_strength",
    "difficulty",
    "weight",
}
_METADATA_PAIR_FIELDS = {
    "pair_id",
    "clip_id",
    "track",
    "candidate_a",
    "candidate_b",
    "text_a",
    "text_b",
    "category",
    "preference_probability",
    "a_probability",
    "b_probability",
    "tie_probability",
    "both_bad_probability",
    "mean_confidence",
    "agreement",
    "disagreement_entropy",
    "difficulty",
    "weight",
}
_SFT_FIELDS = {"example_id", "clip_id", "track", "candidate_id", "completion", "consensus_weight"}
_CELL_FIELDS = {
    "schema",
    "experiment_id",
    "variant_id",
    "track",
    "seed",
    "trained",
    "objective",
    "training_view",
    "hyperparameters",
    "steps",
    "final_loss",
    "checkpoint_signature",
    "reference_signature",
    "compute",
    "diagnostics_summary",
}


def _document(payload: bytes | str | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(payload, (bytes, str)):
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise ViewError("view payload is not valid JSON") from exc
    else:
        parsed = dict(payload)
    if not isinstance(parsed, dict):
        raise ViewError("view payload must be an object")
    return parsed


def _rows(payload: bytes | str | Mapping[str, Any], *, schema: str) -> list[Mapping[str, Any]]:
    document = _document(payload)
    if document.get("schema") != schema or set(document) != {"schema", "rows"}:
        raise ViewError(f"payload is not a {schema} document")
    rows = document["rows"]
    if not isinstance(rows, list) or any(not isinstance(row, Mapping) for row in rows):
        raise ViewError(f"{schema} rows must be an array of objects")
    return [row for row in rows if isinstance(row, Mapping)]


def _checked(row: Mapping[str, Any], fields: set[str], *, what: str) -> Mapping[str, Any]:
    if set(row) != fields:
        unexpected = sorted(set(row) ^ fields)
        raise ViewError(f"{what} row has an unexpected field set (first difference {unexpected[0]!r})")
    return row


def _strict_pair(row: Mapping[str, Any]) -> StrictPair:
    _checked(row, _STRICT_PAIR_FIELDS, what="pair")
    return StrictPair(
        pair_id=str(row["pair_id"]),
        clip_id=str(row["clip_id"]),
        track=str(row["track"]),
        chosen_id=str(row["chosen_id"]),
        rejected_id=str(row["rejected_id"]),
        chosen_text=str(row["chosen_text"]),
        rejected_text=str(row["rejected_text"]),
        category=str(row["category"]),
        agreement=float(row["agreement"]),
        mean_strength=float(row["mean_strength"]),
        difficulty=float(row["difficulty"]),
        weight=float(row["weight"]),
    )


def parse_strict_pairs(payload: bytes | str, *, schema: str) -> tuple[StrictPair, ...]:
    """Strict pair rows of a published pair-strict or validation-pairs view."""
    return tuple(_strict_pair(row) for row in _rows(payload, schema=schema))


def parse_metadata_pairs(payload: bytes | str) -> tuple[MetadataPair, ...]:
    rows = _rows(payload, schema=PAIR_ALL_VIEW_TYPE)
    return tuple(
        MetadataPair(
            pair_id=str(row["pair_id"]),
            clip_id=str(row["clip_id"]),
            track=str(row["track"]),
            candidate_a=str(row["candidate_a"]),
            candidate_b=str(row["candidate_b"]),
            text_a=str(row["text_a"]),
            text_b=str(row["text_b"]),
            category=str(row["category"]),
            preference_probability=float(row["preference_probability"]),
            a_probability=float(row["a_probability"]),
            b_probability=float(row["b_probability"]),
            tie_probability=float(row["tie_probability"]),
            both_bad_probability=float(row["both_bad_probability"]),
            mean_confidence=float(row["mean_confidence"]),
            agreement=float(row["agreement"]),
            disagreement_entropy=float(row["disagreement_entropy"]),
            difficulty=float(row["difficulty"]),
            weight=float(row["weight"]),
        )
        for row in (_checked(row, _METADATA_PAIR_FIELDS, what="metadata pair") for row in rows)
    )


def parse_sft_rows(payload: bytes | str) -> tuple[SftExample, ...]:
    rows = _rows(payload, schema=SFT_VIEW_TYPE)
    return tuple(
        SftExample(
            example_id=str(row["example_id"]),
            clip_id=str(row["clip_id"]),
            track=str(row["track"]),
            candidate_id=str(row["candidate_id"]),
            completion=str(row["completion"]),
            consensus_weight=float(row["consensus_weight"]),
        )
        for row in (_checked(row, _SFT_FIELDS, what="SFT") for row in rows)
    )


def parse_annotations(payload: bytes | str) -> tuple[RawAnnotation, ...]:
    """Raw annotation rows of a published raw-annotations payload."""
    document = _document(payload)
    if document.get("schema") != "dpo.raw-annotations/v1" or set(document) != {"schema", "rows"}:
        raise ViewError("payload is not a dpo.raw-annotations/v1 document")
    rows = document["rows"]
    if not isinstance(rows, list):
        raise ViewError("raw-annotations rows must be an array")
    return tuple(parse_annotation(row) for row in rows if isinstance(row, Mapping))


def parse_cell_result(document: Mapping[str, Any]) -> CellResult:
    """Rebuild one matrix cell; ``document()`` of the result must round-trip."""
    if set(document) != _CELL_FIELDS or document.get("schema") != MATRIX_CELL_TYPE:
        raise ArtifactError("payload is not a dpo.matrix-cell/v1 document")
    compute = document["compute"]
    diagnostics = document["diagnostics_summary"]
    hyperparameters = document["hyperparameters"]
    if not isinstance(compute, Mapping) or not isinstance(diagnostics, Mapping):
        raise ArtifactError("matrix cell compute/diagnostics must be objects")
    if not isinstance(hyperparameters, Mapping):
        raise ArtifactError("matrix cell hyperparameters must be an object")
    final_loss = document["final_loss"]
    objective = document["objective"]
    training_view = document["training_view"]
    reference = document["reference_signature"]
    return CellResult(
        experiment_id=str(document["experiment_id"]),
        variant_id=str(document["variant_id"]),
        track=str(document["track"]),
        seed=int(document["seed"]),
        trained=bool(document["trained"]),
        objective=None if objective is None else str(objective),
        training_view=None if training_view is None else str(training_view),
        hyperparameters=dict(hyperparameters),
        steps=int(document["steps"]),
        final_loss=None if final_loss is None else float(final_loss),
        checkpoint_signature=str(document["checkpoint_signature"]),
        reference_signature=None if reference is None else str(reference),
        compute={str(key): int(value) for key, value in compute.items()},
        diagnostics_summary=dict(diagnostics),
    )


# ---------------------------------------------------------------------------
# Artifact resolution.
# ---------------------------------------------------------------------------


def request_parameters(manifest: ArtifactManifest) -> Mapping[str, Any]:
    request = manifest.semantic["request"]
    parameters = request["parameters"] if isinstance(request, Mapping) else None
    if not isinstance(parameters, Mapping):
        raise ArtifactError(f"artifact {manifest.artifact_id} has no request parameters")
    return parameters


def artifact_track(manifest: ArtifactManifest) -> str:
    track = request_parameters(manifest).get("track")
    if not isinstance(track, str) or track not in TRACKS:
        raise ArtifactError(f"artifact {manifest.artifact_id} does not declare a caption track")
    return track


@dataclass
class TrackViewInputs:
    """Every published view of every track, parsed back into rows."""

    artifact_ids: dict[str, dict[str, str]] = field(default_factory=dict)
    sft_rows: dict[str, tuple[SftExample, ...]] = field(default_factory=dict)
    strict_pairs: dict[str, tuple[StrictPair, ...]] = field(default_factory=dict)
    metadata_pairs: dict[str, tuple[MetadataPair, ...]] = field(default_factory=dict)
    validation_pairs: dict[str, tuple[StrictPair, ...]] = field(default_factory=dict)


def collect_view_inputs(
    store: ArtifactStore,
    manifests: Sequence[ArtifactManifest],
    *,
    required: Sequence[str],
    tracks: Sequence[str],
) -> TrackViewInputs:
    """Parse the given view artifacts, one set per track, requiring ``required``.

    ``tracks`` is what the contract declares, not every track that exists: a
    single-track study publishes views for one track and must not be told the
    other is missing.
    """
    inputs = TrackViewInputs()
    for manifest in manifests:
        key = VIEW_KEYS.get(manifest.artifact_type)
        if key is None:
            continue
        track = artifact_track(manifest)
        seen = inputs.artifact_ids.setdefault(track, {})
        if key in seen and seen[key] != manifest.artifact_id:
            raise ArtifactError(f"two different {manifest.artifact_type} artifacts were given for {track}")
        seen[key] = manifest.artifact_id
        payload = store.read_payload(manifest.artifact_id)
        if key == "sft":
            inputs.sft_rows[track] = parse_sft_rows(payload)
        elif key == "pair_strict":
            inputs.strict_pairs[track] = parse_strict_pairs(payload, schema=PAIR_STRICT_VIEW_TYPE)
        elif key == "pair_all":
            inputs.metadata_pairs[track] = parse_metadata_pairs(payload)
        else:
            inputs.validation_pairs[track] = parse_strict_pairs(payload, schema=VALIDATION_PAIRS_TYPE)
    for track in tracks:
        present = inputs.artifact_ids.get(track, {})
        missing = [key for key in required if key not in present]
        if missing:
            raise ArtifactError(
                f"track {track!r} is missing its {missing[0]!r} view artifact;"
                " every track this contract declares must be published before this stage"
            )
    undeclared = sorted(set(inputs.artifact_ids) - set(tracks))
    if undeclared:
        raise ArtifactError(
            f"a view artifact was given for track {undeclared[0]!r}, which this contract does not declare"
        )
    return inputs


def collect_matrix_cells(
    store: ArtifactStore, manifests: Sequence[ArtifactManifest]
) -> tuple[dict[tuple[str, str, str], CellResult], dict[tuple[str, str, str], str]]:
    """Parsed matrix cells and their artifact ids, keyed (experiment, variant, track)."""
    cells: dict[tuple[str, str, str], CellResult] = {}
    artifacts: dict[tuple[str, str, str], str] = {}
    for manifest in manifests:
        if manifest.artifact_type != MATRIX_CELL_TYPE:
            continue
        cell = parse_cell_result(json.loads(store.read_payload(manifest.artifact_id)))
        key = (cell.experiment_id, cell.variant_id, cell.track)
        if key in artifacts and artifacts[key] != manifest.artifact_id:
            raise ArtifactError(f"two different matrix-cell artifacts were given for {key}")
        cells[key] = cell
        artifacts[key] = manifest.artifact_id
    if not cells:
        raise ArtifactError("this command requires the published matrix-cell artifacts")
    return cells, artifacts
