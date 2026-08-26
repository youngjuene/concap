"""Study-ingest stage: the human study's responses and their analysis, published.

Two artifacts, for the reason annotations and their aggregates are two:
``dpo.study-responses/v1`` is the validated record — every row exactly as the
instrument could have produced it, participants hashed — and
``dpo.study-results/v1`` is a pure function of it. A different analysis
republishes the results with the same responses as parent; the record is
never rewritten.

Both descend from the study export and inherit its study-role exposure; both
are public-derived types, so a reader needs no capability to open them — the
rows are what participants typed and chose, not the sealed clip registry.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dpo.analysis.study import summarize_study
from dpo.contracts.study_contract import StudyContract
from dpo.core.artifacts import ParentEdge
from dpo.pipeline.publishing import ArtifactPublisher
from dpo.pipeline.study_stage import StudyError
from dpo.userstudy.responses import StudyResponse, export_ladders

STUDY_RESPONSES_TYPE = "dpo.study-responses/v1"
STUDY_RESULTS_TYPE = "dpo.study-results/v1"


def publish_study_results(
    publisher: ArtifactPublisher,
    contract: StudyContract,
    *,
    export: Mapping[str, Any],
    export_artifact_id: str,
    responses: Sequence[StudyResponse],
) -> tuple[dict[str, object], dict[str, str]]:
    """Publish the validated responses, then the analysis over them."""
    ladders = export_ladders(export)
    rungs = len(next(iter(ladders.values())))
    declared = int(str(contract.study["rungs"]))
    if rungs != declared:
        raise StudyError(
            f"the study export carries {rungs}-rung ladders but this contract declares"
            f" study.rungs = {declared}; the export was made under a different [study] section"
        )
    if not responses:
        raise StudyError("study ingest needs at least one response")
    track = str(export["track"])
    clip_ids = set(ladders)
    responses_artifact = publisher.publish(
        STUDY_RESPONSES_TYPE,
        {
            "schema": STUDY_RESPONSES_TYPE,
            "track": track,
            "rows": [response.document() for response in responses],
        },
        parents=(ParentEdge(export_artifact_id, "study-export"),),
        stage="study-ingest",
        parameters={"operation": "study-ingest", "track": track},
        row_count=len(responses),
        clips=clip_ids,
        role_exposure={"study"},
        purpose="human-study",
    )
    summary = summarize_study(
        responses,
        rungs=rungs,
        bootstrap_samples=int(str(contract.validation["bootstrap_samples"])),
    )
    document: dict[str, object] = {
        "schema": STUDY_RESULTS_TYPE,
        "track": track,
        "experiment_id": str(export["experiment_id"]),
        "variant_id": str(export["variant_id"]),
        **summary,
    }
    results_artifact = publisher.publish(
        STUDY_RESULTS_TYPE,
        document,
        parents=(ParentEdge(responses_artifact, "study-responses"),),
        stage="study-ingest",
        parameters={"operation": "study-results", "track": track},
        clips=clip_ids,
        role_exposure={"study"},
        purpose="human-study",
    )
    return document, {"study_responses": responses_artifact, "study_results": results_artifact}
