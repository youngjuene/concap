from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from dpo.regen.assignment import PREPARED, REGENERATED, assign
from dpo.regen.config import Calibration, Configuration
from dpo.regen.document import (
    REGEN_SCHEMA,
    REGEN_SCHEMA_V4,
    RegenDocumentError,
    segment_order,
    validate_regen_document,
)
from dpo.regen.tests.test_qa_validation import _segment


def _document(names: list[str], *, schema: str = REGEN_SCHEMA_V4) -> dict[str, Any]:
    configuration = Configuration(
        study_id="flexible",
        corpus_id="documents",
        calibration=Calibration(cue_slots=4, minimum_points=1),
    )
    segments = {}
    for name in names:
        segment = _segment(name)
        segment["clip_id"] = f"clip_{name}"
        segments[name] = segment
    document: dict[str, Any] = {
        "schema": schema,
        "session_id": "flexible-documents",
        "config": configuration.artifact(),
        "segments": segments,
    }
    if schema == REGEN_SCHEMA_V4:
        document["clip_order"] = list(names)
    return document


def test_v4_accepts_one_clip_and_keeps_explicit_order() -> None:
    document = _document(["solo"])

    validate_regen_document(document)

    assert segment_order(document, sequence=0) == ("solo",)
    assert segment_order(document, sequence=7) == ("solo",)


def test_v4_accepts_four_clips_and_rotates_by_sequence() -> None:
    document = _document(["clip_1", "clip_2", "clip_3", "clip_4"])

    validate_regen_document(document)

    assert segment_order(document, sequence=0) == ("clip_1", "clip_2", "clip_3", "clip_4")
    assert segment_order(document, sequence=2) == ("clip_3", "clip_4", "clip_1", "clip_2")
    assert segment_order(document, sequence=5) == ("clip_2", "clip_3", "clip_4", "clip_1")


@pytest.mark.parametrize(
    "clip_order,error",
    [
        (["clip_1", "clip_2"], "missing"),
        (["clip_1", "clip_2", "clip_3", "ghost"], "unknown"),
        (["clip_1", "clip_2", "clip_2"], "declared twice"),
    ],
)
def test_v4_rejects_clip_order_that_is_not_exactly_the_segments(clip_order: list[str], error: str) -> None:
    document = _document(["clip_1", "clip_2", "clip_3"])
    document["clip_order"] = clip_order

    with pytest.raises(RegenDocumentError, match=error):
        validate_regen_document(document)


def test_v4_rejects_duplicate_clip_ids() -> None:
    document = _document(["clip_1", "clip_2", "clip_3"])
    document["segments"]["clip_3"]["clip_id"] = document["segments"]["clip_1"]["clip_id"]

    with pytest.raises(RegenDocumentError, match="same clip as segment clip_1"):
        validate_regen_document(document)


def test_v3_keeps_exact_legacy_ab_validation_and_parity_order() -> None:
    document = _document(["A", "B"], schema=REGEN_SCHEMA)

    validate_regen_document(document)

    assert segment_order(document, sequence=0) == ("A", "B")
    assert segment_order(document, sequence=1) == ("B", "A")

    invalid = deepcopy(document)
    invalid["segments"]["C"] = invalid["segments"].pop("B")
    invalid["segments"]["C"]["segment"] = "C"
    with pytest.raises(RegenDocumentError, match="must be exactly"):
        validate_regen_document(invalid)


def test_assignment_legacy_semantics_are_unchanged() -> None:
    even = assign(0)
    odd = assign(1)

    assert even.record() == {"sequence": 0, "prepared_segment": "A", "regenerated_segment": "B"}
    assert odd.record() == {"sequence": 1, "prepared_segment": "B", "regenerated_segment": "A"}
    assert even.condition_of("A") == PREPARED
    assert even.condition_of("B") == REGENERATED
