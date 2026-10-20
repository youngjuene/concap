"""Sources validate strictly, carry their material and provenance, and import drafts read-only."""

from __future__ import annotations

import copy
from dataclasses import replace

import pytest
from diff_fixtures import relabelled, smoke_document

from diff.caption_sources import (
    QUALITIES,
    QUALITIES_SHA256,
    SourceError,
    SourceSet,
    bundle_problems,
    import_regen_document,
    import_viewing_manifest,
    parse_sources,
    provenance_eligible,
    starting_caption_review_problems,
)
from diff.study_protocol import sha256_of


def test_the_fixture_parses(smoke_sources: SourceSet) -> None:
    assert len(smoke_sources.phase(1)) == 8 and len(smoke_sources.phase(2)) == 3
    assert {source.material for source in smoke_sources.sources} == {"synthetic"}


def test_identity_ignores_material_provenance_and_review(smoke_sources: SourceSet) -> None:
    relabelled_sources = parse_sources(relabelled("development"))
    assert [s.source_id for s in smoke_sources.sources] == [s.source_id for s in relabelled_sources.sources]


def test_the_closed_vocabulary_is_regen_verbatim() -> None:
    from dpo.regen.caption_controls import QUALITIES as REGEN_QUALITIES

    assert sha256_of(QUALITIES) == QUALITIES_SHA256
    assert QUALITIES == REGEN_QUALITIES


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("material",), "unknown", "material"),
        (("status",), "approved", "status"),
        (("evidence", "events", 0, "qualities"), ["sparkly"], "closed vocabulary"),
        (("evidence", "events", 0, "category"), "vehicles", "category"),
        (("text",), "가" * 97, "exceeds 96"),
        (("source_id",), "src-0000000000000000", "does not match"),
    ],
)
def test_invalid_fields_are_refused(path: tuple[object, ...], value: object, message: str) -> None:
    document = copy.deepcopy(smoke_document())
    target: object = document["sources"][0]
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    with pytest.raises(SourceError, match=message):
        parse_sources(document)


def test_overlapping_cues_are_refused() -> None:
    document = copy.deepcopy(smoke_document())
    document["sources"][1]["start_ms"] = 2000
    with pytest.raises(SourceError, match="overlaps"):
        parse_sources(document)


def test_only_human_or_reviewed_provenance_enters_a_bundle(smoke_sources: SourceSet) -> None:
    assert provenance_eligible("human") and provenance_eligible("gemma-draft-reviewed")
    assert not provenance_eligible("gemma-draft-unreviewed") and not provenance_eligible("none")
    assert bundle_problems(smoke_sources.sources[0]) == []


def test_a_starting_caption_serves_participants_only_once_reviewed(smoke_sources: SourceSet) -> None:
    # Researcher, 2026-10-20: live cues need review but, unlike a bundle, no evidence events.
    reviewed = replace(smoke_sources.sources[0], events=(), status="reviewed", provenance_text="human")
    assert starting_caption_review_problems(reviewed) == []
    draft = replace(reviewed, status="draft")
    assert [p.split(": ", 1)[1] for p in starting_caption_review_problems(draft)] == [
        "status is draft, not reviewed"
    ]
    unreviewed = replace(reviewed, provenance_text="model-draft-unreviewed")
    assert "is not human or reviewed" in starting_caption_review_problems(unreviewed)[0]


def test_viewing_manifest_imports_as_unreviewed_drafts_without_events() -> None:
    manifest = {
        "viewing_videos": [
            {
                "id": "viewing-x",
                "status": "ready",
                "cues": [
                    {
                        "start_ms": 0,
                        "end_ms": 5000,
                        "evidence": "Sampled-frame visual context only.",
                        "fallback": {"ko": "소리"},
                    }
                ],
            }
        ]
    }
    imported = import_viewing_manifest(manifest)
    source = imported.sources[0]
    assert source.provenance_text == "gemma-draft-unreviewed" and source.events == ()
    assert source.material == "participant" and "visual context" in (source.note or "")
    assert bundle_problems(source)


def test_regen_document_imports_as_drafts() -> None:
    document = {
        "segments": {
            "A": {
                "segment": "A",
                "clip_id": "clip_x",
                "prepared_track": [
                    {"index": 0, "start_ms": 0, "end_ms": 1000, "text": {"en": "a car", "ko": "차 소리"}}
                ],
            }
        }
    }
    imported = import_regen_document(document, "development")
    source = imported.sources[0]
    assert source.text == "차 소리" and source.material == "development" and source.status == "draft"
