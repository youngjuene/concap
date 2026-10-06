"""Regressions for preflight, atomic page changes, enrollment and event scope."""

from __future__ import annotations

import json
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from dpo.regen.regeneration import RegenTemplateWriter
from dpo.regen.sheet_protocol import PROTOCOL, SheetQuestionnaire
from dpo.regen.study_store import Conflict
from tests.regen.test_sheet_protocol import study_config


def questionnaire(document: Any, media_dir: Path, tmp_path: Path) -> SheetQuestionnaire:
    Image.new("RGB", (64, 64), "#d2b48c").save(media_dir / "practice.png")
    return SheetQuestionnaire(
        document, media_dir, tmp_path / "out", RegenTemplateWriter(), None, study_config(document)
    )


def send(protocol: SheetQuestionnaire, token: str, action: str, data: Any = None) -> dict[str, Any]:
    state = protocol.store.state(token)
    return protocol.mutate(
        token,
        action,
        {
            "key": str(uuid.uuid4()),
            "revision": state["revision"],
            "data": {"page": state["sheet_page"], **(data or {})},
        },
    )


def opposite_tracks(document: Any) -> dict[str, Any]:
    track = document["segments"]["A"]["prepared_track"]
    return {
        f"{visual}|{sound}": deepcopy(track)
        for visual in ("building", "person", "unclassified")
        for sound in ("traffic", "bird")
    }


@pytest.mark.parametrize("missing", ["person|traffic", "unclassified|bird", "building|bird"])
def test_preflight_requires_all_opposite_response_combinations(
    document: Any, media_dir: Path, tmp_path: Path, missing: str
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    tracks = opposite_tracks(document)
    tracks.pop(missing)
    protocol.config["clips"]["A"].update(second_caption_strategy="opposite", second_tracks=tracks)
    token = protocol.create()
    before = protocol.checked(token)
    errors = protocol.setup_missing(before)
    assert errors and any(missing in error for error in errors)
    with pytest.raises(Conflict):
        send(protocol, token, "intro-complete", {"consented": True})
    assert protocol.store.state(token) == before


@pytest.mark.parametrize(
    "defect",
    [
        "A1s",
        "A2s",
        "R2",
        "practice-options",
        "practice-image",
        "first-language",
        "second-timing",
        "second-null",
    ],
)
def test_preflight_rejects_invalid_materials_before_consent(
    document: Any, media_dir: Path, tmp_path: Path, defect: str
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    if defect in {"A1s", "A2s", "R2"}:
        protocol.config["clips"]["A"]["options"][defect] = [{"value": "missing-label"}]
    elif defect == "practice-options":
        protocol.config["practice"]["sound_options"] = [{"value": "missing-label"}]
    elif defect == "practice-image":
        (media_dir / "practice.png").write_bytes(b"not an image")
    elif defect == "first-language":
        track = deepcopy(document["segments"]["A"]["prepared_track"])
        track[0]["text"] = {"ko": "English only"}
        protocol.config["clips"]["A"]["first_track"] = track
    else:
        tracks = opposite_tracks(document)
        if defect == "second-null":
            tracks["person|traffic"] = None
        else:
            tracks["person|traffic"][0]["end_ms"] -= 1
        protocol.config["clips"]["A"].update(second_caption_strategy="opposite", second_tracks=tracks)
    token = protocol.create()
    before = protocol.checked(token)
    assert protocol.setup_missing(before)
    with pytest.raises(Conflict):
        send(protocol, token, "intro-complete", {"consented": True})
    assert protocol.store.state(token) == before


def test_old_invalid_next_page_rolls_back_submission_instead_of_committing_failure(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    token = protocol.create()

    def old_session(state: dict[str, Any]) -> None:
        state.update(sheet_page="P10", sheet_consent={"accepted": True})
        state["sheet_config"]["clips"]["A"]["options"]["R2"] = [{"value": "missing-label"}]

    protocol.store.mutate(
        token, "old-invalid-fixture", protocol.checked(token)["revision"], "fixture", old_session
    )
    before = protocol.store.export(token)
    with pytest.raises(ValueError, match="label"):
        send(protocol, token, "submit", {"answers": {"S1": 0, "S2": 0, "S3": 0}})
    assert protocol.store.export(token) == before


def test_setup_can_return_to_an_earlier_config_without_replaying_old_receipt(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    token = protocol.create()
    revision = protocol.checked(token)["revision"]
    for version in ("A", "B", "A", "B"):
        protocol.config["consent_text"] = f"검토된 동의서 {version}"
        state = protocol.checked(token)
        assert state["sheet_config"]["consent_text"] == f"검토된 동의서 {version}"
        assert state["revision"] == revision + 1
        revision = state["revision"]
        assert protocol.checked(token)["revision"] == revision


def test_concurrent_registration_assigns_distinct_persisted_sequences(
    document: Any, media_dir: Path, tmp_path: Path, monkeypatch: Any
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    store = protocol.store
    original = store.create
    inserted = threading.Barrier(2)

    def concurrent_create(*args: Any, **kwargs: Any) -> str:
        token = original(*args, **kwargs)
        # Both CREATEs finish before the caller resumes: old split enrollment
        # then reads the same row count twice, atomic initialization does not.
        inserted.wait(timeout=10)
        return token

    monkeypatch.setattr(store, "create", concurrent_create)
    with ThreadPoolExecutor(max_workers=2) as workers:
        tokens = list(workers.map(lambda _: protocol.create(), range(2)))
    states = [store.state(token) for token in tokens]
    assert sorted(state["sheet_clip_order"] for state in states) == [["A", "B"], ["B", "A"]]
    assert sorted(state["registration_sequence"] for state in states) == [0, 1]
    for token in tokens:
        events = store.export(token)["events"]
        assert len(events) == 1
        assert json.loads(events[0]["body"])["action"] == "sheet-enrollment"


def test_event_scope_uses_server_page_before_clip_advances(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    token = protocol.create()
    state = protocol.checked(token)
    protocol.store.mutate(
        token,
        "last-page-fixture",
        state["revision"],
        "fixture",
        lambda state: state.update(sheet_page="P11", sheet_consent={"accepted": True}),
    )
    result = send(
        protocol,
        token,
        "submit",
        {
            "answers": {"R1": 0, "R2": ["memory_unknown"]},
            "phase": 99,
            "clip_index": 99,
            "clip_id": "forged",
            "view_id": "forged",
            "condition": "forged",
            "protocol": "forged",
        },
    )
    assert result["page"] == "P1" and result["clip_index"] == 1
    event = json.loads(protocol.store.export(token)["events"][-1]["body"])["data"]
    assert event["phase"] == 1 and event["page"] == event["stage"] == "P11"
    assert event["clip_index"] == 0 and event["clip_id"] == document["segments"]["A"]["clip_id"]
    assert event["view_id"] == f"{state['session_id']}-0-P6"
    assert event["condition"] == "second" and event["protocol"] == PROTOCOL
    assert event["stimulus_assignment"] == study_config(document)["clips"]["A"]["condition"]


def test_practice_events_are_distinct_from_formal_video_scope(
    document: Any, media_dir: Path, tmp_path: Path
) -> None:
    protocol = questionnaire(document, media_dir, tmp_path)
    token = protocol.create()
    send(protocol, token, "intro-complete", {"consented": True})
    send(
        protocol,
        token,
        "practice-complete",
        {
            "answers": {
                "practice_rating": 0,
                "practice_sound": "practice_a",
                "practice_visual": [{"frame_id": "0", "x": 0.5, "y": 0.5}],
            }
        },
    )
    data = json.loads(protocol.store.export(token)["events"][-1]["body"])["data"]
    assert data["page"] == "practice" and data["phase"] == 0 and data["analysis_excluded"] is True
    assert data["clip_index"] is data["clip_id"] is data["view_id"] is data["condition"] is None
