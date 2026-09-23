"""Response wording is frozen, reproducible, and honest about old sessions."""

from __future__ import annotations

from typing import Any

import pytest

from dpo.regen.study_instrument import INSTRUMENT_PROVENANCE, instrument_snapshot
from dpo.regen.study_schema import FINAL_ITEMS, digest
from dpo.regen.tests import test_study

assets = test_study.assets
session = test_study.session


@pytest.mark.parametrize("language", ["en", "ko"])
def test_final_survey_records_frozen_presented_wording(session: Any, monkeypatch: Any, language: str) -> None:
    participant, store, _ = session
    frozen = instrument_snapshot(language)

    def ready(state: dict[str, Any]) -> None:
        # Isolate survey persistence; full playback eligibility has its own integration test.
        state.update(
            stage="final",
            language=language,
            instrument=frozen,
            profile={"hash": "test-profile", "defaults": {"texture": 0.5, "context": 0.5}},
            viewing_hash="test-media",
        )

    store.mutate(participant.token, "fixture-final", 0, "fixture", ready)
    # A later code/translation change must not rewrite a questionnaire already assigned.
    monkeypatch.setattr("dpo.regen.study_api.FINAL_ITEMS", [])
    state = participant.client.get("/api/study/state").json()
    assert state["items"] == frozen["items"]["final"]
    assert state["instrument_hash"] == frozen["hash"]
    assert state["items_provenance"] == INSTRUMENT_PROVENANCE
    values: dict[str, Any] = {}
    for item in state["items"]:
        if item["type"] == "rating":
            values[item["id"]] = "na" if item.get("na") else min(4, int(item.get("points", 5)))
        elif item["type"] == "choice":
            values[item["id"]] = item["options"][0]
        else:
            values[item["id"]] = ""
    payload = {"key": "response", "revision": state["revision"], "data": values}
    response = participant.client.post("/api/study/final-survey", json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["stage"] == "done"
    assert participant.client.post("/api/study/final-survey", json=payload).status_code == 200
    record = store.export(participant.token)["state"]["final_survey"]
    assert record["instrument"] == frozen
    assert record["items_hash"] == digest(frozen["items"]["final"])
    assert record["answers"] == values


def test_existing_unversioned_session_is_not_relabelled_as_frozen(session: Any) -> None:
    participant, store, _ = session

    def legacy(state: dict[str, Any]) -> None:
        state.pop("instrument")
        state.update(stage="final", profile={"hash": "old-profile", "defaults": {}}, viewing_hash="old-media")

    store.mutate(participant.token, "fixture-old", 0, "fixture", legacy)
    state = participant.client.get("/api/study/state").json()
    assert "instrument_hash" not in state
    values: dict[str, Any] = {item["id"]: 4 for item in FINAL_ITEMS if item["type"] == "rating"}
    values["timing"] = "Fast enough"
    response = participant.client.post(
        "/api/study/final-survey", json={"key": "response", "revision": state["revision"], "data": values}
    )
    assert response.status_code == 200, response.text
    record = store.state(participant.token)["final_survey"]
    assert record["instrument"] == {"provenance": "unversioned-legacy", "language": "en"}
    assert record["items_hash"] == digest(FINAL_ITEMS)
