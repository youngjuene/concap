"""The sheet protocol keeps viewing telemetry without adding stage-two surveys."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.regen.study_api import build_study_app
from dpo.regen.tests.test_study import Session, form_values
from dpo.regen.tests.test_study import assets as _assets

assets = _assets

SHEET_POLICY = "interaction-only-sheet-v1"
POSTSTUDY = {
    "interview_prompts": [
        {"id": "D1", "text": "같은 영상을 두 번 보셨는데, 두 번째에 달라진 점이 있었나요?"},
        {
            "id": "D2",
            "text": "(달라졌다고 답한 경우) 어떤 점이 달라졌다고 느끼셨나요?",
            "conditional": True,
        },
    ],
    "debrief": {"text": "연구자가 제공한 테스트용 디브리핑 안내문입니다."},
}


def make_session(assets: Any, tmp_path: Path, *, sheet: bool = True) -> tuple[Any, Session]:
    root, manifest = assets
    path = tmp_path / "study.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    app = build_study_app(path, root, tmp_path / "out")
    session = Session(TestClient(app))
    session.calibrate()
    if sheet:

        def link(state: dict[str, Any]) -> None:
            state["survey_policy"] = SHEET_POLICY
            state["sheet_instrument"] = {"hash": "sheet-fixture-hash"}
            state["poststudy"] = POSTSTUDY

        app.state.store.mutate(session.token, "sheet-fixture", session.state["revision"], "fixture", link)
        session.state = session.client.get("/api/study/state").json()
    session.submit("start-viewing")
    return app, session


def finish_video(app: Any, session: Session) -> None:
    """Credit fixture playback; route tests assert the existing end-of-playback gate."""

    def viewed(state: dict[str, Any]) -> None:
        duration = state["viewing"][state["video_index"]]["duration_ms"]
        state["coverage"] = [[0, duration]]
        state["position_ms"] = duration

    app.state.store.mutate(
        session.token,
        f"viewed-{session.state['video_index']}",
        session.state["revision"],
        "fixture",
        viewed,
    )
    session.state = session.client.get("/api/study/state").json()
    session.submit("video-ended", {"video_id": session.state["video"]["id"]})


def finish_viewing(app: Any, session: Session) -> None:
    for index in range(3):
        finish_video(app, session)
        if index < 2:
            assert session.state["stage"] == "break"
            session.submit("continue")
    assert session.state["stage"] == "poststudy"


def test_legacy_viewing_still_collects_video_and_final_surveys(assets: Any, tmp_path: Path) -> None:
    app, session = make_session(assets, tmp_path, sheet=False)
    for index in range(3):
        finish_video(app, session)
        assert session.state["stage"] == "video-survey"
        session.submit(
            "video-survey",
            {"video_id": f"long-{index}", "answers": form_values(session.state["items"])},
        )
        if index < 2:
            session.submit("continue")
    assert session.state["stage"] == "final"
    session.submit("final-survey", form_values(session.state["items"]))
    assert session.state["stage"] == "done"


def test_sheet_viewing_preserves_logs_and_skips_all_stage_two_surveys(assets: Any, tmp_path: Path) -> None:
    app, session = make_session(assets, tmp_path)
    assert session.state["survey_policy"] == SHEET_POLICY
    assert session.state["items"] == []
    assert "poststudy" not in session.state  # Do not reveal the interview during viewing.
    session.submit("video-ended", {"video_id": "long-0"}, expected=400)
    session.submit("settings", {"video_id": "long-0", "texture": 0.9, "context": 0.1, "origin": "pad"})
    session.submit("playback", {"video_id": "long-0", "position_ms": 0, "sequence": 0, "playing": True})
    session.submit(
        "exposures",
        {
            "video_id": "long-0",
            "entries": [{"id": "sheet-exposure", "cue": 0, "start_ms": 0, "end_ms": 3000}],
        },
    )
    finish_viewing(app, session)
    assert session.state["poststudy"]["interview_prompts"] == POSTSTUDY["interview_prompts"]
    exported = app.state.store.export(session.token)
    assert len(exported["state"]["completions"]) == 3
    assert exported["state"]["video_surveys"] == []
    assert "final_survey" not in exported["state"]
    assert exported["state"]["sheet_instrument"]["hash"] == "sheet-fixture-hash"
    assert exported["exposures"][0]["id"] == "sheet-exposure"
    actions = [
        json.loads(event["body"]).get("action") for event in exported["events"] if event["body"] != "fixture"
    ]
    assert {"settings", "playback", "exposures", "video-ended"} <= set(actions)


@pytest.mark.parametrize(
    "action,stage", [("video-survey", "video-survey"), ("final-survey", "final"), ("draft", "final")]
)
def test_sheet_policy_rejects_old_survey_routes_even_with_stale_stage(
    assets: Any, tmp_path: Path, action: str, stage: str
) -> None:
    app, session = make_session(assets, tmp_path)
    app.state.store.mutate(
        session.token,
        "stale-stage-fixture",
        session.state["revision"],
        "fixture",
        lambda state: state.update(stage=stage),
    )
    session.state = session.client.get("/api/study/state").json()
    result = session.submit(action, {"video_id": "long-0", "answers": {}}, expected=409)
    assert "survey" in result.json()["error"].lower()


def test_poststudy_acknowledgements_are_required_persisted_and_idempotent(
    assets: Any, tmp_path: Path
) -> None:
    app, session = make_session(assets, tmp_path)
    session.submit("interview-complete", {"researcher_confirmed": True}, expected=409)
    finish_viewing(app, session)
    session.submit("interview-complete", {}, expected=400)
    session.submit(
        "interview-complete", {"researcher_confirmed": True, "D1": "invented transcript"}, expected=400
    )
    payload = {
        "key": "interview-complete-once",
        "revision": session.state["revision"],
        "data": {"researcher_confirmed": True},
    }
    first = session.client.post("/api/study/interview-complete", json=payload)
    retry = session.client.post("/api/study/interview-complete", json=payload)
    assert first.status_code == retry.status_code == 200
    session.state = retry.json()
    assert session.state["stage"] == "debrief"
    session.submit("debrief-complete", {"researcher_confirmed": True}, expected=400)
    session.submit("debrief-complete", {"researcher_confirmed": True, "participant_acknowledged": True})
    assert session.state["stage"] == "done"
    exported = app.state.store.export(session.token)
    assert exported["state"]["poststudy"]["interview_completion"]["researcher_confirmed"] is True
    assert exported["state"]["poststudy"]["debrief_completion"]["participant_acknowledged"] is True
    assert "recording" not in exported["state"]["poststudy"]
    assert "transcript" not in exported["state"]["poststudy"]
    completed = [event for event in exported["events"] if '"action": "interview-complete"' in event["body"]]
    assert len(completed) == 1


def test_unconfigured_debrief_cannot_be_marked_complete(assets: Any, tmp_path: Path) -> None:
    app, session = make_session(assets, tmp_path)
    finish_viewing(app, session)
    session.submit("interview-complete", {"researcher_confirmed": True})

    def remove_text(state: dict[str, Any]) -> None:
        state["poststudy"]["debrief"] = {}

    app.state.store.mutate(
        session.token, "unconfigured-debrief", session.state["revision"], "fixture", remove_text
    )
    session.state = session.client.get("/api/study/state").json()
    session.submit(
        "debrief-complete", {"researcher_confirmed": True, "participant_acknowledged": True}, expected=400
    )
    assert app.state.store.state(session.token)["stage"] == "debrief"
