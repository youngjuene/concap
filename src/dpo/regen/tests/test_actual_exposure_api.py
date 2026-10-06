"""Actual display intervals retain verifiable caption identity across cue boundaries.

The existing integration fixture never enters FastAPI lifespan. Jobs are claimed
and completed directly, so these tests do not start an inference process/model.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest
from fastapi.testclient import TestClient

from dpo.regen.study_schema import digest
from dpo.regen.study_store import StudyStore
from dpo.regen.tests import test_study

assets = test_study.assets
session = test_study.session

GENERATED_TEXT = "A steady engine hum passes nearby."
Watch = tuple[test_study.Session, StudyStore]


@pytest.fixture
def watching(session: Any) -> Watch:
    participant, store, _ = session
    participant.calibrate()
    participant.submit("start-viewing")
    return participant, store


def claim(watching: Watch, *, cue: int = 0, content_key: str | None = None) -> dict[str, Any]:
    participant, store = watching
    for _ in range(32):
        job = store.claim(previous_session=None)
        assert job is not None, "Expected an API-enqueued, model-free caption job"
        if (
            job["session"] == participant.token
            and job["cue"] == cue
            and (content_key is None or job["content_key"] == content_key)
        ):
            return job
        store.finish(job, {"text": "Unused fixture job.", "fallback": True, "reason": "fixture"})
    pytest.fail("Requested fixture caption job was not claimed")


def finish(watching: Watch, *, failed: bool = False) -> dict[str, Any]:
    job = claim(watching)
    result = {"text": GENERATED_TEXT, "fallback": False}
    if failed:
        result = {
            "text": json.loads(job["spec"])["fallback"],
            "fallback": True,
            "reason": "inference_deadline",
        }
    watching[1].finish(job, result)
    return job


def entry(
    watching: Watch,
    job: dict[str, Any] | None,
    *,
    fallback: bool = False,
    reason: str = "not_ready_at_boundary",
    cue: int = 0,
    start: float | None = None,
    end: float | None = None,
) -> dict[str, Any]:
    participant, store = watching
    held = store.state(participant.token)
    video = held["viewing"][held["video_index"]]
    nominal = video["cues"][cue]
    value = {
        "id": uuid.uuid4().hex,
        "video_id": video["id"],
        "display_interval_version": 2,
        "cue": cue,
        "cue_start_ms": nominal["start_ms"],
        "cue_end_ms": nominal["end_ms"],
        "start_ms": nominal["start_ms"] + 100 if start is None else start,
        "end_ms": nominal["start_ms"] + 900 if end is None else end,
        "text": nominal["fallback"][held["language"]] if fallback else GENERATED_TEXT,
        "fallback": fallback,
        "job_id": job["id"] if job is not None else None,
        "settings_revision": job["revision"] if job is not None else None,
        "axes": None if fallback else json.loads(job["spec"])["axes"] if job else None,
    }
    if fallback:
        value["fallback_reason"] = reason
    return value


def submit(watching: Watch, values: list[dict[str, Any]], expected: int = 200) -> None:
    participant, _ = watching
    participant.submit(
        "exposures", {"video_id": participant.state["video"]["id"], "entries": values}, expected
    )


def stored(watching: Watch) -> list[dict[str, Any]]:
    participant, store = watching
    return list(store.export(participant.token)["exposures"])


@pytest.mark.parametrize("start,end", [(4500, 6200), (5100, 6200), (299500, 300000)])
def test_v2_generated_display_may_outlive_its_nominal_cue(watching: Watch, start: float, end: float) -> None:
    value = entry(watching, finish(watching), start=start, end=end)
    submit(watching, [value])
    assert stored(watching) == [value]


def test_v2_fallback_display_may_outlive_its_nominal_cue(watching: Watch) -> None:
    value = entry(watching, None, fallback=True, start=4900, end=7300)
    submit(watching, [value])
    assert stored(watching) == [value]


@pytest.mark.parametrize("version", [None, 1])
def test_legacy_minimal_exposure_keeps_its_original_window_rule(watching: Watch, version: int | None) -> None:
    value = {"id": uuid.uuid4().hex, "cue": 0, "start_ms": 100, "end_ms": 1000, "fallback": True}
    if version is not None:
        value["display_interval_version"] = version
    submit(watching, [value])
    submit(watching, [{**value, "id": uuid.uuid4().hex, "end_ms": 6200}], expected=400)
    assert stored(watching) == [{**value, "video_id": "long-0"}]


@pytest.mark.parametrize("change", [{"start_ms": -1}, {"end_ms": 300001}, {"end_ms": 50}])
def test_v2_still_rejects_invalid_actual_video_intervals(watching: Watch, change: dict[str, Any]) -> None:
    submit(watching, [{**entry(watching, finish(watching)), **change}], expected=400)
    assert stored(watching) == []


def test_v2_display_cannot_begin_before_its_caption_source_cue(watching: Watch) -> None:
    job = claim(watching, cue=1)
    watching[1].finish(job, {"text": GENERATED_TEXT, "fallback": False})
    submit(watching, [entry(watching, job, cue=1, start=4900, end=6200)], expected=400)
    assert stored(watching) == []


@pytest.mark.parametrize("field", ["cue_start_ms", "cue_end_ms"])
@pytest.mark.parametrize("missing", [False, True])
def test_v2_requires_exact_nominal_cue_metadata(watching: Watch, field: str, missing: bool) -> None:
    value = entry(watching, finish(watching))
    if missing:
        value.pop(field)
    else:
        value[field] += 1
    submit(watching, [value], expected=400)
    assert stored(watching) == []


@pytest.mark.parametrize(
    "change",
    [
        {"text": "An invented caption never returned by this job."},
        {"axes": {"texture": 0.13, "context": 0.27}},
        {"axes": None},
        {"settings_revision": 100},
        {"settings_revision": None},
        {"job_id": "missing-job"},
        {"job_id": None},
    ],
)
def test_v2_generated_exposure_cannot_forge_its_caption_source(
    watching: Watch, change: dict[str, Any]
) -> None:
    submit(watching, [{**entry(watching, finish(watching)), **change}], expected=400)
    assert stored(watching) == []


@pytest.mark.parametrize("failed", [False, True])
def test_v2_generated_exposure_requires_a_successful_result(watching: Watch, failed: bool) -> None:
    job = finish(watching, failed=True) if failed else claim(watching)
    submit(watching, [entry(watching, job)], expected=400)
    assert stored(watching) == []


@pytest.mark.parametrize("fallback", [False, True])
def test_v2_job_must_belong_to_the_claimed_cue(watching: Watch, fallback: bool) -> None:
    job = finish(watching)
    submit(watching, [entry(watching, job, fallback=fallback, cue=1)], expected=400)
    assert stored(watching) == []


@pytest.mark.parametrize("fallback", [False, True])
def test_v2_job_must_belong_to_the_current_video(watching: Watch, fallback: bool) -> None:
    participant, store = watching
    original = finish(watching, failed=True)
    held = store.state(participant.token)
    spec = {**json.loads(original["spec"]), "video_hash": digest(held["viewing"][1])}
    key = uuid.uuid4().hex
    store.enqueue(participant.token, held, 0, key, spec, limit=32)
    job = claim(watching, content_key=key)
    store.finish(job, {"text": GENERATED_TEXT, "fallback": False})
    submit(watching, [entry(watching, job, fallback=fallback)], expected=400)
    assert stored(watching) == []


@pytest.mark.parametrize("fallback", [False, True])
def test_v2_job_must_belong_to_the_same_participant(watching: Watch, fallback: bool) -> None:
    participant, store = watching
    foreign = test_study.Session(TestClient(participant.client.app))
    foreign.calibrate()
    foreign.submit("start-viewing")
    job = claim((foreign, store))
    store.finish(job, {"text": GENERATED_TEXT, "fallback": False})
    submit(watching, [entry(watching, job, fallback=fallback)], expected=400)
    assert stored(watching) == []


@pytest.mark.parametrize("fallback", [False, True])
def test_v2_keeps_the_displayed_job_revision_after_controls_change(watching: Watch, fallback: bool) -> None:
    participant, _ = watching
    job = finish(watching, failed=fallback)
    value = entry(watching, job, fallback=fallback, reason="job_failed")
    participant.submit("settings", {"video_id": "long-0", "texture": 0.1, "context": 0.9})
    assert participant.state["settings_revision"] != value["settings_revision"]
    submit(watching, [value])
    assert stored(watching) == [value]


@pytest.mark.parametrize(
    "reason", ["not_ready_at_boundary", "initial_caption_timeout", "caption_fetch_failed"]
)
def test_v2_authored_fallback_can_explain_missing_job(watching: Watch, reason: str) -> None:
    value = entry(watching, None, fallback=True, reason=reason)
    submit(watching, [value])
    assert stored(watching) == [value]


@pytest.mark.parametrize(
    "change",
    [
        {"text": "This is not the reviewed fallback."},
        {"axes": {"texture": 0.75, "context": 0.25}},
        {"fallback_reason": "model_not_configured"},
        {"fallback_reason": None},
        {"job_id": "missing-job", "settings_revision": 0},
    ],
)
def test_v2_fallback_cannot_forge_its_authored_content_or_reason(
    watching: Watch, change: dict[str, Any]
) -> None:
    submit(watching, [{**entry(watching, None, fallback=True), **change}], expected=400)
    assert stored(watching) == []


def test_v2_fallback_reason_is_required(watching: Watch) -> None:
    value = entry(watching, None, fallback=True)
    value.pop("fallback_reason")
    submit(watching, [value], expected=400)
    assert stored(watching) == []


def test_v2_fallback_job_revision_must_match_its_source(watching: Watch) -> None:
    job = finish(watching, failed=True)
    value = entry(watching, job, fallback=True, reason="job_failed")
    value["settings_revision"] = job["revision"] + 1
    submit(watching, [value], expected=400)
    assert stored(watching) == []


def test_v2_job_failed_fallback_is_bound_to_the_failed_job(watching: Watch) -> None:
    value = entry(watching, finish(watching, failed=True), fallback=True, reason="job_failed")
    submit(watching, [value])
    assert stored(watching) == [value]


@pytest.mark.parametrize("job_state", ["missing", "pending", "succeeded"])
def test_v2_job_failed_reason_requires_an_actual_failed_result(watching: Watch, job_state: str) -> None:
    job = None if job_state == "missing" else claim(watching)
    if job_state == "succeeded":
        assert job is not None
        watching[1].finish(job, {"text": GENERATED_TEXT, "fallback": False})
    submit(watching, [entry(watching, job, fallback=True, reason="job_failed")], expected=400)
    assert stored(watching) == []


def test_v2_not_ready_can_name_a_job_that_succeeded_after_display(watching: Watch) -> None:
    job = claim(watching)
    value = entry(watching, job, fallback=True, reason="not_ready_at_boundary")
    watching[1].finish(job, {"text": GENERATED_TEXT, "fallback": False})
    submit(watching, [value])
    assert stored(watching) == [value]
    assert stored(watching)[0]["fallback_reason"] == "not_ready_at_boundary"


def test_v2_not_ready_can_name_an_unfinished_job(watching: Watch) -> None:
    value = entry(watching, claim(watching), fallback=True)
    submit(watching, [value])
    assert stored(watching) == [value]


@pytest.mark.parametrize("version", [None, 2])
def test_exposure_payload_size_limit_is_preserved(watching: Watch, version: int | None) -> None:
    if version == 2:
        value = entry(watching, finish(watching))
    else:
        value = {"id": uuid.uuid4().hex, "cue": 0, "start_ms": 100, "end_ms": 900, "fallback": True}
    value["diagnostic"] = "x" * 3001
    submit(watching, [value], expected=400)
    assert stored(watching) == []


def test_v2_retries_keep_actual_intervals_immutable(watching: Watch) -> None:
    value = entry(watching, finish(watching), start=4900, end=6300)
    submit(watching, [value])
    submit(watching, [value])
    assert stored(watching) == [value]
    submit(watching, [{**value, "end_ms": 6400}], expected=409)
    assert stored(watching) == [value]


def test_v2_source_failure_rejects_the_whole_exposure_batch(watching: Watch) -> None:
    value = entry(watching, finish(watching))
    forged = {**value, "id": uuid.uuid4().hex, "text": "Forged second entry."}
    submit(watching, [value, forged], expected=400)
    assert stored(watching) == []
