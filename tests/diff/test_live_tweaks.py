"""Phase 1's second captions on the session's GPU: queue order, the one retry, the ladder, checks and
the log."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from phase1_fixtures import approved_setup
from test_live_server import FakeAttempt, full_output, job_for, stream
from test_phase1_tweak import first_step

from diff.generation_worker import WorkerStartupError
from diff.live_generation import LiveScheduler, LiveService, Spec, TweakRequest
from diff.phase1_tweak import TWEAK_LADDER, tweak_job_for, tweak_request
from diff.phase1_usable import check_record


class Scripted:
    """A worker that answers each call with the next scripted result (or raises it)."""

    def __init__(self, *results: Mapping[str, Any] | Exception) -> None:
        self.results = list(results)
        self.calls: list[str] = []

    def __call__(self, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append(payload["job_id"])
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return dict(result)


@pytest.fixture
def requests(tmp_path: Path) -> list[TweakRequest]:
    setup = approved_setup(tmp_path)
    made = []
    for clip in setup["clips"]:
        source, job, steer = tweak_request(clip, first_step(), "smoke", root=tmp_path)
        made.append(TweakRequest("s1", clip["key"], source, job, steer))
    return made


def service_with(attempt: Any, tmp_path: Path) -> LiveService:
    service = LiveService(attempt, tmp_path / "logs", tweak_check=check_record, clock=lambda: 0.0)
    assert service.claim("s1")
    return service


def run(service: LiveService) -> None:
    while service.step():
        pass


def written(text: str) -> dict[str, Any]:
    return {"status": "completed", "output": full_output(text)}


# Each fixture clip's second caption, naming its attended sound (s1), as the gate requires.
NAMING = ("버스가 낮게 공회전한다.", "차량이 쉼 없이 오간다.", "새가 높게 지저귄다.")


def last_rung(request: TweakRequest, clip: dict[str, Any], rung: int = len(TWEAK_LADDER) - 1) -> TweakRequest:
    source, job, steer = tweak_job_for(clip, request.steer["slots"], "smoke", rung)
    return TweakRequest(request.session_id, request.clip, source, job, steer)


def test_tweaks_run_in_order_and_only_for_the_claimed_session(
    requests: list[TweakRequest], tmp_path: Path
) -> None:
    service = service_with(Scripted(*(written(text) for text in NAMING)), tmp_path)
    outcomes: list[tuple[str, dict[str, Any]]] = []
    for request in requests:
        assert service.submit_tweak(request, lambda r, o: outcomes.append((r.clip, o)))
    assert not service.submit_tweak(requests[0], lambda r, o: None)  # already queued
    stranger = TweakRequest("s2", "x", requests[0].source, requests[0].job, {})
    assert not service.submit_tweak(stranger, lambda r, o: None)
    run(service)
    assert [clip for clip, _ in outcomes] == ["kyoto_101", "lima_202", "oslo_303"]
    assert [(o["status"], o["text"]) for _, o in outcomes] == [("ready", text) for text in NAMING]
    lines = [json.loads(line) for line in (tmp_path / "logs" / "live-s1.jsonl").read_text().splitlines()]
    assert [line["kind"] for line in lines] == ["tweak"] * 3
    assert lines[0]["steer"]["slots"]["mode"] == "scene" and lines[0]["outcome"] == "ready"
    assert lines[0]["checks"]["profile"] == "tweak-v3"


def test_a_refused_caption_goes_up_the_ladder_and_fails_only_on_the_last_rung(
    requests: list[TweakRequest], tmp_path: Path
) -> None:
    clip = approved_setup(tmp_path)["clips"][0]
    service = service_with(Scripted(written("bus noise"), written("bus noise")), tmp_path)
    outcomes: list[dict[str, Any]] = []
    service.submit_tweak(requests[0], lambda r, o: outcomes.append(o))
    run(service)  # the flow queues the next rung itself (participant_server.tweak_done)
    service.submit_tweak(last_rung(requests[0], clip), lambda r, o: outcomes.append(o))
    run(service)
    reason = "check failed: non_korean_script, attended_unnamed"
    assert [(o["rung"], o["status"], o["reason"]) for o in outcomes] == [
        (0, "next", reason),
        (len(TWEAK_LADDER) - 1, "failed", reason),
    ]
    assert outcomes[0]["job_id"] == requests[0].job.job_id != outcomes[1]["job_id"]


def test_the_first_wording_s_second_try_is_queued_while_its_first_is_running(
    requests: list[TweakRequest], tmp_path: Path
) -> None:
    clip = approved_setup(tmp_path)["clips"][0]
    again = last_rung(requests[0], clip, 1)
    assert again.job.job_id == requests[0].job.job_id  # the same prompt, at the next rung
    service = service_with(Scripted(written("bus noise"), written(NAMING[0])), tmp_path)
    outcomes: list[dict[str, Any]] = []

    def done(request: TweakRequest, outcome: dict[str, Any]) -> None:
        outcomes.append(outcome)
        if outcome["status"] == "next":  # as participant_server.tweak_done, before the service moves on
            assert service.submit_tweak(again, done)
            assert not service.submit_tweak(again, done)  # but only once

    service.submit_tweak(requests[0], done)
    run(service)
    assert [(o["rung"], o["status"]) for o in outcomes] == [(0, "next"), (1, "ready")]


def test_an_attempt_that_already_ran_is_never_queued_again(
    requests: list[TweakRequest], tmp_path: Path
) -> None:
    """A page poll reads the pending attempt, and the attempt may finish before the poll submits it
    (the 2026-10-27 walk ran a delivered caption's retry twice); the stale submission is refused."""
    lost = {"status": "worker_lost", "error": "gone"}
    service = service_with(Scripted(lost, written(NAMING[0])), tmp_path)
    outcomes: list[dict[str, Any]] = []
    service.submit_tweak(requests[0], lambda r, o: outcomes.append(o))
    run(service)
    assert [o["status"] for o in outcomes] == ["retry", "ready"]
    retried = replace(requests[0], attempt=2)
    for stale in (requests[0], retried):  # what the poll read before each finished
        assert not service.submit_tweak(stale, lambda r, o: outcomes.append(o))
    assert not service.step() and len(outcomes) == 2


def test_the_second_caption_is_stored_as_shown_and_logged_as_written(
    requests: list[TweakRequest], tmp_path: Path
) -> None:
    labelled = {"status": "completed", "output": full_output("자막: 「버스가 낮게 공회전한다.」")}
    copied = {"status": "completed", "output": full_output(f"「{requests[1].source.text}」")}
    service = service_with(Scripted(labelled, copied), tmp_path)
    outcomes: list[dict[str, Any]] = []
    for request in requests[:2]:
        service.submit_tweak(request, lambda r, o: outcomes.append(o))
    run(service)
    assert outcomes[0]["status"] == "ready" and outcomes[0]["text"] == "버스가 낮게 공회전한다."
    flags = [finding["code"] for finding in outcomes[1]["checks"]["flags"]]
    assert "identical_to_source" in flags  # a quoted copy is still a copy
    lines = [json.loads(line) for line in (tmp_path / "logs" / "live-s1.jsonl").read_text().splitlines()]
    assert lines[0]["schema"] == "diff.tweak-attempt/v3"
    assert lines[0]["output"]["text"] == "자막: 「버스가 낮게 공회전한다.」"
    assert (
        lines[0]["shown_text"] == "버스가 낮게 공회전한다." and lines[0]["display_rule"] == "diff.display/v2"
    )


@pytest.mark.parametrize(
    ("written", "reason"),
    [("자막:", "check failed: empty_text"), ("「...」", "nothing readable after the display rule")],
)
def test_a_second_caption_with_nothing_readable_skips_the_step(
    requests: list[TweakRequest], tmp_path: Path, written: str, reason: str
) -> None:
    # Researcher, 2026-10-20: a caption never goes blank; a blank second caption is refused instead.
    clip = approved_setup(tmp_path)["clips"][0]
    output = {"status": "completed", "output": full_output(written)}
    service = service_with(Scripted(output, output), tmp_path)
    outcomes: list[dict[str, Any]] = []
    for request in (requests[0], last_rung(requests[0], clip)):
        service.submit_tweak(request, lambda r, o: outcomes.append(o))
        run(service)
    assert [(o["status"], o["reason"]) for o in outcomes] == [("next", reason), ("failed", reason)]
    assert "text" not in outcomes[0]


def test_one_identical_retry_after_a_transient_failure_and_none_after_a_timeout(
    requests: list[TweakRequest], tmp_path: Path
) -> None:
    attempt = Scripted(
        {"status": "worker_lost", "error": "pipe closed"},
        {"status": "worker_lost", "error": "pipe closed"},
        WorkerStartupError("no GPU memory"),
        {"status": "completed", "output": full_output("차량 소리가 들린다.")},
        {"status": "timeout", "error": "30 s"},
    )
    service = service_with(attempt, tmp_path)
    outcomes: list[tuple[str, int, str]] = []
    for request in requests:
        service.submit_tweak(request, lambda r, o: outcomes.append((r.clip, o["attempt"], o["status"])))
    run(service)
    assert outcomes == [
        ("kyoto_101", 1, "retry"),
        ("kyoto_101", 2, "failed"),
        ("lima_202", 1, "retry"),  # a worker that cannot start is a lost worker
        ("lima_202", 2, "ready"),
        ("oslo_303", 1, "failed"),
    ]
    assert attempt.calls[0] == attempt.calls[1] == requests[0].job.job_id


def test_a_video_s_requests_come_before_tweaks_and_detach_drops_them(
    requests: list[TweakRequest], tmp_path: Path
) -> None:
    fake = FakeAttempt()
    service = LiveService(fake, tmp_path / "logs", tweak_check=check_record, clock=lambda: 0.0)
    service.attach("s1", LiveScheduler(stream(cues=2), job_for))
    service.settings(Spec(1, "specific", "시스템"), 0)
    service.submit_tweak(requests[0], lambda r, o: None)
    service.step()
    service.step()
    assert len(fake.calls) == 2 and requests[0].job.job_id not in fake.calls
    service.detach("s1")
    assert not service.tweaks
