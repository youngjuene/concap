"""Live phase 2: steps, latency targeting, supersession, the retry rule and the attempt log."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from diff.caption_sources import CaptionSource, parse_sources
from diff.claude_writer import request_body
from diff.job_plan import Job, live_job, make_job, trida_call_settings, writer_call_settings
from diff.live_generation import (
    LatencyEstimator,
    LiveScheduler,
    LiveService,
    LiveStream,
    Spec,
    opaque_token,
    slider_steps,
    starting_caption_problems,
)
from diff.model_profile import CLAUDE_WRITER, TRIDA_7B
from diff.rewrite_prompts import RenderedPrompt
from diff.study_protocol import WRITER_CONDITION

VIDEO = "live_test_video"


def stream(cues: int = 12, cue_ms: int = 5000) -> LiveStream:
    document = {
        "schema": "diff.caption-sources/v1",
        "sources": [
            {
                "schema": "diff.caption-source/v1",
                "phase": 2,
                "unit": {"kind": "video", "key": VIDEO},
                "cue_index": index,
                "start_ms": index * cue_ms,
                "end_ms": (index + 1) * cue_ms,
                "text": f"차가 지나간다 {index}.",
                "evidence": {"events": []},
                "provenance": {
                    "text": "synthetic",
                    "evidence": "synthetic",
                    "reviewed_by": [],
                    "reviewed_at": None,
                    "origin": "test",
                },
                "status": "draft",
                "material": "synthetic",
            }
            for index in range(cues)
        ],
    }
    return LiveStream.from_sources(parse_sources(document).sources, VIDEO)


def job_for(source: CaptionSource, spec: Spec) -> Job:
    prompt = RenderedPrompt("test-live", 1, "0" * 64, 1, None, f"{source.text}|{spec}")
    cell = {
        "cell_id": f"t{spec.texture}-{spec.context}",
        "quality_limit": spec.texture,
        "context": spec.context,
    }
    return make_job(source, "smoke", "trida_tau_090", cell, TRIDA_7B, prompt, trida_call_settings(96, 0.9))


def completed(text: str, latency_ms: int = 8000) -> dict[str, Any]:
    return {"status": "completed", "output": {"text": text}, "latency_ms": latency_ms, "seed": 1}


@pytest.mark.parametrize(
    ("values", "steps"),
    [
        ((0.0, 0.0), (0, "broad")),
        ((0.5, 0.5), (1, "specific")),
        ((1.0, 1.0), (2, "relation")),
        ((0.62, 0.34), (1, "specific")),
    ],
)
def test_sliders_snap_to_prompt_steps(values: tuple[float, float], steps: tuple[int, str]) -> None:
    assert slider_steps(*values) == steps


@pytest.mark.parametrize("bad", [-0.1, 1.2, True, "0.5"])
def test_slider_values_outside_0_to_1_are_refused(bad: Any) -> None:
    with pytest.raises(ValueError, match="from 0 to 1"):
        slider_steps(bad, 0.5)


def test_latency_estimate_starts_at_1s_and_follows_the_device() -> None:
    # Researcher, 2026-10-12: no floor, so a fast model shows a tweak at the next boundary.
    estimator = LatencyEstimator()
    assert estimator.estimate_ms() == 1000
    for _ in range(5):
        estimator.observe(4000)
    assert estimator.estimate_ms() == 4000
    assert LatencyEstimator(floor_ms=6000).estimate_ms() == 6000  # a floor still applies when asked for
    for latency in [8000] * 9 + [12000]:
        estimator.observe(latency)
    assert estimator.estimate_ms() == 8000  # p90 of the last 20: 5 × 4 s, 9 × 8 s, 1 × 12 s
    estimator.observe(12000)
    estimator.observe(12000)
    assert estimator.estimate_ms() == 12000


def test_a_video_opens_with_rewrites_of_its_first_cues() -> None:
    scheduler = LiveScheduler(stream(), job_for)
    assert scheduler.next_request(0.0) is None  # nothing before the calibrated settings arrive
    scheduler.set_spec(Spec(2, "specific", "guidance"), 0, 0.0)
    requests = [scheduler.next_request(0.0) for _ in range(4)]
    assert [r.cue for r in requests if r] == [0, 1, 2] and requests[3] is None


def test_a_tweak_targets_the_cue_playing_one_latency_later() -> None:
    # Researcher, 2026-10-21: a change shows as soon as it is written, mid-cue.
    scheduler = LiveScheduler(stream(), job_for)
    scheduler.set_spec(Spec(2, "specific"), 0, 0.0)
    scheduler.set_playback(10_000, True, 100.0)
    scheduler.set_spec(Spec(4, "relation"), 10_000, 100.0)
    request = scheduler.next_request(100.0)
    # 10 s + L (1 s) = 11 s, inside cue 2 (10-15 s) with 4 s to go: cue 2 itself
    assert request is not None and request.target_ms == 11_000 and request.cue == 2
    assert request.revision == 2 and request.spec == Spec(4, "relation")
    # 4 s later the playhead is at 14 s: 15 s is the start of cue 3
    later = scheduler.next_request(104.0)
    assert later is not None and later.cue == 3


def test_a_tweak_too_close_to_a_cue_s_end_goes_to_the_next_cue() -> None:
    # Researcher, 2026-10-21: every caption stays on screen at least 3 s.
    scheduler = LiveScheduler(stream(), job_for)
    scheduler.set_playback(11_200, True, 0.0)
    scheduler.set_spec(Spec(1, "broad"), 11_200, 0.0)
    request = scheduler.next_request(0.0)
    # 11.2 s + 1 s = 12.2 s: cue 2 ends at 15 s, 2.8 s left is under MIN_SHOWN_MS (3 s reading + 0.5 s poll)
    assert request is not None and request.cue == 3


STARTING = f"starting-caption:{VIDEO}"


def test_each_cue_shows_its_own_caption_and_an_older_spec_comes_back_from_the_cache() -> None:
    # Researcher, 2026-10-20: a cue shows its caption under the current spec, else under earlier
    # settings, else its starting caption; never another cue's caption.
    scheduler = LiveScheduler(stream(), job_for)
    assert scheduler.ready() == {}  # no settings yet
    scheduler.set_spec(Spec(2, "specific"), 0, 0.0)
    first = scheduler.next_request(0.0)
    assert first is not None
    scheduler.set_spec(Spec(0, "broad"), 0, 1.0)  # the participant moved on during the call
    scheduler.complete(first, completed("이전 설정의 자막"))
    ready = scheduler.ready()
    assert ready[0] == {
        "job_id": first.job.job_id,
        "text": "이전 설정의 자막",
        "current": False,
        "settled": False,
    }
    assert ready[1] == {
        "job_id": f"{STARTING}:1",
        "text": "차가 지나간다 1.",
        "current": False,
        "settled": False,
    }
    scheduler.set_spec(Spec(2, "specific"), 0, 2.0)  # and came back
    assert scheduler.ready()[0] == {**ready[0], "current": True, "settled": True}
    assert scheduler.next_request(2.0).cue == 1  # type: ignore[union-attr]


def test_until_its_rewrite_is_ready_a_cue_keeps_its_own_caption_under_the_old_settings() -> None:
    scheduler = LiveScheduler(stream(), job_for)
    scheduler.set_spec(Spec(1, "specific"), 0, 0.0)
    for _ in range(3):  # the worker wrote cues 0-2 ahead under the starting settings
        request = scheduler.next_request(0.0)
        assert request is not None
        scheduler.complete(request, completed(f"이전 설정 {request.cue}", latency_ms=900))
    scheduler.set_playback(8500, True, 0.0)
    scheduler.set_spec(Spec(2, "relation"), 8500, 0.0)  # late in cue 1: the tweak goes to cue 2
    request = scheduler.next_request(0.0)
    assert request is not None and request.cue == 2
    shown = scheduler.ready()[2]
    assert shown["text"] == "이전 설정 2" and shown["current"] is False  # cue 2's own caption, not cue 1's


@pytest.mark.parametrize(
    "checks",
    [
        {"failures": [{"code": "undecodable_bytes", "detail": "1"}], "flags": []},
        {"failures": [], "flags": [{"code": "repetition", "detail": "가로 가로 가로"}]},
    ],
)
def test_an_unusable_caption_gives_way_to_the_starting_caption(checks: dict[str, Any]) -> None:
    scheduler = LiveScheduler(stream(), job_for)
    scheduler.set_spec(Spec(2, "specific"), 0, 0.0)
    request = scheduler.next_request(0.0)
    assert request is not None
    scheduler.complete(request, {**completed("자막: 차가 지\ufffd나간다"), "checks": checks})
    assert scheduler.ready()[0] == {
        "job_id": f"{STARTING}:0",
        "text": "차가 지나간다 0.",
        "current": False,
        "settled": True,  # nothing better is coming, so the video may start on it
    }


def test_a_usable_caption_is_shown_under_the_display_rule() -> None:
    # The display rule (2026-10-20) removes the leading label and the wrapping quotes, nothing inside.
    scheduler = LiveScheduler(stream(), job_for)
    scheduler.set_spec(Spec(2, "specific"), 0, 0.0)
    request = scheduler.next_request(0.0)
    assert request is not None
    scheduler.complete(request, completed("자막: 「차가 천천히 지나간다」"))
    assert scheduler.ready()[0]["text"] == "차가 천천히 지나간다" and scheduler.ready()[0]["current"]


@pytest.mark.parametrize("written", ["자막:", "「」", "...", "  ", ""])
def test_a_caption_never_goes_blank(written: str, tmp_path: Path) -> None:
    # Researcher, 2026-10-20: nothing readable left of Trida's text shows the cue's starting caption.
    service = LiveService(FakeAttempt([written]), tmp_path, check=None, clock=lambda: 0.0)
    scheduler = LiveScheduler(stream(cues=1), job_for)
    service.attach("session-a", scheduler)
    service.settings(Spec(1, "specific", "참고 문장"), 0)
    assert service.step()
    assert service.ready()[0] == {
        "job_id": f"{STARTING}:0",
        "text": "차가 지나간다 0.",
        "current": False,
        "settled": True,
    }
    line = json.loads((tmp_path / "live-session-a.jsonl").read_text(encoding="utf-8"))
    assert line["output"]["text"] == written  # Trida's own text stays in the log
    assert line["usable"] is False


@pytest.mark.parametrize(
    ("status", "retried"),
    [("worker_lost", True), ("device_error", True), ("timeout", False), ("runner_error", False)],
)
def test_only_a_transient_failure_earns_one_identical_retry(status: str, retried: bool) -> None:
    scheduler = LiveScheduler(stream(cues=1), job_for)
    scheduler.set_spec(Spec(2, "specific"), 0, 0.0)
    first = scheduler.next_request(0.0)
    assert first is not None
    scheduler.complete(first, {"status": status})
    second = scheduler.next_request(0.0)
    assert (second is not None) is retried
    if second is not None:
        assert second.job.job_id == first.job.job_id
        scheduler.complete(second, {"status": status})
        assert scheduler.next_request(0.0) is None  # never a second retry


def test_opaque_tokens_hide_job_ids_and_differ_per_session() -> None:
    token = opaque_token(b"secret", "session-a", "job-123")
    assert token == opaque_token(b"secret", "session-a", "job-123")
    assert token != opaque_token(b"secret", "session-b", "job-123") and "job" not in token


class FakeAttempt:
    def __init__(self, texts: list[str]) -> None:
        self.texts, self.calls = texts, []

    def __call__(self, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        self.calls.append((payload["job_id"], timeout))
        return completed(self.texts[len(self.calls) - 1], latency_ms=9000)


def test_the_service_runs_requests_and_logs_every_attempt(tmp_path: Path) -> None:
    attempt = FakeAttempt(["자막: 「첫 자막」", "둘째 자막", "셋째 자막"])
    now = [0.0]
    service = LiveService(
        attempt, tmp_path, check=lambda source, job, result: {"ok": True}, clock=lambda: now[0]
    )
    scheduler = LiveScheduler(stream(), job_for)
    service.attach("session-a", scheduler)
    with pytest.raises(RuntimeError, match="already serves"):
        service.attach("session-b", LiveScheduler(stream(), job_for))
    service.settings(Spec(2, "specific", "참고 문장"), 0)
    assert service.step() and service.step() and service.step() and not service.step()
    assert [cue for cue, entry in sorted(service.ready().items()) if entry["current"]] == [0, 1, 2]
    assert service.ready()[3]["job_id"] == f"{STARTING}:3"  # the rest show their starting captions
    assert service.ready()[0]["text"] == "첫 자막"  # shown under the display rule
    assert attempt.calls[0][1] == 30.0
    lines = [
        json.loads(line)
        for line in (tmp_path / "live-session-a.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(lines) == 3 and lines[0]["schema"] == "diff.live-attempt/v5"
    assert lines[0]["output"]["text"] == "자막: 「첫 자막」" and lines[0]["shown_text"] == "첫 자막"
    assert lines[0]["usable"] is True
    assert lines[0]["display_rule"] == "diff.display/v2"
    assert (
        lines[0]["cue"] == 0
        and lines[0]["system_message"] == "참고 문장"
        and lines[0]["checks"] == {"ok": True}
    )
    assert lines[0]["latency_estimate_ms"] == 1000 and lines[2]["latency_estimate_ms"] == 9000
    service.detach("session-a")
    assert service.free


def test_live_jobs_take_the_writer_s_call_under_the_tuner() -> None:
    # Researcher, 2026-10-27: phase 2's live captions take phase 1's writer call (Trida at τ 0.9 before).
    job = live_job(stream().cues[0], 2, "specific", "당신은 오디오 자막 조정기입니다.", "smoke")
    assert (job.profile_id, job.runner) == (CLAUDE_WRITER.profile_id, "claude")
    assert job.condition == WRITER_CONDITION and job.generate == writer_call_settings()
    assert job.cell == {"cell_id": "t2-specific", "quality_limit": 2, "context": "specific"}
    assert request_body(job)["system"] == "당신은 오디오 자막 조정기입니다."  # the tuner is the system prompt


def test_starting_captions_longer_than_the_live_limit_are_named() -> None:
    long_stream = stream(cues=2)
    assert starting_caption_problems(long_stream, 96) == []
    assert starting_caption_problems(long_stream, 5) == [
        f"{VIDEO} cue 0: 10 characters, over 5",
        f"{VIDEO} cue 1: 10 characters, over 5",
    ]
