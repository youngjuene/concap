"""Phase 1's second-caption writer on the Claude API: the request a job sends, what a response becomes,
and the status each failure earns, against a fake client (no network, no key)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from phase1_fixtures import approved_setup
from test_live_tweaks import requests, run, service_with  # noqa: F401 — the fixture
from test_phase1_tweak import first_step

from diff.claude_writer import ClaudeWriter, request_body
from diff.live_generation import TweakRequest
from diff.model_profile import TRIDA_7B
from diff.phase1_tweak import tweak_request
from diff.phase1_usable import check_record
from diff.study_protocol import sha256_of

REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def answered(error: type[anthropic.APIStatusError], code: int) -> anthropic.APIStatusError:
    """An API error as the SDK raises it for an HTTP status."""
    return error(str(code), response=httpx2.Response(code, request=REQUEST), body=None)


def message(
    text: str = " 버스가 낮게 공회전한다. ",
    stop_reason: str = "end_turn",
    *,
    model: str = "claude-sonnet-5-5",
    iterations: list[Any] | None = None,
    category: str | None = None,
) -> SimpleNamespace:
    """A response as the SDK returns it: an empty thinking block first, as on this model."""
    return SimpleNamespace(
        id="msg_1",
        _request_id="req_1",
        model=model,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if category else None,
        usage=SimpleNamespace(input_tokens=900, output_tokens=40, iterations=iterations),
    )


class FakeClient:
    """``Anthropic`` as the writer uses it: ``with_options`` then ``beta.messages.create``."""

    def __init__(self, *answers: Any) -> None:
        self.answers = list(answers)
        self.bodies: list[dict[str, Any]] = []
        self.timeouts: list[float] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def with_options(self, *, timeout: float) -> FakeClient:
        self.timeouts.append(timeout)
        return self

    def create(self, **body: Any) -> Any:
        self.bodies.append(body)
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def tweak(root: Path) -> tuple[Any, Any, Mapping[str, Any]]:
    """kyoto_101's second-caption request, scene-aware."""
    clip = approved_setup(root)["clips"][0]
    return tweak_request(clip, first_step(), "smoke", root=root)


def test_the_request_is_the_job_s_settings_its_system_message_and_its_prompt(tmp_path: Path) -> None:
    source, job, steer = tweak(tmp_path)
    client = FakeClient(message())
    result = ClaudeWriter(client).attempt(job.record(), 30.0)
    body = client.bodies[0]
    assert (
        body
        == request_body(job)
        == {
            **job.generate,
            "system": job.prompt.system_message,
            "messages": [{"role": "user", "content": job.prompt.text}],
        }
    )
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(body)  # API 2026-10-21; adaptive
    assert client.timeouts == [15.0]  # half the deadline, with one SDK retry inside it
    assert result["status"] == "completed" and result["output"]["text"] == "버스가 낮게 공회전한다."
    assert (result["output"]["completion"], result["output"]["terminator"]) == ("terminated", "end_turn")
    assert result["api"] == {
        "sdk": anthropic.__version__,
        "request_sha256": sha256_of(body),
        "request_id": "req_1",
        "message_id": "msg_1",
        "model": "claude-sonnet-5-5",
        "fallback": False,
        "stop_reason": "end_turn",
        "refusal_category": None,
        "input_tokens": 900,
        "output_tokens": 40,
    }
    assert not check_record(source, steer, result)["excluded"]


def test_an_output_cut_at_max_tokens_is_incomplete_so_the_gate_refuses_it(tmp_path: Path) -> None:
    source, job, steer = tweak(tmp_path)
    result = ClaudeWriter(FakeClient(message(stop_reason="max_tokens"))).attempt(job.record(), 30.0)
    assert result["status"] == "completed" and result["output"]["completion"] == "incomplete"
    record = check_record(source, steer, result)
    assert record["excluded"] and "incomplete" in [f["code"] for f in record["failures"]]


def test_a_declined_request_is_refused_and_a_fallback_is_recorded(tmp_path: Path) -> None:
    _, job, _ = tweak(tmp_path)
    declined = ClaudeWriter(FakeClient(message("", "refusal", category="cyber"))).attempt(job.record(), 30.0)
    assert declined["status"] == "refused" and "declined (cyber)" in declined["error"]
    rescued = message(model="claude-sonnet-5", iterations=[SimpleNamespace(type="fallback_message")])
    served = ClaudeWriter(FakeClient(rescued)).attempt(job.record(), 30.0)
    assert served["status"] == "completed" and served["api"]["fallback"]
    assert served["api"]["model"] == "claude-sonnet-5"


@pytest.mark.parametrize(
    ("error", "status"),
    [
        (anthropic.APITimeoutError(request=REQUEST), "api_unavailable"),
        (anthropic.APIConnectionError(request=REQUEST), "api_unavailable"),
        (answered(anthropic.RateLimitError, 429), "api_unavailable"),
        (answered(anthropic.InternalServerError, 529), "api_unavailable"),
        (answered(anthropic.AuthenticationError, 401), "refused"),
        (answered(anthropic.BadRequestError, 400), "refused"),
        (TypeError("Could not resolve authentication method"), "refused"),  # no key in the environment
    ],
)
def test_a_busy_or_unreachable_api_is_transient_and_any_other_failure_is_refused(
    tmp_path: Path, error: Exception, status: str
) -> None:
    _, job, _ = tweak(tmp_path)
    result = ClaudeWriter(FakeClient(error)).attempt(job.record(), 30.0)
    assert result["status"] == status and type(error).__name__ in result["error"]
    assert result["api"] == {"request_sha256": sha256_of(request_body(job))} and "output" not in result


def test_only_the_writer_s_own_intact_jobs_are_sent(tmp_path: Path) -> None:
    _, job, _ = tweak(tmp_path)
    client = FakeClient()
    tampered = {**job.record(), "prompt": {**job.record()["prompt"], "text": "다른 프롬프트"}}
    trida = replace(job, profile_id=TRIDA_7B.profile_id, runner=TRIDA_7B.runner)
    for record in (tampered, trida.record()):
        assert ClaudeWriter(client).attempt(record, 30.0)["status"] == "refused"
    assert client.bodies == []


def test_the_server_retries_a_busy_api_once_and_logs_what_it_said(
    requests: list[TweakRequest],  # noqa: F811
    tmp_path: Path,
) -> None:
    client = FakeClient(answered(anthropic.RateLimitError, 429), message("버스가 낮게 공회전한다."))
    service = service_with(ClaudeWriter(client).attempt, tmp_path)
    outcomes: list[dict[str, Any]] = []
    service.submit_tweak(requests[0], lambda r, o: outcomes.append(o))
    run(service)
    assert [o["status"] for o in outcomes] == ["retry", "ready"]
    assert outcomes[1]["served_by"] == "claude-sonnet-5-5" and outcomes[0]["served_by"] is None
    assert client.bodies[0] == client.bodies[1] == request_body(requests[0].job)
    lines = [json.loads(line) for line in (tmp_path / "logs" / "live-s1.jsonl").read_text().splitlines()]
    assert lines[1]["api"]["request_sha256"] == sha256_of(request_body(requests[0].job))
    assert lines[1]["token_ids_sha256"] is None  # no tokenizer here: the request's hash stands in
