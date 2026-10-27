"""Claude Sonnet 5.5, phase 1's second-caption writer since 2026-10-27, through the Claude API.

The researcher moved the writer from a local model to the Claude API after the 2026-10-24 to 26
benches (plan "Second captions: responsive, written live"). Trida cut verbs, looped and wrote lists of
names; Qwen2.5-7B and Qwen3.5 2B/4B/9B drifted into other scripts ("гу음", "中低音", Chinese
paragraphs), copied the prompt, and dropped sounds to stay short
(artifacts/diff/bench/candidates-20261026-live/MODELS.md).

The call is in-process: :class:`ClaudeWriter` is a ``LiveService`` attempt, like Trida's worker
behind ``Supervisor.attempt``, but needs no GPU. A job's ``generate`` settings are the request's
keywords apart from the prompt (``job_plan.writer_call_settings``), so the job's identity pins the
model, effort and output limit, and :func:`request_body` rebuilds the exact request from the job.
API version 2026-10-21 takes no sampling parameters, and thinking is left adaptive at low effort,
so the output is not reproducible call by call: every request and response is logged instead
(``api``), with the request's SHA-256 for the audit. A refused caption goes to the ladder's
next wording, as before.

Failures map onto the live statuses: a connection error, a timeout, 429 or 5xx (after the SDK's own
retry) is ``api_unavailable``, transient, so the server tries the call once more; any other API
error, a model refusal, or no credentials is ``refused``. An output cut at ``max_tokens`` has no
terminator, so the technical check counts it incomplete and the ladder moves on.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Final

from diff.job_plan import Job, PlanError, job_from_record
from diff.model_profile import CLAUDE_WRITER
from diff.output_checks import GeneratedOutput
from diff.study_protocol import sha256_of

# One SDK retry inside the call's deadline (each try gets half of it); the server adds one more.
SDK_RETRIES: Final = 1
UNAVAILABLE: Final = "api_unavailable"


def request_body(job: Job) -> dict[str, Any]:
    """What the writer sends for one job: its call settings, the fixed system message and the prompt."""
    return {
        **job.generate,
        "system": job.prompt.system_message,
        "messages": [{"role": "user", "content": job.prompt.text}],
    }


def output_record(message: Any) -> dict[str, Any]:
    """A response as the checks read an output: its text blocks joined, ended only at ``end_turn``."""
    text = "".join(block.text for block in message.content if block.type == "text")
    ended = message.stop_reason == "end_turn"
    return GeneratedOutput(
        raw_text=text,
        text=text.strip(),
        terminator="end_turn" if ended else None,
        terminator_index=None,
        tail_token_ids=(),
        tail_text="",
        leftover_mask_count=0,
        special_ids_in_text=(),
        new_token_count=int(message.usage.output_tokens),
    ).record()


def api_record(message: Any, body: Mapping[str, Any]) -> dict[str, Any]:
    """What the API says about one call, beside the output: who served it and what it cost."""
    import anthropic

    iterations = getattr(message.usage, "iterations", None) or []
    details = getattr(message, "stop_details", None)
    return {
        "sdk": anthropic.__version__,
        "request_sha256": sha256_of(dict(body)),
        "request_id": getattr(message, "_request_id", None),
        "message_id": message.id,
        "model": message.model,
        "fallback": any(getattr(entry, "type", None) == "fallback_message" for entry in iterations),
        "stop_reason": message.stop_reason,
        "refusal_category": getattr(details, "category", None),
        "input_tokens": int(message.usage.input_tokens),
        "output_tokens": int(message.usage.output_tokens),
    }


class ClaudeWriter:
    """The writer's client; ``attempt`` is a ``LiveService`` attempt (``client`` is a fake in tests)."""

    def __init__(self, client: Any = None) -> None:
        if client is None:
            import anthropic

            client = anthropic.Anthropic(max_retries=SDK_RETRIES)
        self.client = client

    def attempt(self, payload: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        """One second caption: the job the server sent, its request, and the result the checks read."""
        try:
            job = job_from_record(payload)
        except PlanError as exc:
            return {"status": "refused", "error": str(exc)}
        if job.runner != CLAUDE_WRITER.runner or job.profile_id != CLAUDE_WRITER.profile_id:
            return {"status": "refused", "error": f"{job.job_id}: not a job for {CLAUDE_WRITER.profile_id}"}
        body = request_body(job)
        started = time.monotonic()
        try:
            message = self._create(body, timeout)
        except Exception as exc:  # noqa: BLE001 — every failure becomes a recorded attempt status
            return {
                "status": failure_status(exc),
                "error": f"{type(exc).__name__}: {exc}",
                "latency_ms": round((time.monotonic() - started) * 1000),
                "api": {"request_sha256": sha256_of(body)},
            }
        latency_ms = round((time.monotonic() - started) * 1000)
        api = api_record(message, body)
        if message.stop_reason == "refusal":
            return {
                "status": "refused",
                "error": f"the model declined ({api['refusal_category']})",
                "latency_ms": latency_ms,
                "api": api,
            }
        return {
            "status": "completed",
            "error": None,
            "output": output_record(message),
            "latency_ms": latency_ms,
            "api": api,
        }

    def _create(self, body: Mapping[str, Any], timeout: float) -> Any:
        client = self.client.with_options(timeout=timeout / (SDK_RETRIES + 1))
        return client.beta.messages.create(**body)

    def warm_up(self) -> dict[str, Any]:
        """One short call before any participant: the credentials work and the model answers."""
        from diff.job_plan import writer_call_settings

        started = time.monotonic()
        message = self._create(
            {**writer_call_settings(), "messages": [{"role": "user", "content": "안녕하세요."}]}, 60.0
        )
        return {"wall_ms": round((time.monotonic() - started) * 1000), "model": message.model}


def failure_status(exc: BaseException) -> str:
    """The attempt status an API exception earns: transient when the service is busy or unreachable."""
    import anthropic

    if isinstance(exc, anthropic.APIConnectionError):  # timeouts included
        return UNAVAILABLE
    if isinstance(exc, anthropic.APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500):
        return UNAVAILABLE
    return "refused"
