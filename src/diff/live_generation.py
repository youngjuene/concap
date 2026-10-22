"""Live phase-2 captions: Trida rewrites the cue captions while a participant watches.

Researcher's design of 2026-10-11 (plan "Live phase 2"):

* The participant moves two sliders at any time. A move is snapped to steps
  (texture 0–2, context broad/specific/relation) and becomes the session's
  current *spec*, together with the session's tuner system message.
* A change shows as soon as it is written (researcher, 2026-10-21; until then
  a caption changed only at a cue boundary). Generation takes about ``L``
  seconds, so a tweak at playback position ``p`` is written first for the cue
  playing at ``p + L``, and the browser swaps it in mid-cue when it arrives:
  about a second after the move. Every caption stays on screen at least 3 s
  before a swap replaces it, and a swap needs 3 s of its cue left (the
  browser's ``MIN_READ_MS``), so quick slider moves never make the caption
  flicker. When that cue would end within ``MIN_SHOWN_MS`` of ``p + L``, the
  tweak goes to the next cue instead. ``L`` follows the device's load (researcher,
  2026-10-12, no floor): the 90th percentile of the last 20 live latencies,
  starting at 1 s.
* While the worker is idle it keeps filling later cues under the current spec,
  up to three cues ahead. A cue that would end before its caption could be
  ready is skipped.
* Every window shows a caption about its own audio (researcher, 2026-10-20).
  At a cue boundary the browser shows the cue's caption as the scheduler
  serves it, the first of: Trida's caption under the current spec; Trida's
  latest usable caption for this cue under earlier settings (after a move too
  late for the next boundary, or while the worker is busy, failing or
  restarting); the cue's starting caption. Never the previous cue's caption.
  A finished caption for an older spec stays in the cache and is current
  again if the participant returns to that spec. A model call cannot be
  cancelled without killing the worker.
* Before playback, the first cues are written at the calibrated settings, so a
  video opens with Trida's rewrite.
* A caption is shown as Trida wrote it, under the display rule
  (``output_checks.DISPLAY_RULE``, researcher 2026-10-20): a leading prompt
  label and one wrapping pair of quotes are removed, and inside, a run of
  commas becomes one (version 2, 2026-10-22). The attempt log keeps Trida's
  text and the shown text.
* A caption is usable (:func:`usable`) when something readable is left of it
  under the display rule and its check found no failure and no repetition.
  Trida writing only "자막:", a cut-off caption or a word loop is therefore
  never shown; the next choice above is. A call that fails is
  logged and not asked again, except for one identical retry after a transient worker or
  device failure (the brief's rule).

The scheduler is pure (time is passed in), so it is tested without threads or
a model. :class:`LiveService` adds the worker thread, the supervised Trida
worker (:mod:`diff.generation_worker`, ``kind="live"``) and the attempt log.
Nothing here imports torch: the worker renders, tokenizes and runs the job.

The same worker also writes phase 1's second captions (researcher, 2026-10-18):
:meth:`LiveService.submit_tweak` queues one per clip while the participant is
still in phase 1, and the thread runs them whenever no video scheduler has a
request, in the order submitted. Unlike a live caption, a tweak is checked
before it counts: an output that fails its check, or a call that fails, ends
that clip's tweak with the reason, except for one identical retry after a
transient worker or device failure.
"""

from __future__ import annotations

import bisect
import hashlib
import hmac
import json
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Final

from diff.caption_sources import CaptionSource
from diff.generation_worker import TRANSIENT, WorkerStartupError
from diff.job_plan import Job
from diff.output_checks import DISPLAY_RULE, readable, shown_text
from diff.study_protocol import CONTEXT_STEPS, LIVE_TEXTURE_MAX

# v2 (2026-10-20): each attempt also records the shown text and the display rule.
# v3 (2026-10-20): and where the shown text came from, Trida or the cue's starting caption.
# v4 (2026-10-20): ``usable`` instead; which caption each window showed is in the exposure log.
LIVE_ATTEMPT_SCHEMA: Final = "diff.live-attempt/v4"
STARTING_CAPTION: Final = "starting-caption"
TWEAK_ATTEMPT_SCHEMA: Final = "diff.tweak-attempt/v2"
LATENCY_WINDOW: Final = 20
LATENCY_QUANTILE: Final = 0.9
LATENCY_FLOOR_MS: Final = 0
LATENCY_START_MS: Final = 1000
LOOKAHEAD_CUES: Final = 3
# A tweak for the cue playing needs this long left at p + L: the page's 3 s reading minimum + its 0.5 s poll.
MIN_SHOWN_MS: Final = 3500
LIVE_DEADLINE_S: Final = 30.0
TEXTURE_STEPS: Final = LIVE_TEXTURE_MAX


def usable(result: Mapping[str, Any]) -> bool:
    """Whether a completed live caption may be shown (researcher, 2026-10-20).

    Something readable is left under the display rule, and its check found no failure and no
    repetition.
    """
    written = str((result.get("output") or {}).get("text", ""))
    checks = result.get("checks") or {}
    repeated = any(finding.get("code") == "repetition" for finding in checks.get("flags", []))
    return readable(shown_text(written)) and not checks.get("failures") and not repeated


def slider_steps(texture: float, context: float) -> tuple[int, str]:
    """Continuous slider values (0–1) as the prompt's steps: texture 0–2, context in thirds."""
    for name, value in (("texture", texture), ("context", context)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a number from 0 to 1, got {value!r}")
    return round(texture * TEXTURE_STEPS), CONTEXT_STEPS[min(len(CONTEXT_STEPS) - 1, int(context * 3))]


@dataclass(frozen=True)
class Spec:
    """What a caption is written under: the snapped steps and the session's tuner system message."""

    texture: int
    context: str
    system_message: str = ""


class LatencyEstimator:
    """``L``: the 90th percentile of recent completed latencies, with a starting value (and any floor)."""

    def __init__(
        self,
        *,
        window: int = LATENCY_WINDOW,
        quantile: float = LATENCY_QUANTILE,
        floor_ms: int = LATENCY_FLOOR_MS,
        start_ms: int = LATENCY_START_MS,
    ) -> None:
        self.samples: deque[int] = deque(maxlen=window)
        self.quantile, self.floor_ms, self.start_ms = quantile, floor_ms, start_ms

    def observe(self, latency_ms: int) -> None:
        self.samples.append(int(latency_ms))

    def estimate_ms(self) -> int:
        if not self.samples:
            return max(self.floor_ms, self.start_ms)
        ordered = sorted(self.samples)
        index = min(len(ordered) - 1, max(0, round(self.quantile * len(ordered)) - 1))
        return max(self.floor_ms, ordered[index])


@dataclass(frozen=True)
class LiveStream:
    """One video's cues and their starting captions, in time order."""

    video_id: str
    cues: tuple[CaptionSource, ...]

    @classmethod
    def from_sources(cls, sources: Sequence[CaptionSource], video_id: str) -> LiveStream:
        cues = tuple(sorted((s for s in sources if s.unit_key == video_id), key=lambda s: s.cue_index))
        if not cues:
            raise ValueError(f"no cues for video {video_id}")
        if [cue.cue_index for cue in cues] != list(range(len(cues))):
            raise ValueError(f"{video_id}: cue indices must run 0..n-1")
        return cls(video_id, cues)

    @property
    def duration_ms(self) -> int:
        return self.cues[-1].end_ms

    def cue_at(self, position_ms: int) -> int:
        """The cue on screen at ``position_ms``; past the end, the last cue."""
        starts = [cue.start_ms for cue in self.cues]
        return max(0, min(len(self.cues) - 1, bisect.bisect_right(starts, position_ms) - 1))

    def first_cue_from(self, position_ms: int) -> int:
        """The first cue starting at or after ``position_ms``; ``len(cues)`` when none does."""
        return bisect.bisect_left([cue.start_ms for cue in self.cues], position_ms)


def starting_caption_problems(stream: LiveStream, max_chars: int) -> list[str]:
    """Starting captions a live rewrite on the protocol's canvas cannot hold."""
    return [
        f"{stream.video_id} cue {cue.cue_index}: {len(cue.text)} characters, over {max_chars}"
        for cue in stream.cues
        if len(cue.text) > max_chars
    ]


@dataclass(frozen=True)
class LiveRequest:
    cue: int
    job: Job
    spec: Spec
    revision: int
    latency_estimate_ms: int
    position_ms: int
    target_ms: int


@dataclass
class LiveScheduler:
    """One session's live state for one video; every method takes the current time."""

    stream: LiveStream
    job_for: Callable[[CaptionSource, Spec], Job]
    estimator: LatencyEstimator = field(default_factory=LatencyEstimator)
    lookahead: int = LOOKAHEAD_CUES
    spec: Spec | None = None
    revision: int = 0
    position_ms: int = 0
    playing: bool = False
    reported_at: float = 0.0
    results: dict[str, dict[str, Any]] = field(default_factory=dict)
    failures: dict[str, list[str]] = field(default_factory=dict)
    in_flight: set[str] = field(default_factory=set)
    written: dict[int, list[str]] = field(default_factory=dict)  # completed job IDs per cue, in order

    def position(self, now: float) -> int:
        if not self.playing:
            return self.position_ms
        return min(self.stream.duration_ms, self.position_ms + round((now - self.reported_at) * 1000))

    def set_spec(self, spec: Spec, position_ms: int, now: float) -> bool:
        """A snapped settings change; ``False`` when the steps and system message did not change."""
        self.position_ms, self.reported_at = int(position_ms), now
        if spec == self.spec:
            return False
        self.spec, self.revision = spec, self.revision + 1
        return True

    def set_playback(self, position_ms: int, playing: bool, now: float) -> None:
        self.position_ms, self.playing, self.reported_at = int(position_ms), bool(playing), now

    def job(self, cue: int) -> Job:
        assert self.spec is not None
        return self.job_for(self.stream.cues[cue], self.spec)

    def next_request(self, now: float) -> LiveRequest | None:
        """The next cue worth writing under the current spec, or ``None`` when there is none.

        Playing: start at the cue that will be playing one latency from now, the
        browser swapping it in mid-cue, unless that cue would end within
        ``MIN_SHOWN_MS``; then at the next cue. Paused or not yet started: start
        at the cue at the current position (cue 0 before playback).
        """
        if self.spec is None:
            return None
        estimate = self.estimator.estimate_ms()
        position = self.position(now)
        target = position + estimate if self.playing else position
        if target >= self.stream.duration_ms:
            return None
        first = self.stream.cue_at(target)
        if self.playing and self.stream.cues[first].end_ms - target < MIN_SHOWN_MS:
            first += 1
        for cue in range(first, min(first + self.lookahead, len(self.stream.cues))):
            job = self.job(cue)
            if job.job_id in self.results or job.job_id in self.in_flight or not self.may_try(job.job_id):
                continue
            self.in_flight.add(job.job_id)
            return LiveRequest(cue, job, self.spec, self.revision, estimate, position, target)
        return None

    def may_try(self, job_id: str) -> bool:
        """No attempt yet, or exactly one earlier attempt that failed transiently."""
        failed = self.failures.get(job_id, [])
        return not failed or (len(failed) == 1 and failed[0] in TRANSIENT)

    def complete(self, request: LiveRequest, result: Mapping[str, Any]) -> None:
        self.in_flight.discard(request.job.job_id)
        if result.get("status") == "completed":
            self.results[request.job.job_id] = dict(result)
            self.written.setdefault(request.cue, []).append(request.job.job_id)
            if result.get("latency_ms") is not None:
                self.estimator.observe(int(result["latency_ms"]))
        else:
            self.failures.setdefault(request.job.job_id, []).append(str(result.get("status")))

    def ready(self) -> dict[int, dict[str, Any]]:
        """Every cue's caption as the browser would show it now, each about its own window.

        The first of: Trida's usable caption under the current spec (``current``); its latest
        usable caption for this cue under earlier settings; the cue's starting caption, whose
        ``job_id`` is ``starting-caption:<video>:<cue>``. ``settled``: the current spec's attempt
        for this cue is over (written, unusable or failed for good), so nothing better is coming.
        """
        if self.spec is None:
            return {}
        ready: dict[int, dict[str, Any]] = {}
        for cue, source in enumerate(self.stream.cues):
            job_id = self.job(cue).job_id
            settled = job_id in self.results or not self.may_try(job_id)
            choices = [job_id, *reversed(self.written.get(cue, []))]
            shown = next((j for j in choices if j in self.results and usable(self.results[j])), None)
            if shown is None:
                ready[cue] = {
                    "job_id": f"{STARTING_CAPTION}:{self.stream.video_id}:{cue}",
                    "text": source.text,
                    "current": False,
                    "settled": settled,
                }
            else:
                text = shown_text(str(self.results[shown]["output"]["text"]))
                ready[cue] = {"job_id": shown, "text": text, "current": shown == job_id, "settled": settled}
        return ready


def opaque_token(secret: bytes, session_id: str, internal_id: str) -> str:
    """What the browser sees instead of a job ID (plan §6.5)."""
    return hmac.new(secret, f"{session_id}\0{internal_id}".encode(), hashlib.sha256).hexdigest()[:20]


Attempt = Callable[[Mapping[str, Any], float], dict[str, Any]]
Check = Callable[[CaptionSource, Job, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class TweakRequest:
    """One clip's second caption for one session; ``attempt`` counts from 1."""

    session_id: str
    clip: str
    source: CaptionSource
    job: Job
    steer: Mapping[str, Any]
    attempt: int = 1


TweakDone = Callable[[TweakRequest, dict[str, Any]], None]


def tweak_outcome(
    request: TweakRequest, result: Mapping[str, Any], checks: Mapping[str, Any] | None
) -> dict[str, Any]:
    """What one attempt means for the clip: ``ready``, ``retry`` or ``failed`` with the reason."""
    status = result.get("status")
    base = {"attempt": request.attempt, "result_status": status, "job_id": request.job.job_id}
    if status == "completed" and checks is not None and not checks.get("excluded"):
        text = shown_text(result["output"]["text"])
        if not readable(text):  # a second caption never goes blank: the clip's second step is skipped
            return {
                **base,
                "status": "failed",
                "reason": "nothing readable after the display rule",
                "checks": dict(checks),
            }
        return {**base, "status": "ready", "text": text, "checks": dict(checks)}
    if status == "completed":
        codes = [finding["code"] for finding in (checks or {}).get("failures", [])]
        reason = "check failed: " + ", ".join(codes) if codes else "no check result"
        return {**base, "status": "failed", "reason": reason, "checks": dict(checks or {})}
    if status in TRANSIENT and request.attempt == 1:
        return {**base, "status": "retry", "reason": f"{status}: {result.get('error')}"}
    return {**base, "status": "failed", "reason": f"{status}: {result.get('error')}"}


class LiveService:
    """One Trida worker (one GPU) serving one session at a time, on its own thread.

    ``attempt`` is ``Supervisor.attempt`` bound to ``kind="live"`` in production
    and a fake in tests. ``check`` turns a result into the check record logged
    beside it.
    """

    def __init__(
        self,
        attempt: Attempt,
        log_dir: Path,
        *,
        check: Check | None = None,
        tweak_check: Check | None = None,
        deadline_s: float = LIVE_DEADLINE_S,
        clock: Callable[[], float] = time.monotonic,
        idle_wait_s: float = 0.25,
    ) -> None:
        self.attempt, self.log_dir, self.check, self.tweak_check = attempt, log_dir, check, tweak_check
        self.deadline_s, self.clock, self.idle_wait_s = deadline_s, clock, idle_wait_s
        self.lock = threading.Condition()
        self.session_id: str | None = None
        self.scheduler: LiveScheduler | None = None
        self.tweaks: deque[tuple[TweakRequest, TweakDone]] = deque()
        self.running_tweak: str | None = None
        self.stopping = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    # -- session slot ---------------------------------------------------------
    def attach(self, session_id: str, scheduler: LiveScheduler) -> None:
        with self.lock:
            if self.session_id not in (None, session_id):
                raise RuntimeError("this live service already serves a session")
            self.session_id, self.scheduler = session_id, scheduler
            self.lock.notify_all()

    def claim(self, session_id: str) -> bool:
        """Hold this GPU for a session from its entry on; ``False`` if another session holds it."""
        with self.lock:
            if self.session_id not in (None, session_id):
                return False
            self.session_id = session_id
            return True

    def stop(self, session_id: str) -> None:
        """End the current video's scheduler but keep the session's claim (between videos)."""
        with self.lock:
            if self.session_id == session_id:
                self.scheduler = None

    def detach(self, session_id: str) -> None:
        with self.lock:
            if self.session_id == session_id:
                self.session_id, self.scheduler = None, None
                self.tweaks = deque(item for item in self.tweaks if item[0].session_id != session_id)

    @property
    def free(self) -> bool:
        return self.session_id is None

    # -- calls from the participant server -------------------------------------
    def settings(self, spec: Spec, position_ms: int) -> None:
        with self.lock:
            if self.scheduler is not None and self.scheduler.set_spec(spec, position_ms, self.clock()):
                self.lock.notify_all()

    def playback(self, position_ms: int, playing: bool) -> None:
        with self.lock:
            if self.scheduler is not None:
                self.scheduler.set_playback(position_ms, playing, self.clock())
                self.lock.notify_all()

    def ready(self) -> dict[int, dict[str, Any]]:
        with self.lock:
            return {} if self.scheduler is None else self.scheduler.ready()

    def submit_tweak(self, request: TweakRequest, done: TweakDone) -> bool:
        """Queue a clip's second caption for the claimed session; ``False`` if not claimed or queued."""
        with self.lock:
            if request.session_id != self.session_id:
                return False
            if request.job.job_id == self.running_tweak or any(
                queued.job.job_id == request.job.job_id for queued, _ in self.tweaks
            ):
                return False
            self.tweaks.append((request, done))
            self.lock.notify_all()
            return True

    # -- the worker thread -----------------------------------------------------
    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        with self.lock:
            self.stopping = True
            self.lock.notify_all()
        self.thread.join(timeout=5)

    def step(self) -> bool:
        """Run at most one request; ``True`` if one ran. The thread calls this in a loop.

        A video's requests come first; a queued tweak runs when the video has none.
        """
        with self.lock:
            scheduler, session_id = self.scheduler, self.session_id
            request = scheduler.next_request(self.clock()) if scheduler is not None else None
            tweak = None
            if request is None and self.tweaks and self.tweaks[0][0].session_id == session_id:
                tweak = self.tweaks.popleft()
                self.running_tweak = tweak[0].job.job_id
        if tweak is not None:
            self._tweak(*tweak)
            return True
        if request is None or scheduler is None or session_id is None:
            return False
        result = self._call(request.job)
        if self.check and result.get("output"):  # the scheduler shows a caption only if it is usable
            source = scheduler.stream.cues[request.cue]
            result = {**result, "checks": dict(self.check(source, request.job, result))}
        with self.lock:
            scheduler.complete(request, result)
            self._log(session_id, scheduler, request, result)
        return True

    def _call(self, job: Job) -> dict[str, Any]:
        """One model call; a worker that cannot start counts as a lost worker, not a dead thread."""
        try:
            return self.attempt(job.record(), self.deadline_s)
        except WorkerStartupError as exc:
            return {"status": "worker_lost", "error": f"worker did not start: {exc}"}

    def _tweak(self, request: TweakRequest, done: TweakDone) -> None:
        result = self._call(request.job)
        checks = None
        if result.get("status") == "completed" and result.get("output") and self.tweak_check is not None:
            checks = dict(self.tweak_check(request.source, request.job, result))
        outcome = tweak_outcome(request, result, checks)
        self._log_tweak(request, result, checks, outcome)
        try:
            done(request, outcome)
        finally:
            with self.lock:
                self.running_tweak = None
                if outcome["status"] == "retry" and self.session_id == request.session_id:
                    self.tweaks.appendleft((replace(request, attempt=request.attempt + 1), done))
                    self.lock.notify_all()

    def _log_tweak(
        self,
        request: TweakRequest,
        result: Mapping[str, Any],
        checks: Mapping[str, Any] | None,
        outcome: Mapping[str, Any],
    ) -> None:
        record = {
            "schema": TWEAK_ATTEMPT_SCHEMA,
            "kind": "tweak",
            "session_id": request.session_id,
            "clip": request.clip,
            "attempt": request.attempt,
            "job": request.job.record(),
            "system_message": request.job.prompt.system_message,
            "steer": dict(request.steer),
            "status": result.get("status"),
            "error": result.get("error"),
            "token_ids_sha256": result.get("token_ids_sha256"),
            "latency_ms": result.get("latency_ms"),
            "wall_ms": result.get("wall_ms"),
            "output": result.get("output"),
            "shown_text": shown_text(result["output"]["text"]) if result.get("output") else None,
            "display_rule": DISPLAY_RULE,
            "checks": checks,
            "outcome": outcome["status"],
            "reason": outcome.get("reason"),
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / f"live-{request.session_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _run(self) -> None:
        while True:
            with self.lock:
                if self.stopping:
                    return
            if not self.step():
                with self.lock:
                    if not self.stopping:
                        self.lock.wait(self.idle_wait_s)

    def _log(
        self, session_id: str, scheduler: LiveScheduler, request: LiveRequest, result: Mapping[str, Any]
    ) -> None:
        record = {
            "schema": LIVE_ATTEMPT_SCHEMA,
            "session_id": session_id,
            "video_id": scheduler.stream.video_id,
            "cue": request.cue,
            "job": request.job.record(),
            "spec": {"texture": request.spec.texture, "context": request.spec.context},
            "system_message": request.spec.system_message,
            "settings_revision": request.revision,
            "current_revision_at_finish": scheduler.revision,
            "latency_estimate_ms": request.latency_estimate_ms,
            "position_ms": request.position_ms,
            "target_ms": request.target_ms,
            "status": result.get("status"),
            "error": result.get("error"),
            "seed": result.get("seed"),
            "token_ids_sha256": result.get("token_ids_sha256"),
            "latency_ms": result.get("latency_ms"),
            "wall_ms": result.get("wall_ms"),
            "output": result.get("output"),
            "shown_text": shown_text(result["output"]["text"]) if result.get("output") else None,
            "usable": usable(result) if result.get("output") else False,
            "display_rule": DISPLAY_RULE,
            "checks": result.get("checks"),
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / f"live-{session_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
