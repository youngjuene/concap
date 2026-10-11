"""Live phase-2 captions: Dream rewrites the cue captions while a participant watches.

Researcher's design of 2026-10-11 (plan "Live phase 2"):

* The participant moves two sliders at any time. A move is snapped to steps
  (texture 0–4, context broad/specific/relation) and becomes the session's
  current *spec*, together with the participant's guidance sentence.
* Generation takes about ``L`` seconds, and a caption may only change at a cue
  boundary, so a tweak at playback position ``p`` is written first for the
  first cue that starts at or after ``p + L``: the earliest boundary at which
  it can be ready. The delay the participant sees is therefore between ``L``
  and ``L`` plus one cue. ``L`` follows the device's load: the 90th percentile
  of the last 20 live latencies, never below 6 s, starting at 7 s.
* While the worker is idle it keeps filling later cues under the current spec,
  up to three cues ahead. A cue that would end before its caption could be
  ready is skipped.
* At a cue boundary with nothing new ready, the previous caption stays (the
  browser does this). A finished caption for an older spec is kept in the
  cache, never shown for the newer one, and served again if the participant
  returns to that spec. A Dream call cannot be cancelled without killing the
  worker.
* Before playback, the first cues are written at the calibrated settings, so a
  video opens with Dream's rewrite.
* A caption is shown as Dream wrote it. A call that fails is logged and not
  asked again, except for one identical retry after a transient worker or
  device failure (the brief's rule).

The scheduler is pure (time is passed in), so it is tested without threads or
a model. :class:`LiveService` adds the worker thread, the supervised Dream
worker (:mod:`diff.generation_worker`, ``kind="live"``) and the attempt log.
Nothing here imports torch: the worker renders, tokenizes and runs the job.
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from diff.caption_sources import CaptionSource
from diff.generation_worker import TRANSIENT
from diff.job_plan import Job
from diff.study_protocol import CONTEXT_STEPS

LIVE_ATTEMPT_SCHEMA: Final = "diff.live-attempt/v1"
LATENCY_WINDOW: Final = 20
LATENCY_QUANTILE: Final = 0.9
LATENCY_FLOOR_MS: Final = 6000
LATENCY_START_MS: Final = 7000
LOOKAHEAD_CUES: Final = 3
LIVE_DEADLINE_S: Final = 30.0
TEXTURE_STEPS: Final = 4


def slider_steps(texture: float, context: float) -> tuple[int, str]:
    """Continuous slider values (0–1) as the prompt's steps: texture 0–4, context in thirds."""
    for name, value in (("texture", texture), ("context", context)):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{name} must be a number from 0 to 1, got {value!r}")
    return round(texture * TEXTURE_STEPS), CONTEXT_STEPS[min(len(CONTEXT_STEPS) - 1, int(context * 3))]


@dataclass(frozen=True)
class Spec:
    """What a caption is written under: the snapped steps and the participant's guidance."""

    texture: int
    context: str
    guidance: str = ""


class LatencyEstimator:
    """``L``: the 90th percentile of recent completed latencies, floored, with a starting value."""

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

    def position(self, now: float) -> int:
        if not self.playing:
            return self.position_ms
        return min(self.stream.duration_ms, self.position_ms + round((now - self.reported_at) * 1000))

    def set_spec(self, spec: Spec, position_ms: int, now: float) -> bool:
        """A snapped settings change; ``False`` when the steps and guidance did not change."""
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

        Playing: start at the first boundary one latency from now. Paused or not
        yet started: start at the next boundary from the current position (cue 0
        before playback), which is where the caption can next change.
        """
        if self.spec is None:
            return None
        estimate = self.estimator.estimate_ms()
        position = self.position(now)
        target = position + estimate if self.playing else position
        first = self.stream.first_cue_from(target)
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
            if result.get("latency_ms") is not None:
                self.estimator.observe(int(result["latency_ms"]))
        else:
            self.failures.setdefault(request.job.job_id, []).append(str(result.get("status")))

    def ready(self) -> dict[int, dict[str, Any]]:
        """Captions written under the current spec, by cue; failures are shown as written."""
        if self.spec is None:
            return {}
        ready: dict[int, dict[str, Any]] = {}
        for cue in range(len(self.stream.cues)):
            job = self.job(cue)
            result = self.results.get(job.job_id)
            if result is not None:
                ready[cue] = {"job_id": job.job_id, "text": str((result.get("output") or {}).get("text", ""))}
        return ready


def opaque_token(secret: bytes, session_id: str, internal_id: str) -> str:
    """What the browser sees instead of a job ID (plan §6.5)."""
    return hmac.new(secret, f"{session_id}\0{internal_id}".encode(), hashlib.sha256).hexdigest()[:20]


Attempt = Callable[[Mapping[str, Any], float], dict[str, Any]]


class LiveService:
    """One Dream worker (one GPU) serving one session at a time, on its own thread.

    ``attempt`` is ``Supervisor.attempt`` bound to ``kind="live"`` in production
    and a fake in tests. ``check`` turns a result into the check record logged
    beside it.
    """

    def __init__(
        self,
        attempt: Attempt,
        log_dir: Path,
        *,
        check: Callable[[CaptionSource, Job, Mapping[str, Any]], Mapping[str, Any]] | None = None,
        deadline_s: float = LIVE_DEADLINE_S,
        clock: Callable[[], float] = time.monotonic,
        idle_wait_s: float = 0.25,
    ) -> None:
        self.attempt, self.log_dir, self.check = attempt, log_dir, check
        self.deadline_s, self.clock, self.idle_wait_s = deadline_s, clock, idle_wait_s
        self.lock = threading.Condition()
        self.session_id: str | None = None
        self.scheduler: LiveScheduler | None = None
        self.stopping = False
        self.thread = threading.Thread(target=self._run, daemon=True)

    # -- session slot ---------------------------------------------------------
    def attach(self, session_id: str, scheduler: LiveScheduler) -> None:
        with self.lock:
            if self.session_id not in (None, session_id):
                raise RuntimeError("this live service already serves a session")
            self.session_id, self.scheduler = session_id, scheduler
            self.lock.notify_all()

    def detach(self, session_id: str) -> None:
        with self.lock:
            if self.session_id == session_id:
                self.session_id, self.scheduler = None, None

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

    # -- the worker thread -----------------------------------------------------
    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        with self.lock:
            self.stopping = True
            self.lock.notify_all()
        self.thread.join(timeout=5)

    def step(self) -> bool:
        """Run at most one request; ``True`` if one ran. The thread calls this in a loop."""
        with self.lock:
            scheduler, session_id = self.scheduler, self.session_id
            request = scheduler.next_request(self.clock()) if scheduler is not None else None
        if request is None or scheduler is None or session_id is None:
            return False
        result = self.attempt(request.job.record(), self.deadline_s)
        with self.lock:
            scheduler.complete(request, result)
            self._log(session_id, scheduler, request, result)
        return True

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
        source = scheduler.stream.cues[request.cue]
        record = {
            "schema": LIVE_ATTEMPT_SCHEMA,
            "session_id": session_id,
            "video_id": scheduler.stream.video_id,
            "cue": request.cue,
            "job": request.job.record(),
            "spec": {"texture": request.spec.texture, "context": request.spec.context},
            "guidance": request.spec.guidance,
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
            "checks": dict(self.check(source, request.job, result))
            if self.check and result.get("output")
            else None,
        }
        self.log_dir.mkdir(parents=True, exist_ok=True)
        with (self.log_dir / f"live-{session_id}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
