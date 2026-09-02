"""Captions written off the request path, for one GPU shared by two kinds of work.

Two things want the writer besides the participant pressing Show caption. The
auditions — one caption per source, shown while a token is held — must exist
before a shot opens, and both instruments used to write all of them for every
shot before accepting the first request, which blocks start-up for minutes on
a real corpus. And the neighbours of the settings a participant just chose —
one mute per source, one step of balance either way, one step of grain either
way, a dozen or so — are the settings they are most likely to ask for next;
written while they read the current caption, the next press is instant.

Neither changes what a participant is shown. Identical settings return the
identical caption from the cache whichever path wrote it first, and the
specification's own rule is that a revisit costs nothing; nothing says the
cache may not be warm. What the log needs to know is that a cache hit no
longer means the participant revisited: revisits are read from their own
caption.request events, and the route reports ``prefetched`` for what it is.

One daemon thread and two queues. Urgent work — this shot's auditions, the
neighbours of the last request — goes ahead of the warm-up sweep over every
audition in document order. A foreground request always wins: the route
declares itself waiting before it takes the cache lock, and the worker checks
for waiters between generations and yields. A generation already on the GPU
finishes first; that is the one cost, bounded by one caption.
"""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Hashable, Iterable, Iterator
from contextlib import contextmanager

from dpo.caption.writer import CachedWriter, CaptionRequest, WriterError


class BackgroundWriter:
    def __init__(self, cached: CachedWriter, *, enabled: bool = True) -> None:
        self.cached = cached
        self.enabled = enabled
        self._urgent: deque[tuple[Hashable | None, CaptionRequest]] = deque()
        self._warm: deque[tuple[Hashable | None, CaptionRequest]] = deque()
        self._pending: dict[Hashable, int] = {}
        self._ready: dict[Hashable, threading.Event] = {}
        self._waiting = 0
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._closed = False
        self.failures: list[tuple[CaptionRequest, str]] = []
        self._wrote: set[str] = set()
        self._thread = threading.Thread(target=self._run, name="caption-background", daemon=True)
        if enabled:
            self._thread.start()

    # ---- what the apps ask for ---------------------------------------------

    def warm(self, requests: Iterable[CaptionRequest], *, key: Hashable) -> None:
        """Queue a group behind everything urgent; ``wait_for(key)`` blocks on it."""
        self._enqueue(self._warm, requests, key)

    def urgent(self, requests: Iterable[CaptionRequest], *, key: Hashable | None = None) -> None:
        """Queue ahead of the warm-up sweep: the current shot, the next likely press."""
        self._enqueue(self._urgent, requests, key)

    def wait_for(self, key: Hashable, timeout: float | None = None) -> bool:
        """Block until every request queued under ``key`` has been written."""
        with self._lock:
            event = self._ready.get(key)
            if event is None:
                return True
        return event.wait(timeout)

    @contextmanager
    def foreground(self) -> Iterator[None]:
        """Wrap a participant's request so the worker yields the GPU to it."""
        with self._lock:
            self._waiting += 1
        try:
            yield
        finally:
            with self._lock:
                self._waiting -= 1
                self._wake.notify_all()

    def wrote(self, request: CaptionRequest) -> bool:
        """Whether this writer, not a participant's request, first wrote the caption."""
        with self._lock:
            return self.cached.key_for(request) in self._wrote

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._wake.notify_all()

    # ---- the worker --------------------------------------------------------

    def _enqueue(
        self,
        queue: deque[tuple[Hashable | None, CaptionRequest]],
        requests: Iterable[CaptionRequest],
        key: Hashable | None,
    ) -> None:
        items = [(key, request) for request in requests if self.cached.lookup(request) is None]
        if not self.enabled:
            # Synchronous: the template writer is instant, and a test wants an
            # answer it can assert on rather than a thread it must join.
            for _, request in items:
                self._write(request)
            return
        with self._lock:
            if key is not None:
                if items:
                    self._pending[key] = self._pending.get(key, 0) + len(items)
                    self._ready.setdefault(key, threading.Event())
                elif key not in self._ready:
                    done = threading.Event()
                    done.set()
                    self._ready[key] = done
            queue.extend(items)
            self._wake.notify_all()

    def _next(self) -> tuple[Hashable | None, CaptionRequest] | None:
        with self._lock:
            while not self._closed:
                if self._waiting == 0 and (self._urgent or self._warm):
                    return self._urgent.popleft() if self._urgent else self._warm.popleft()
                self._wake.wait()
            return None

    def _run(self) -> None:
        while True:
            item = self._next()
            if item is None:
                return
            key, request = item
            self._write(request)
            if key is not None:
                with self._lock:
                    self._pending[key] -= 1
                    if self._pending[key] <= 0:
                        self._ready[key].set()

    def _write(self, request: CaptionRequest) -> None:
        try:
            _, hit = self.cached.write_attributed_cached(request)
            if not hit:
                with self._lock:
                    self._wrote.add(self.cached.key_for(request))
        except WriterError as exc:
            # A background failure must not take the server down; the route
            # will hit the same failure on its own request and report it.
            self.failures.append((request, str(exc)))


__all__ = ["BackgroundWriter"]
