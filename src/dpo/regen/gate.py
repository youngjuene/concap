"""What stands between the internet and the instrument once it is published.

The instrument was built for a kiosk and has no login, because the researcher
is in the room. Published through a tunnel, three things the room used to
provide are missing, and this module provides them. None of them is on unless
the operator turns it on, so a kiosk run is byte-for-byte what it was.

*An access code.* A new enrolment is refused unless the request carries the
code held in a file the operator controls. The file is read on every
enrolment, so changing it closes the study to new participants at once, with
nothing restarted; an empty or missing file closes it too. A participant who
is already enrolled resumes without a code, because a reload must never
strand a session.

*The log stays on the study machine.* ``/api/log`` exports a participant's
whole session. On a kiosk that download is the researcher's tool; on the
internet it is anyone's, given a guessable identifier. So it answers only
requests that reach the server from this machine with no proxy in front — a
proxy announces itself with a forwarding header, and the tunnel is a proxy.

*A rate limit by address.* Nothing in front of a tunnel throttles, so a token
bucket per client address does, with a tighter one on enrolment: the one
request that spends a sequence number the study's alternation is balanced on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from starlette.requests import Request

LOOPBACK = frozenset({"127.0.0.1", "::1"})
FORWARDING_HEADERS = ("x-forwarded-for", "x-forwarded-host", "forwarded")


def read_code(path: Path) -> str | None:
    """The access code in ``path``, or None when the file is missing or empty."""
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def client_address(request: Request) -> str:
    """Who is asking: the first forwarded address if a proxy says so, else the peer."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    return request.client.host if request.client else "unknown"


def is_local(request: Request) -> bool:
    """True only for a request from this machine that no proxy carried."""
    if any(request.headers.get(name) for name in FORWARDING_HEADERS):
        return False
    client = request.client
    return client is not None and client.host in LOOPBACK


class Buckets:
    """Token buckets by key: ``rate`` tokens a second, holding at most ``burst``.

    ``clock`` is monotonic seconds; tests hand in their own.
    """

    def __init__(self, rate: float, burst: int, clock: Callable[[], float] = time.monotonic) -> None:
        if rate <= 0 or burst < 1:
            raise ValueError("a bucket needs a positive rate and a burst of at least one")
        self.rate = float(rate)
        self.burst = int(burst)
        self.clock = clock
        self._levels: dict[str, tuple[float, float]] = {}

    def allow(self, key: str) -> bool:
        now = self.clock()
        tokens, at = self._levels.get(key, (float(self.burst), now))
        tokens = min(float(self.burst), tokens + (now - at) * self.rate)
        allowed = tokens >= 1.0
        self._levels[key] = (tokens - 1.0 if allowed else tokens, now)
        if len(self._levels) > 10_000:
            # An address not seen for as long as a bucket takes to refill is a
            # full bucket again, and a full bucket is the default entry.
            horizon = now - self.burst / self.rate
            self._levels = {k: v for k, v in self._levels.items() if v[1] > horizon}
        return allowed


@dataclass(frozen=True)
class Gate:
    """The operator's choices. The default is a kiosk: nothing gated."""

    code_file: Path | None = None
    log_local_only: bool = False
    requests_per_second: float | None = None
    burst: int = 60
    enrolments_per_minute: float | None = None

    @classmethod
    def public(cls, code_file: Path | None) -> Gate:
        """What a published instrument runs with.

        Twenty requests a second with a burst of sixty is far above one
        session's page loads — a screen is a dozen requests, §4's strip five
        frames, §5's lanes a handful of stems — and far below what an
        enumeration or a flood needs. Five enrolments a minute from one
        address is a lab behind one NAT, not a script.
        """
        return cls(
            code_file=code_file,
            log_local_only=True,
            requests_per_second=20.0,
            burst=60,
            enrolments_per_minute=5.0,
        )

    @property
    def requires_code(self) -> bool:
        return self.code_file is not None

    def code(self) -> str | None:
        return read_code(self.code_file) if self.code_file is not None else None

    def record(self) -> dict[str, object]:
        """What the operator's status line says about the gate."""
        return {
            "access_code": "required" if self.requires_code else "none",
            "log_download": "this machine only" if self.log_local_only else "anywhere",
            "rate_limit": (
                f"{self.requests_per_second:g}/s, burst {self.burst}" if self.requests_per_second else "none"
            ),
        }
