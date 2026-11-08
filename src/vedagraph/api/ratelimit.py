"""Limits on ``POST /api/v1/ask``, the one route that spends money per request.

Two independent limits, checked in this order because they protect different things:

* **Per client IP, sliding window.** Stops one caller from draining the budget. Checked
  first so a flood from one address is turned away without touching the shared cap.
* **Global daily cap on LLM calls.** The hard ceiling on spend, however many callers there
  are. Counted per UTC day and charged only when a request is about to reach the provider,
  so a request that fails validation or finds the backend unconfigured costs nothing.

State is in process memory. That is correct for one worker and an undercount for N: each
worker enforces its own limit. A deployment that scales out needs a shared store, and this
class is the one place to put it.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from fastapi import Request, status

from vedagraph.api.errors import ApiError

#: Above this many tracked addresses, a call sweeps out the ones whose window has emptied.
#: Without it a scan from many addresses would grow the table for the life of the process.
_SWEEP_THRESHOLD = 10_000


class RateLimitedError(ApiError):
    """A limit was hit. ``Retry-After`` says when trying again could succeed."""

    status_code = status.HTTP_429_TOO_MANY_REQUESTS
    code = "RATE_LIMITED"

    def __init__(self, detail: str, *, retry_after: int, hint: str | None = None) -> None:
        super().__init__(detail, hint=hint, headers={"Retry-After": str(retry_after)})


class DailyCapReachedError(RateLimitedError):
    code = "DAILY_CAP_REACHED"


def client_ip(request: Request, trusted_proxies: int = 1) -> str:
    """The caller's address, honouring ``X-Forwarded-For`` from ``trusted_proxies`` hops.

    Each trusted proxy appends the address it saw, so the entry ``trusted_proxies`` from the
    right is the first one our own infrastructure wrote. Everything left of it is
    client-controlled: taking the first entry would let a caller pick its own bucket on every
    request. With fewer entries than hops the header cannot be trusted to have passed through
    our proxies, so the socket address is used: a mismatched setting then shares a bucket
    rather than letting a caller forge its own.

    The right count is a property of the deployment, not of this code. Behind one proxy it is
    1. Browser -> Next.js rewrite -> a host's load balancer -> API is at least 2, and if 1 were
    left set there every visitor would share the frontend's address, and with it one bucket.
    """
    forwarded = request.headers.get("x-forwarded-for", "")
    entries = [part.strip() for part in forwarded.split(",") if part.strip()]
    if len(entries) >= trusted_proxies:
        return entries[len(entries) - trusted_proxies]
    return request.client.host if request.client else "unknown"


class AskLimiter:
    def __init__(
        self,
        *,
        per_minute: int,
        daily_cap: int,
        trusted_proxies: int = 1,
        monotonic: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._per_minute = per_minute
        self._daily_cap = daily_cap
        self.trusted_proxies = trusted_proxies
        self._monotonic = monotonic
        self._now = now
        self._lock = threading.Lock()
        self._hits: dict[str, deque[float]] = {}
        self._day = self._now().date()
        self._used = 0

    def check_ip(self, ip: str) -> None:
        """Record a request from ``ip``, or raise if it already made its quota this minute."""
        with self._lock:
            now = self._monotonic()
            if len(self._hits) > _SWEEP_THRESHOLD:
                self._sweep(now)
            window = self._hits.setdefault(ip, deque())
            self._expire(window, now)
            if len(window) >= self._per_minute:
                retry_after = max(1, math.ceil(window[0] + 60 - now))
                raise RateLimitedError(
                    "Too many questions from this address.",
                    retry_after=retry_after,
                    hint=f"At most {self._per_minute} questions per minute are accepted.",
                )
            window.append(now)

    def charge_daily(self) -> None:
        """Reserve one LLM call against today's cap, or raise if none is left."""
        with self._lock:
            now = self._now()
            if now.date() != self._day:
                self._day = now.date()
                self._used = 0
            if self._used >= self._daily_cap:
                tomorrow = datetime.combine(
                    self._day + timedelta(days=1), datetime.min.time(), tzinfo=UTC
                )
                raise DailyCapReachedError(
                    "Ask VedaGraph has reached its daily limit.",
                    retry_after=max(1, math.ceil((tomorrow - now).total_seconds())),
                    hint="The limit resets at 00:00 UTC.",
                )
            self._used += 1

    @staticmethod
    def _expire(window: deque[float], now: float) -> None:
        while window and window[0] <= now - 60:
            window.popleft()

    def _sweep(self, now: float) -> None:
        for ip in list(self._hits):
            window = self._hits[ip]
            self._expire(window, now)
            if not window:
                del self._hits[ip]
