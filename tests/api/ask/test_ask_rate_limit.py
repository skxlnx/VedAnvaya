"""The two limits on POST /api/v1/ask: per client IP, and a global daily cap on LLM calls.

Each test replaces the app's limiter with one driven by a fake clock, so no test sleeps and
none depends on the wall-clock minute or day it happens to run in.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request

from tests.api.ask.test_ask_route import StubProvider
from vedagraph.api import ratelimit
from vedagraph.api.config import ApiSettings
from vedagraph.api.ratelimit import AskLimiter, client_ip
from vedagraph.api.routes import ask as ask_route

QUESTION = {"question": "Who is Indra?"}


class FakeClock:
    def __init__(self) -> None:
        self.seconds = 1000.0
        self.wall = datetime(2026, 11, 7, 12, 0, 0, tzinfo=UTC)

    def advance(self, seconds: float) -> None:
        self.seconds += seconds
        self.wall += timedelta(seconds=seconds)

    def limiter(
        self, *, per_minute: int = 10, daily_cap: int = 1000, trusted_proxies: int = 1
    ) -> AskLimiter:
        return AskLimiter(
            per_minute=per_minute,
            daily_cap=daily_cap,
            trusted_proxies=trusted_proxies,
            monotonic=lambda: self.seconds,
            now=lambda: self.wall,
        )


@pytest.fixture(autouse=True)
def stub_provider(monkeypatch: pytest.MonkeyPatch) -> StubProvider:
    stub = StubProvider()
    monkeypatch.setattr(ask_route, "get_llm_provider", lambda: stub)
    return stub


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def install(client: TestClient, limiter: AskLimiter) -> None:
    app = client.app
    assert isinstance(app, FastAPI)
    app.state.ask_limiter = limiter


def ask(client: TestClient, ip: str | None = None) -> object:
    headers = {"X-Forwarded-For": ip} if ip else {}
    return client.post("/api/v1/ask", json=QUESTION, headers=headers)


# ---------------------------------------------------------------------------
# Per client IP
# ---------------------------------------------------------------------------


def test_the_eleventh_request_in_a_minute_is_429_in_the_api_error_format(
    client: TestClient, clock: FakeClock
) -> None:
    install(client, clock.limiter())

    statuses = [ask(client, "203.0.113.7").status_code for _ in range(10)]  # type: ignore[attr-defined]
    response = client.post("/api/v1/ask", json=QUESTION, headers={"X-Forwarded-For": "203.0.113.7"})

    assert statuses == [200] * 10
    assert response.status_code == 429
    body = response.json()
    assert body["error"] == "RATE_LIMITED"
    assert set(body) == {"error", "detail", "hint"}
    assert response.headers["Retry-After"] == "60"


def test_retry_after_counts_down_and_the_window_slides(
    client: TestClient, clock: FakeClock
) -> None:
    install(client, clock.limiter(per_minute=2))
    headers = {"X-Forwarded-For": "203.0.113.7"}

    client.post("/api/v1/ask", json=QUESTION, headers=headers)
    clock.advance(20)
    client.post("/api/v1/ask", json=QUESTION, headers=headers)
    clock.advance(10)
    limited = client.post("/api/v1/ask", json=QUESTION, headers=headers)

    # The oldest request was 30s ago, so it leaves the window in 30s.
    assert limited.status_code == 429
    assert limited.headers["Retry-After"] == "30"

    clock.advance(30)
    assert client.post("/api/v1/ask", json=QUESTION, headers=headers).status_code == 200


def test_addresses_are_limited_independently(client: TestClient, clock: FakeClock) -> None:
    install(client, clock.limiter(per_minute=1))

    assert ask(client, "203.0.113.1").status_code == 200  # type: ignore[attr-defined]
    assert ask(client, "203.0.113.1").status_code == 429  # type: ignore[attr-defined]
    assert ask(client, "203.0.113.2").status_code == 200  # type: ignore[attr-defined]


def test_a_client_cannot_pick_its_own_bucket_by_forging_x_forwarded_for(
    client: TestClient, clock: FakeClock
) -> None:
    """The proxy appends the real address; whatever the client sent sits to its left."""
    install(client, clock.limiter(per_minute=1))

    first = ask(client, "198.51.100.1, 203.0.113.7")
    second = ask(client, "198.51.100.2, 203.0.113.7")

    assert first.status_code == 200  # type: ignore[attr-defined]
    assert second.status_code == 429  # type: ignore[attr-defined]


def _request(headers: dict[str, str], client: tuple[str, int] | None) -> Request:
    return Request(
        {
            "type": "http",
            "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
            "client": client,
        }
    )


def test_client_ip_falls_back_to_the_socket_peer_without_the_header() -> None:
    assert client_ip(_request({}, ("192.0.2.9", 1234))) == "192.0.2.9"
    assert client_ip(_request({"X-Forwarded-For": " , "}, ("192.0.2.9", 1234))) == "192.0.2.9"
    assert client_ip(_request({}, None)) == "unknown"


def test_a_request_rejected_by_the_ip_limit_never_reaches_the_provider(
    client: TestClient, clock: FakeClock, stub_provider: StubProvider
) -> None:
    install(client, clock.limiter(per_minute=1))

    ask(client)
    ask(client)

    assert stub_provider.calls == 1


# ---------------------------------------------------------------------------
# Global daily cap
# ---------------------------------------------------------------------------


def test_the_daily_cap_is_global_across_addresses_and_429s_with_retry_after(
    client: TestClient, clock: FakeClock, stub_provider: StubProvider
) -> None:
    install(client, clock.limiter(daily_cap=2))

    assert ask(client, "203.0.113.1").status_code == 200  # type: ignore[attr-defined]
    assert ask(client, "203.0.113.2").status_code == 200  # type: ignore[attr-defined]
    response = ask(client, "203.0.113.3")

    assert response.status_code == 429  # type: ignore[attr-defined]
    assert response.json()["error"] == "DAILY_CAP_REACHED"  # type: ignore[attr-defined]
    # 12:00 UTC to the next midnight.
    assert response.headers["Retry-After"] == str(12 * 3600)  # type: ignore[attr-defined]
    assert stub_provider.calls == 2


def test_the_daily_cap_resets_at_utc_midnight(client: TestClient, clock: FakeClock) -> None:
    install(client, clock.limiter(daily_cap=1))

    assert ask(client, "203.0.113.1").status_code == 200  # type: ignore[attr-defined]
    assert ask(client, "203.0.113.2").status_code == 429  # type: ignore[attr-defined]

    clock.advance(12 * 3600)

    assert ask(client, "203.0.113.3").status_code == 200  # type: ignore[attr-defined]


def test_requests_that_make_no_llm_call_do_not_spend_the_cap(
    client: TestClient, clock: FakeClock
) -> None:
    install(client, clock.limiter(daily_cap=1))

    invalid = client.post("/api/v1/ask", json={"question": ""})
    assert invalid.status_code == 422

    assert ask(client).status_code == 200  # type: ignore[attr-defined]


def test_an_unconfigured_provider_does_not_spend_the_cap(
    client: TestClient, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vedagraph.llm.errors import LLMConfigurationError

    def unconfigured() -> None:
        raise LLMConfigurationError("no key")

    install(client, clock.limiter(daily_cap=1))
    monkeypatch.setattr(ask_route, "get_llm_provider", unconfigured)

    assert ask(client).status_code == 503  # type: ignore[attr-defined]
    assert ask(client).status_code == 503  # type: ignore[attr-defined]


def test_a_daily_cap_of_zero_refuses_every_question(client: TestClient, clock: FakeClock) -> None:
    install(client, clock.limiter(daily_cap=0))

    assert ask(client).status_code == 429  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_the_limits_come_from_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VEDAGRAPH_ASK_DAILY_CAP", "42")
    monkeypatch.setenv("VEDAGRAPH_ASK_RATE_LIMIT_PER_MINUTE", "3")

    settings = ApiSettings()

    assert settings.ask_daily_cap == 42
    assert settings.ask_rate_limit_per_minute == 3


def test_the_defaults_are_ten_a_minute(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VEDAGRAPH_ASK_DAILY_CAP", raising=False)
    monkeypatch.delenv("VEDAGRAPH_ASK_RATE_LIMIT_PER_MINUTE", raising=False)

    assert ApiSettings(_env_file=None).ask_rate_limit_per_minute == 10  # type: ignore[call-arg]


def test_trusted_proxies_selects_the_entry_that_many_hops_from_the_right() -> None:
    header = {"X-Forwarded-For": "198.51.100.9, 203.0.113.7, 192.0.2.50"}

    assert client_ip(_request(header, None), 1) == "192.0.2.50"
    assert client_ip(_request(header, None), 2) == "203.0.113.7"
    assert client_ip(_request(header, None), 3) == "198.51.100.9"
    # Fewer entries than hops: unverifiable, so the socket peer, never a forgeable entry.
    assert client_ip(_request(header, ("192.0.2.9", 1)), 9) == "192.0.2.9"
    assert client_ip(_request({"X-Forwarded-For": "1.2.3.4"}, ("192.0.2.9", 1)), 2) == "192.0.2.9"


def test_two_trusted_proxies_separate_visitors_behind_a_shared_frontend(
    client: TestClient, clock: FakeClock
) -> None:
    """Visitor -> frontend -> host proxy: the last entry is the frontend for everyone."""
    install(client, clock.limiter(per_minute=1, trusted_proxies=2))

    assert ask(client, "198.51.100.1, 192.0.2.50").status_code == 200  # type: ignore[attr-defined]
    assert ask(client, "198.51.100.2, 192.0.2.50").status_code == 200  # type: ignore[attr-defined]
    assert ask(client, "198.51.100.1, 192.0.2.50").status_code == 429  # type: ignore[attr-defined]


def test_expired_addresses_are_swept_so_the_table_does_not_grow_forever(
    clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ratelimit, "_SWEEP_THRESHOLD", 50)
    limiter = clock.limiter()

    for n in range(200):
        limiter.check_ip(f"10.0.{n // 256}.{n % 256}")
    clock.advance(61)
    limiter.check_ip("10.9.9.9")

    assert len(limiter._hits) == 1
