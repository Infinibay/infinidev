from datetime import datetime, timezone

import pytest

from retry_client import Response, RetryClient, RetryPolicy
from retry_client.errors import RetryExhausted


class FakeClock:
    def __init__(self) -> None:
        self.sleeps: list[float] = []

    def now(self) -> datetime:
        return datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


class FakeTransport:
    def __init__(self, responses: list[Response]) -> None:
        self.responses = iter(responses)

    def send(self, request: object) -> Response:
        return next(self.responses)


def test_http_date_controls_delay() -> None:
    clock = FakeClock()
    transport = FakeTransport([
        Response(503, {"Retry-After": "Tue, 11 Aug 2026 12:00:30 GMT"}),
        Response(200),
    ])

    assert RetryClient(transport, clock, RetryPolicy()).send(object()).status == 200
    assert clock.sleeps == [30]


def test_exponential_fallback_and_success() -> None:
    clock = FakeClock()
    transport = FakeTransport([Response(503), Response(502), Response(200)])

    assert RetryClient(transport, clock, RetryPolicy(base_delay=0.5)).send(object()).status == 200
    assert clock.sleeps == [0.5, 1.0]


def test_exhaustion_never_sleeps_after_final_attempt() -> None:
    clock = FakeClock()
    transport = FakeTransport([Response(503), Response(503), Response(503)])

    with pytest.raises(RetryExhausted):
        RetryClient(transport, clock, RetryPolicy(max_attempts=3)).send(object())

    assert clock.sleeps == [0.25, 0.5]
