"""Tests for the lock shared by every subscription-backed evaluator."""

from __future__ import annotations

from pathlib import Path

import pytest

from infinidev.engine.subscription_safety import (
    pace_llm_request,
    paced_llm_requests,
    subscription_single_flight,
)


def test_subscription_lock_rejects_a_second_campaign(tmp_path: Path) -> None:
    path = tmp_path / "global.lock"

    with subscription_single_flight(path):
        with pytest.raises(RuntimeError, match="another subscription-backed evaluation"):
            with subscription_single_flight(path):
                pass


def test_request_pacing_is_scoped_and_waits_between_calls(monkeypatch) -> None:
    timeline = iter([10.0, 10.0, 10.25, 12.0])
    sleeps: list[float] = []
    monkeypatch.setattr("infinidev.engine.subscription_safety.time.monotonic", lambda: next(timeline))
    monkeypatch.setattr("infinidev.engine.subscription_safety.time.sleep", sleeps.append)

    with paced_llm_requests(2.0):
        pace_llm_request()
        pace_llm_request()

    assert sleeps == [1.75]


def test_request_pacing_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setattr(
        "infinidev.engine.subscription_safety.time.sleep",
        lambda _: (_ for _ in ()).throw(AssertionError("unexpected sleep")),
    )
    pace_llm_request()
