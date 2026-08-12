from datetime import datetime, timezone

from retry_client.retry_after import parse_retry_after


NOW = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def test_delta_seconds() -> None:
    assert parse_retry_after("12", now=NOW) == 12


def test_http_date() -> None:
    assert parse_retry_after("Tue, 11 Aug 2026 12:00:30 GMT", now=NOW) == 30


def test_past_http_date_clamps_to_zero() -> None:
    assert parse_retry_after("Tue, 11 Aug 2026 11:59:00 GMT", now=NOW) == 0
