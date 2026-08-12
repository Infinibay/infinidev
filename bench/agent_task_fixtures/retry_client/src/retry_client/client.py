"""Retry orchestration."""

from __future__ import annotations

from retry_client.clock import Clock
from retry_client.errors import RetryExhausted
from retry_client.models import Response
from retry_client.policy import RetryPolicy
from retry_client.retry_after import parse_retry_after
from retry_client.transport import Transport


class RetryClient:
    def __init__(self, transport: Transport, clock: Clock, policy: RetryPolicy) -> None:
        self.transport = transport
        self.clock = clock
        self.policy = policy

    def send(self, request: object) -> Response:
        for attempt in range(self.policy.max_attempts):
            response = self.transport.send(request)
            if response.status not in self.policy.retryable_statuses:
                return response
            header = response.headers.get("Retry-After")
            delay = (
                parse_retry_after(header, now=self.clock.now())
                if header is not None
                else self.policy.delay_for(attempt)
            )
            self.clock.sleep(delay)
        raise RetryExhausted(f"request failed after {self.policy.max_attempts} attempts")
