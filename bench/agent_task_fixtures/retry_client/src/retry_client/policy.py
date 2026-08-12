"""Retry policy and exponential delay calculation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.25
    retryable_statuses: frozenset[int] = frozenset({429, 502, 503, 504})

    def delay_for(self, retry_index: int) -> float:
        return self.base_delay * (2**retry_index)
