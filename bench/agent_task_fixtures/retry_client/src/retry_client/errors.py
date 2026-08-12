"""Retry client errors."""


class RetryExhausted(RuntimeError):
    """Raised when every permitted attempt returns a retryable response."""
