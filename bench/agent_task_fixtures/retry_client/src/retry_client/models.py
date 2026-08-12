"""Transport-neutral response model."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Response:
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
