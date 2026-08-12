"""Transport contract."""

from typing import Protocol

from retry_client.models import Response


class Transport(Protocol):
    def send(self, request: object) -> Response: ...
