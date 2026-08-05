"""Memory-safety bounds for untrusted subprocess and provider streams."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from infinidev.engine import llm_client
from infinidev.tools.shell.execute_command_tool import _extend_bounded


def test_foreground_output_buffer_keeps_only_bounded_tail() -> None:
    buffer = bytearray(b"old")

    discarded = _extend_bounded(buffer, b"0123456789", 6)

    assert discarded == 7
    assert bytes(buffer) == b"456789"


def test_llm_stream_stops_before_unbounded_chunk_retention(monkeypatch) -> None:
    class Stream:
        def __init__(self) -> None:
            self.closed = False

        def __iter__(self):
            delta = SimpleNamespace(
                reasoning_content="x", content="", tool_calls=None,
            )
            while True:
                yield SimpleNamespace(choices=[SimpleNamespace(delta=delta)])

        def close(self) -> None:
            self.closed = True

    stream = Stream()
    litellm = SimpleNamespace(completion=lambda **kwargs: stream)
    monkeypatch.setattr(llm_client, "_MAX_STREAM_CHUNKS", 3)

    with pytest.raises(RuntimeError, match="safety limit"):
        llm_client._stream_and_assemble(litellm, {}, lambda chunk: None)

    assert stream.closed is True
