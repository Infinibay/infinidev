"""Provider reasoning normalization and history-preservation tests."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.behavior.reasoning_content import (
    ReasoningStreamAccumulator,
    extract_reasoning,
    reasoning_history_fields,
)


def test_reasoning_content_is_the_common_normalized_path() -> None:
    result = extract_reasoning(SimpleNamespace(reasoning_content="visible summary"))

    assert result.text == "visible summary"
    assert result.sources == ("reasoning_content",)
    assert result.visibility == "provider_exposed"


def test_minimax_reasoning_details_are_read_without_signatures() -> None:
    message = SimpleNamespace(
        reasoning_details=[
            {"type": "reasoning.text", "text": "inspect the failing parser"},
            {"type": "reasoning.signature", "signature": "opaque-secret"},
        ]
    )

    result = extract_reasoning(message)

    assert result.text == "inspect the failing parser"
    assert "opaque-secret" not in result.text
    assert result.sources == ("reasoning_details",)


def test_anthropic_redacted_blocks_are_preserved_but_never_classified() -> None:
    blocks = [
        {"type": "thinking", "thinking": "I should run the focused test", "signature": "sig"},
        {"type": "redacted_thinking", "data": "encrypted"},
    ]
    message = SimpleNamespace(thinking_blocks=blocks)

    result = extract_reasoning(message)
    history = reasoning_history_fields(message)

    assert result.text == "I should run the focused test"
    assert "encrypted" not in result.text
    assert history["thinking_blocks"] == blocks


def test_gemini_thought_signatures_are_protocol_only() -> None:
    message = SimpleNamespace(
        reasoning_content="check the returned status",
        provider_specific_fields={"thought_signatures": ["opaque-a"]},
    )

    result = extract_reasoning(message)
    history = reasoning_history_fields(message)

    assert result.text == "check the returned status"
    assert "opaque-a" not in result.text
    assert history["provider_specific_fields"] == {
        "thought_signatures": ["opaque-a"]
    }


def test_minimax_cumulative_stream_snapshots_emit_only_the_suffix() -> None:
    accumulator = ReasoningStreamAccumulator()

    first = accumulator.consume(
        SimpleNamespace(reasoning_details=[{"text": "inspect"}])
    )
    second = accumulator.consume(
        SimpleNamespace(reasoning_details=[{"text": "inspect parser"}])
    )
    duplicate = accumulator.consume(
        SimpleNamespace(reasoning_details=[{"text": "inspect parser"}])
    )

    assert first == "inspect"
    assert second == " parser"
    assert duplicate == ""
