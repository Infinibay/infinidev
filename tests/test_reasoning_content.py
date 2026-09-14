"""Provider reasoning normalization and history-preservation tests."""

from __future__ import annotations

from types import SimpleNamespace

from infinidev.engine.behavior.reasoning_content import (
    ReasoningStreamAccumulator,
    extract_reasoning,
    reasoning_history_fields,
    trim_superseded_reasoning,
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


# ── superseded-reasoning trimming ──────────────────────────────────────
#
# MiniMax-M3 bills a re-sent reasoning field at one prompt token per eight
# characters (measured against the live API), and the loop re-sends every
# assistant turn on every later request. Only the newest assistant turn still
# needs its reasoning: that is the one whose tool results travel with it.


def _minimax_message(reasoning: str, call_id: str) -> dict:
    """One assistant turn shaped exactly as MiniMax-M3 returns it."""
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": reasoning,
        "provider_specific_fields": {
            "name": "MiniMax AI",
            "reasoning_details": [
                {
                    "type": "reasoning.text",
                    "id": "reasoning-text-1",
                    "format": "MiniMax-response-v1",
                    "index": 0,
                    "text": reasoning,
                }
            ],
        },
        "tool_calls": [
            {"id": call_id, "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
        ],
    }


def test_trim_drops_reasoning_from_superseded_turns_only() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        _minimax_message("first round deliberation", "call_1"),
        {"role": "tool", "content": "result", "tool_call_id": "call_1"},
        _minimax_message("second round deliberation", "call_2"),
    ]

    removed = trim_superseded_reasoning(messages)

    assert removed > 0
    # The newest turn keeps both copies of its reasoning.
    assert messages[3]["reasoning_content"] == "second round deliberation"
    assert messages[3]["provider_specific_fields"]["reasoning_details"][0]["text"] == (
        "second round deliberation"
    )
    # The closed turn keeps neither, and loses the now-empty provider block.
    assert "reasoning_content" not in messages[1]
    assert "provider_specific_fields" not in messages[1]
    # Nothing else about the turn changed: the tool chain stays valid.
    assert messages[1]["tool_calls"] == [
        {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{}"}}
    ]
    assert messages[2]["content"] == "result"


def test_trim_is_a_no_op_before_a_second_assistant_turn() -> None:
    messages = [_minimax_message("only turn", "call_1")]

    assert trim_superseded_reasoning(messages) == 0
    assert messages[0]["reasoning_content"] == "only turn"


def test_trim_never_drops_opaque_signature_material() -> None:
    """Anthropic rejects a tool-use chain whose thinking blocks came back bare."""
    messages = [
        {
            "role": "assistant",
            "content": "",
            "thinking_blocks": [
                {"type": "thinking", "thinking": "prose that could go", "signature": "sig-abc"},
            ],
            "tool_calls": [{"id": "call_1"}],
        },
        {"role": "tool", "content": "r", "tool_call_id": "call_1"},
        {
            "role": "assistant",
            "content": "",
            "thinking_blocks": [{"type": "redacted_thinking", "data": "encrypted"}],
            "provider_specific_fields": {"thought_signatures": ["opaque-token"]},
            "tool_calls": [{"id": "call_2"}],
        },
    ]

    removed = trim_superseded_reasoning(messages)

    assert removed == 0
    assert messages[0]["thinking_blocks"][0]["signature"] == "sig-abc"
    assert messages[2]["provider_specific_fields"]["thought_signatures"] == ["opaque-token"]


def test_trim_keeps_a_signature_bearing_field_and_drops_a_plain_one_beside_it() -> None:
    messages = [
        {
            "role": "assistant",
            "content": "",
            "reasoning_details": [{"type": "reasoning.text", "text": "plain"}],
            "provider_specific_fields": {
                "reasoning_details": [{"type": "reasoning.text", "text": "plain"}],
                "thought_signatures": ["keep-me"],
            },
        },
        {"role": "assistant", "content": "", "reasoning_content": "newest"},
    ]

    trim_superseded_reasoning(messages)

    assert "reasoning_details" not in messages[0]
    assert "reasoning_details" not in messages[0]["provider_specific_fields"]
    assert messages[0]["provider_specific_fields"]["thought_signatures"] == ["keep-me"]


def test_trim_handles_a_transcript_without_assistant_turns() -> None:
    messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "go"}]

    assert trim_superseded_reasoning(messages) == 0
