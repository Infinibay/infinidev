"""Tests for content-free per-section prompt measurements."""

from __future__ import annotations

from infinidev.engine.prompt_composition import (
    measure_prompt_composition,
    measure_request_payload,
    user_section_chars,
)


def test_top_level_sections_are_measured_without_counting_nested_tags_twice() -> None:
    prompt = (
        "<task>\n"
        "Do work.\n"
        "<constraint>nested</constraint>\n"
        "</task>\n\n"
        "<plan>\nOne step\n</plan>"
    )

    sections = user_section_chars(prompt)

    assert sections["task"] == len(prompt.split("\n\n", 1)[0])
    assert sections["plan"] == len(prompt.split("\n\n", 1)[1])
    assert "constraint" not in sections
    assert sections["unclassified"] == 2  # The separator itself.


def test_composition_includes_system_user_and_function_schema_costs() -> None:
    system = "system guidance"
    user = "<task>\nrepair it\n</task>"
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    result = measure_prompt_composition(system, user, tools, iteration=3)

    assert result["iteration"] == 3
    assert result["system_chars"] == len(system)
    assert result["user_chars"] == len(user)
    assert result["tool_schema_chars"] > 0
    assert result["request_static_chars"] == (
        result["system_chars"] + result["user_chars"] + result["tool_schema_chars"]
    )
    assert result["user_layer_chars"]["objective"] == len(user)


def test_request_payload_measures_growing_transcript_by_role() -> None:
    messages = [
        {"role": "system", "content": "rules"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "1"}]},
        {"role": "tool", "content": "a large result"},
    ]

    result = measure_request_payload(
        messages, [{"name": "read_file"}], mode="function_calling", sequence=2
    )

    assert result["sequence"] == 2
    assert result["message_count"] == 4
    assert result["message_content_chars_by_role"]["tool"] > 10
    assert result["request_payload_chars"] > result["message_payload_chars"]
