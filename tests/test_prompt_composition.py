"""Tests for conditional composition and content-free measurements."""

from __future__ import annotations

from infinidev.engine.prompt_composition import (
    CACHE_BREAKPOINT_MARKER,
    ConditionalPromptFragment,
    append_dynamic_system_layer,
    measure_prompt_composition,
    measure_request_payload,
    select_conditional_fragments,
    user_section_chars,
)
from infinidev.engine.task_policies.router import resolve_task_profile


def test_conditional_fragment_selection_enforces_axes_exclusions_and_budget() -> None:
    profile = resolve_task_profile(
        "Refactoriza este módulo sin cambiar su comportamiento."
    )
    fragments = (
        ConditionalPromptFragment(
            id="chosen",
            policy_id="refactor.preserve_behavior",
            content="preserve behavior",
            roles=frozenset({"developer"}),
            phases=frozenset({"execute"}),
            requires_operations=frozenset({"refactor"}),
            requires_authority=frozenset({"modify"}),
            priority=10,
        ),
        ConditionalPromptFragment(
            id="excluded",
            policy_id="refactor.preserve_behavior",
            content="wrong method",
            roles=frozenset({"developer"}),
            phases=frozenset({"execute"}),
            excludes_operations=frozenset({"refactor"}),
        ),
        ConditionalPromptFragment(
            id="over-budget",
            policy_id="refactor.preserve_behavior",
            content="x" * 200,
            roles=frozenset({"developer"}),
            phases=frozenset({"execute"}),
        ),
    )

    result = select_conditional_fragments(
        profile,
        fragments,
        role="developer",
        phase="execute",
        max_utf8_bytes=40,
    )

    assert [fragment.id for fragment in result.fragments] == ["chosen"]
    assert ("excluded", "excluded-operation") in result.omitted
    assert ("over-budget", "budget") in result.omitted


def test_model_route_fragments_compose_with_generic_fragments() -> None:
    profile = resolve_task_profile("Corrige este fallo reproducible.")
    fragments = (
        ConditionalPromptFragment(
            id="generic",
            policy_id="bugfix.root_cause",
            content="generic method",
            roles=frozenset({"developer"}),
            phases=frozenset({"execute"}),
        ),
        ConditionalPromptFragment(
            id="gpt56-only",
            policy_id="bugfix.root_cause",
            content="model adjustment",
            roles=frozenset({"developer"}),
            phases=frozenset({"execute"}),
            model_routes=frozenset({"openai_subscription:gpt-5.6"}),
        ),
        ConditionalPromptFragment(
            id="not-sol",
            policy_id="bugfix.root_cause",
            content="excluded adjustment",
            roles=frozenset({"developer"}),
            phases=frozenset({"execute"}),
            excluded_model_routes=frozenset({"openai_subscription:gpt-5.6-sol"}),
        ),
    )

    result = select_conditional_fragments(
        profile,
        fragments,
        role="developer",
        phase="execute",
        max_utf8_bytes=100,
        provider="openai_subscription",
        model="openai/responses/gpt-5.6-sol",
    )

    assert [fragment.id for fragment in result.fragments] == ["generic", "gpt56-only"]
    assert ("not-sol", "excluded-model-route") in result.omitted


def test_model_specific_fragment_fails_closed_without_a_route() -> None:
    profile = resolve_task_profile("Corrige este fallo reproducible.")
    fragment = ConditionalPromptFragment(
        id="model-only",
        policy_id="bugfix.root_cause",
        content="adjustment",
        roles=frozenset({"developer"}),
        phases=frozenset({"execute"}),
        model_routes=frozenset({"minimax:minimax-m3"}),
    )

    result = select_conditional_fragments(
        profile,
        (fragment,),
        role="developer",
        phase="execute",
        max_utf8_bytes=100,
    )

    assert result.fragments == ()
    assert result.omitted == (("model-only", "model-route-unavailable"),)


def test_dynamic_layer_stays_after_the_cacheable_prefix() -> None:
    result = append_dynamic_system_layer(
        "stable core",
        "task-local method",
        cache_boundary=True,
    )

    assert result.startswith(f"stable core\n\n{CACHE_BREAKPOINT_MARKER}")
    assert result.index(CACHE_BREAKPOINT_MARKER) < result.index("task-local method")


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
    system = (
        "system guidance\n\n"
        f"{CACHE_BREAKPOINT_MARKER}\n\n"
        '<prompt-fragment id="refactor.developer" version="1" '
        f'sha256="{"a" * 64}" policy="refactor.preserve_behavior@1">\n'
        "method\n</prompt-fragment>"
    )
    user = "<task>\nrepair it\n</task>"
    tools = [{"type": "function", "function": {"name": "read_file"}}]

    result = measure_prompt_composition(system, user, tools, iteration=3)

    assert result["iteration"] == 3
    assert result["system_chars"] == len(system)
    assert result["stable_system_chars"] < result["system_chars"]
    assert result["dynamic_system_chars"] > 0
    assert result["conditional_fragment_ids"] == ["refactor.developer"]
    assert result["conditional_fragments"][0]["policy"] == (
        "refactor.preserve_behavior@1"
    )
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
