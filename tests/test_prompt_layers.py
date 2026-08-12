from __future__ import annotations

from infinidev.engine.loop.context import build_system_prompt
from infinidev.engine.prompt_layers import (
    PromptLayer,
    PromptLayerKind,
    append_to_layer,
    classify_user_section,
    compose_layers,
)


def test_prompt_layer_kinds_are_distinct() -> None:
    assert {kind.value for kind in PromptLayerKind} == {
        "behavior",
        "execution-policy",
        "task-policy",
        "objective",
        "context-evidence",
    }


def test_system_prompt_marks_behavior_and_execution_policy() -> None:
    prompt = build_system_prompt("unused")

    assert '<behavior-layer provenance="infinidev-core">' in prompt
    assert '<execution-policy-layer provenance="infinidev-harness-and-repository">' in prompt
    assert prompt.index("<behavior-layer") < prompt.index("<execution-policy-layer")


def test_append_to_layer_does_not_leak_behavior_into_execution_policy() -> None:
    prompt = compose_layers(
        [
            PromptLayer(PromptLayerKind.BEHAVIOR, "identity", "core"),
            PromptLayer(PromptLayerKind.EXECUTION_POLICY, "run tests", "harness"),
        ]
    )

    result = append_to_layer(
        prompt,
        PromptLayerKind.BEHAVIOR,
        "prefer concise handoffs",
        provenance="study",
    )

    behavior = result.split("</behavior-layer>", 1)[0]
    execution = result.split("<execution-policy-layer", 1)[1]
    assert "prefer concise handoffs" in behavior
    assert "prefer concise handoffs" not in execution


def test_existing_user_sections_have_explicit_layer_semantics() -> None:
    assert classify_user_section("task") is PromptLayerKind.OBJECTIVE
    assert classify_user_section("current-action") is PromptLayerKind.EXECUTION_POLICY
    assert classify_user_section("project-knowledge") is PromptLayerKind.CONTEXT_EVIDENCE
