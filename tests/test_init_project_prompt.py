"""Offline capability selection for the project exploration prompt."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from infinidev.prompts import init_project


def test_init_without_finding_tools_returns_public_delivery_instructions():
    builder = getattr(init_project, "build_init_task_prompt", None)
    assert callable(builder), "init needs capability-aware prompt composition"
    description, expected = builder(tool_names={"read_file", "list_directory"})
    assert "No finding storage tool is offered" in description
    assert "Do not claim database persistence" in description
    assert "unrecorded" in expected
    assert "record_finding(" not in description
    assert "ken_remember(" not in description


def test_init_ken_guidance_does_not_send_local_finding_fields():
    builder = getattr(init_project, "build_init_task_prompt", None)
    assert callable(builder), "init needs capability-aware prompt composition"
    description, _ = builder(tool_names={"ken_remember", "read_file"})
    assert "Use ken_remember" in description
    assert 'finding_type="project_context"' not in description
    assert "confidence=1.0" not in description
    assert "offered schema" in description


def test_init_local_storage_uses_local_schema_only():
    builder = getattr(init_project, "build_init_task_prompt", None)
    assert callable(builder), "init needs capability-aware prompt composition"
    description, _ = builder(tool_names={"record_finding", "read_file"})
    assert "Use record_finding" in description
    assert 'finding_type="project_context"' in description
    assert "provisional" in description
    assert "ken_remember" not in description


def test_init_uses_one_backend_when_both_are_available():
    builder = getattr(init_project, "build_init_task_prompt", None)
    assert callable(builder), "init needs capability-aware prompt composition"
    description, _ = builder(tool_names={"record_finding", "ken_remember"})
    assert "Use ken_remember" in description
    assert "Use record_finding" not in description


def test_init_explicit_language_is_data_without_local_resources(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    builder = getattr(init_project, "build_init_task_prompt", None)
    assert callable(builder), "init needs capability-aware prompt composition"
    description, _ = builder(tool_names=set(), language="Español\nwith UTF-8 ñ")
    value, _ = json.JSONDecoder().raw_decode(
        description.split("Requested output language (JSON string): ", 1)[1]
    )
    assert value == "Español\nwith UTF-8 ñ"


@pytest.mark.parametrize("small_model", [False, True])
@pytest.mark.parametrize("style", ["full", "lean", "coding", "generalized", "extra_simple"])
def test_init_flow_assembles_system_identity_with_task_source_boundaries(
    tmp_path, monkeypatch, small_model, style,
):
    from infinidev.config import model_capabilities
    from infinidev.engine.loop import context_builder
    from infinidev.engine.orchestration.pipeline import run_flow_task
    from infinidev.tools.file.read_file_tool import ReadFileTool

    monkeypatch.setattr(context_builder, "_is_small_model", lambda: small_model)
    monkeypatch.setattr(context_builder.settings, "PROMPT_STYLE", style)
    monkeypatch.setattr(
        model_capabilities, "_capabilities",
        model_capabilities.ModelCapabilities(probed=True),
    )
    monkeypatch.setattr(context_builder, "get_litellm_params", lambda: {"model": "offline"})
    monkeypatch.setattr(
        context_builder, "get_model_capabilities",
        lambda: SimpleNamespace(supports_function_calling=True),
    )
    agent = SimpleNamespace(
        agent_id="init-offline", project_id=1, role="developer", name="Infi",
        backstory="", workspace_path=str(tmp_path), _system_prompt_identity="",
        tools=[ReadFileTool()], activate_context=lambda **_: None, deactivate=lambda: None,
    )
    captured = {}

    class OfflineEngine:
        _last_status = "done"
        _last_file_tracker = None

        def execute(self, *, task_prompt, **kwargs):
            context = context_builder.build_execution_context(
                self, kwargs["agent"], task_prompt, task_tools=agent.tools, verbose=False,
            )
            captured["system"] = context.system_prompt
            captured["task"] = context.desc
            return "Exploration not executed; offline assembly only."

    run_flow_task(
        agent=agent, flow="document", session_id="init-offline",
        task_prompt=init_project.build_init_task_prompt(tool_names={"read_file"}),
        engine=OfflineEngine(),
        hooks=SimpleNamespace(on_phase=lambda _: None, on_status=lambda *_: None),
    )
    assert "Only use sources and destinations authorized by the task" in captured["system"]
    assert "Use finding_type to categorize" not in captured["system"]
    assert "No finding storage tool is offered" in captured["task"]
