"""Guardrail corrections need the Task context that produced the result."""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

from infinidev.engine.file_change_tracker import FileChangeTracker
from infinidev.engine.loop import guardrail_runner
from infinidev.engine.loop.action_record import ActionRecord
from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.loop.loop_plan import LoopPlan, PlanStep
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.loop.user_guidance import UserGuidance
from infinidev.engine.multimodal import ImageAttachment


def _response(text):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None))],
        usage=None,
    )


def _state():
    return LoopState(
        plan=LoopPlan(steps=[PlanStep(index=1, title="Verify tenant migration", status="active")]),
        notes=["Rollback must restore the original tenant identifiers."],
        history=[ActionRecord(step_index=0, summary="Observed legacy tenant foreign keys.")],
        user_guidance=[UserGuidance(text="Keep existing identifiers exactly.")],
        iteration_count=3,
    )


def _ctx(state):
    return SimpleNamespace(
        state=state, file_tracker=None, project_id=1, agent_id="developer",
        max_prompt_tokens=None, max_total_calls=None, history_window=0, start_iteration=0,
        desc="Implement tenant migration", expected="Verified rollback preserves tenant IDs",
        max_context_tokens=1_000_000, skip_plan=False, task=None, is_small=False,
        system_prompt="Developer", tool_schemas=[], resumed=False,
        guardrail=None, guardrail_max_retries=3, max_per_action=0, manual_tc=False,
        llm_params={}, tool_dispatch={},
    )


def _correct(ctx, guard, **kwargs):
    return guardrail_runner.apply_guardrail(
        ctx, "ORIGINAL", guard, 3, {}, ctx.system_prompt, ctx.desc, ctx.expected,
        ctx.state, [], {}, **kwargs,
    )


def test_standalone_correction_keeps_task_plan_notes_evidence_and_literal_guidance(monkeypatch):
    requests = []

    def call_llm(params, messages, schemas):
        requests.append(deepcopy(messages))
        return _response("CORRECTED")

    monkeypatch.setattr(guardrail_runner, "_call_llm", call_llm)
    ctx = _ctx(_state())

    result = _correct(ctx, lambda value: (value == "CORRECTED", "Explain the rollback."))

    assert result == "CORRECTED"
    projected = str(requests[0])
    for expected in (
        ctx.desc, ctx.expected, "Verify tenant migration",
        "Rollback must restore the original tenant identifiers.",
        "Observed legacy tenant foreign keys.", "Keep existing identifiers exactly.",
        "Explain the rollback.", "ORIGINAL",
    ):
        assert expected in projected


def test_context_callback_is_fresh_per_correction_and_its_messages_are_not_mutated(monkeypatch):
    source_messages = [
        {"role": "system", "content": "Original system"},
        {"role": "user", "content": "Scope revision 0"},
    ]
    original = deepcopy(source_messages)
    contexts = []
    requests = []

    def build_context():
        contexts.append(deepcopy(source_messages))
        return source_messages

    def call_llm(params, messages, schemas):
        requests.append(deepcopy(messages))
        source_messages[1]["content"] = "Scope revision 1"
        return _response(f"CORRECTION-{len(requests)}")

    monkeypatch.setattr(guardrail_runner, "_call_llm", call_llm)
    ctx = _ctx(LoopState())

    result = _correct(
        ctx, lambda value: (value == "CORRECTION-2", "Still missing a requirement"),
        build_context_messages=build_context,
    )

    assert result == "CORRECTION-2"
    assert len(contexts) == 2
    assert "Scope revision 0" in str(requests[0])
    assert "Scope revision 1" in str(requests[1])
    assert source_messages == [original[0], {"role": "user", "content": "Scope revision 1"}]
    assert len(requests[0]) == len(requests[1]) == 3


@pytest.mark.parametrize("accepted", [True, False])
def test_context_is_not_built_without_a_correction_budget(monkeypatch, accepted):
    def unexpected(*args, **kwargs):
        pytest.fail("an unnecessary correction consumed context or called the model")

    monkeypatch.setattr(guardrail_runner, "_call_llm", unexpected)
    ctx = _ctx(LoopState(total_prompt_tokens=10))
    ctx.max_prompt_tokens = 10

    result = _correct(
        ctx, lambda value: (accepted, "feedback"), build_context_messages=unexpected,
    )

    assert result == "ORIGINAL"


def test_loop_engine_uses_its_normal_context_in_a_guardrail_correction(monkeypatch):
    requests = []

    def call_llm(params, messages, schemas):
        requests.append(deepcopy(messages))
        return _response("CORRECTED")

    monkeypatch.setattr(guardrail_runner, "_call_llm", call_llm)
    engine = LoopEngine()
    engine._cr_hooks = SimpleNamespace(_enabled=False)
    engine._project_knowledge = [{
        "finding_type": "project_context", "topic": "tenant storage",
        "content": "Tenant IDs are externally assigned and immutable.",
    }]
    engine.session_notes = ["The deployment uses PostgreSQL 18."]
    ctx = _ctx(_state())

    ctx.guardrail = lambda value: (value == "CORRECTED", "Missing rollback evidence")
    result = engine._validate_final_result(ctx, "ORIGINAL")

    assert result.text == "CORRECTED"
    assert result.status == "done"
    projected = str(requests[0])
    assert "Tenant IDs are externally assigned and immutable." in projected
    assert "The deployment uses PostgreSQL 18." in projected
    assert "Keep existing identifiers exactly." in projected


@pytest.mark.parametrize(("tool_name", "exit_code"), [
    pytest.param("edit_file", 0, id="file-edit"),
    pytest.param("execute_command", 0, id="shell-edit"),
    pytest.param("execute_command", 1, id="failed-shell-with-partial-edit"),
])
def test_retry_context_refreshes_source_changed_by_a_corrective_tool(
    tmp_path, monkeypatch, tool_name, exit_code,
):
    from infinidev.tools.base.context import set_context

    path = tmp_path / "tenant.py"
    path.write_text("OLD_SOURCE = 1\n")
    set_context(agent_id="developer", workspace_path=str(tmp_path))
    state = LoopState()
    state.cache_file(str(path), "OLD_SOURCE = 1\n", pinned=True)
    ctx = _ctx(state)
    ctx.file_tracker = FileChangeTracker()
    requests = []
    arguments = (
        {"file_path": str(path), "old_string": "OLD_SOURCE", "new_string": "NEW_SOURCE"}
        if tool_name == "edit_file" else {"command": "generate tenant.py"}
    )

    def call_llm(params, messages, schemas):
        requests.append(deepcopy(messages))
        if len(requests) == 1:
            tool_call = SimpleNamespace(
                id="edit",
                function=SimpleNamespace(name=tool_name, arguments=json.dumps(arguments)),
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(
                    content="", tool_calls=[tool_call],
                ))],
                usage=None,
            )
        return _response("FIRST-DRAFT" if len(requests) == 2 else "CORRECTED")

    def execute(*args):
        path.write_text("NEW_SOURCE = 1\n")
        if exit_code:
            return json.dumps({"error": "Command timed out after writing source"})
        return json.dumps({
            "success": True, "exit_code": 0, "stdout": "", "stderr": "",
        })

    monkeypatch.setattr(guardrail_runner, "_call_llm", call_llm)
    monkeypatch.setattr(guardrail_runner, "execute_tool_call", execute)
    monkeypatch.setattr("infinidev.engine.tool_executor.reindex_if_enabled", lambda path: None)

    result = _correct(ctx, lambda value: (value == "CORRECTED", "Missing evidence"))

    assert result == "CORRECTED"
    final_context = str(requests[-1])
    assert "NEW_SOURCE" in final_context
    assert "OLD_SOURCE" not in final_context
    assert "NEW_SOURCE" in state.opened_files[str(path)].content


def test_worker_correction_receives_pending_user_guidance_and_images(tmp_path, monkeypatch):
    requests = []
    pending = [{
        "id": "user-1", "kind": "user_guidance", "author": "user",
        "content": "Use this tenant schema and preserve its identifiers.",
    }]
    attachment = ImageAttachment(
        path=tmp_path / "tenant-schema.png", mime_type="image/png",
        data_url="data:image/png;base64,WORKER_IMAGE", size_bytes=8,
    )

    def poll(actor):
        return json.dumps([pending.pop(0)]) if pending else None

    def call_llm(params, messages, schemas):
        requests.append(deepcopy(messages))
        return _response("CORRECTED")

    monkeypatch.setattr(guardrail_runner, "_call_llm", call_llm)
    engine = LoopEngine()
    engine._cr_hooks = SimpleNamespace(_enabled=False)
    engine._team_actor = "worker-1"
    engine._team_runtime = SimpleNamespace(
        poll=poll, guidance_attachments=lambda event_id: [attachment],
    )
    ctx = _ctx(LoopState(iteration_count=3))

    ctx.guardrail = lambda value: (value == "CORRECTED", "Missing rollback evidence")
    result = engine._validate_final_result(ctx, "ORIGINAL")

    assert result.text == "CORRECTED"
    assert result.status == "done"
    assert "Use this tenant schema and preserve its identifiers." in str(requests[0])
    image_urls = [
        block["image_url"]["url"]
        for message in requests[0] if isinstance(message["content"], list)
        for block in message["content"] if block.get("type") == "image_url"
    ]
    assert image_urls == [attachment.data_url]
    assert ctx.state.user_guidance[0].text == (
        "Use this tenant schema and preserve its identifiers."
    )
    assert "WORKER_IMAGE" not in ctx.state.model_dump_json()
