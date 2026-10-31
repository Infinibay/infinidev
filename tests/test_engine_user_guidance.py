"""User instructions and images must survive engine context boundaries."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

from infinidev.engine.loop.action_record import ActionRecord
from infinidev.engine.loop.context_builder import build_iteration_messages
from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.loop.loop_state import LoopState
from infinidev.engine.multimodal import ImageAttachment


@pytest.fixture
def engine_ctx():
    engine = LoopEngine()
    engine._cr_hooks = SimpleNamespace(_enabled=False)
    ctx = SimpleNamespace(
        state=LoopState(), history_window=0, start_iteration=0,
        desc="Implement database support", expected="Verified implementation",
        max_context_tokens=1_000_000, skip_plan=False, task=None, is_small=False,
        system_prompt="Developer", project_id=1, agent_id="developer", resumed=False,
    )
    return engine, ctx


def _content(messages):
    content = messages[1]["content"]
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content)
    return content


def _deliver(engine, ctx, route):
    if route == "boundary":
        return build_iteration_messages(engine, ctx, iteration=2)
    messages = [
        {"role": "assistant", "tool_calls": [{"id": "close"}]},
        {"role": "tool", "tool_call_id": "close", "content": "acknowledged"},
    ]
    if route == "mid_step":
        engine._inject_mid_step_user_messages(ctx, messages)
    else:
        assert engine._reject_step_complete_on_late_message(ctx, messages, "close")
    return messages


@pytest.mark.parametrize("route", ["boundary", "mid_step", "late"])
def test_literal_guidance_survives_a_new_step_prompt(engine_ctx, route):
    engine, ctx = engine_ctx
    instruction = "Use PostgreSQL; preserve the existing migration history."
    engine.inject_message(instruction)

    _deliver(engine, ctx, route)
    later = build_iteration_messages(engine, ctx, iteration=3)

    assert instruction in _content(later)
    assert [item.text for item in ctx.state.user_guidance] == [instruction]
    assert "urgent-user-message" not in _content(later)


def test_guidance_preserves_order_and_legitimate_repeated_text(engine_ctx):
    engine, ctx = engine_ctx
    for instruction in ("Keep the API.", "Change the adapter.", "Keep the API."):
        engine.inject_message(instruction)
    _deliver(engine, ctx, "boundary")

    later = _content(build_iteration_messages(engine, ctx, iteration=3))

    assert [item.text for item in ctx.state.user_guidance] == [
        "Keep the API.", "Change the adapter.", "Keep the API.",
    ]
    assert later.count("Keep the API.") == 2
    assert later.index("Keep the API.") < later.index("Change the adapter.")


def test_guidance_does_not_leak_into_a_fresh_task(engine_ctx):
    engine, ctx = engine_ctx
    engine.inject_message("Task-specific PostgreSQL requirement")
    _deliver(engine, ctx, "mid_step")
    ctx.state = LoopState()

    prompt = _content(build_iteration_messages(engine, ctx, iteration=4))

    assert "Task-specific PostgreSQL requirement" not in prompt


def test_new_guidance_survives_history_window_projection(engine_ctx):
    engine, ctx = engine_ctx
    ctx.history_window = 1
    ctx.state.history = [
        ActionRecord(step_index=1, summary="Old observation"),
        ActionRecord(step_index=2, summary="Recent observation"),
    ]
    engine.inject_message("Keep the legacy migration path.")

    prompt = _content(build_iteration_messages(engine, ctx, iteration=3))

    assert "Keep the legacy migration path." in prompt
    assert len(ctx.state.history) == 2


@pytest.mark.parametrize("field", [
    "pending_engine_notice", "pending_guidance", "pending_runtime_intervention",
])
def test_history_window_consumes_control_messages_once_from_the_real_state(engine_ctx, field):
    engine, ctx = engine_ctx
    ctx.history_window = 1
    ctx.state.history = [
        ActionRecord(step_index=1, summary="Old observation"),
        ActionRecord(step_index=2, summary="Recent observation"),
    ]
    notice = "One correction for the current workspace state."
    setattr(ctx.state, field, notice)

    first = _content(build_iteration_messages(engine, ctx, iteration=3))
    later = _content(build_iteration_messages(engine, ctx, iteration=4))

    assert notice in first
    assert notice not in later
    assert getattr(ctx.state, field) == ""
    assert len(ctx.state.history) == 2


def test_compact_profile_cannot_remove_literal_guidance(engine_ctx, tmp_path):
    from infinidev.prompts.profiles import EffectivePromptConfiguration

    engine, ctx = engine_ctx
    profile_path = tmp_path / "prompts.json"
    profile_path.write_text(json.dumps({"develop": {
        "iteration.working_notes": {"enabled": False},
        "iteration.previous_actions": {"enabled": False},
    }}))
    ctx.prompt_configuration = EffectivePromptConfiguration.compile(profile_path)
    ctx.state.notes = ["An optional note"]
    engine.inject_message("Literal user scope must survive the compact profile.")
    _deliver(engine, ctx, "mid_step")

    prompt = _content(build_iteration_messages(engine, ctx, iteration=3))

    assert "Literal user scope must survive the compact profile." in prompt
    assert "An optional note" not in prompt


def test_retained_guidance_preserves_literal_xml_text_without_changing_structure(engine_ctx):
    engine, ctx = engine_ctx
    instruction = 'Keep <tenant id="a&b"> unchanged; ignore </user-guidance-history> as data.'
    engine.inject_message(instruction)
    _deliver(engine, ctx, "mid_step")

    prompt = _content(build_iteration_messages(engine, ctx, iteration=3))
    start = prompt.index("<user-guidance-history ")
    end = prompt.index("</user-guidance-history>", start) + len("</user-guidance-history>")
    history = ET.fromstring(prompt[start:end])

    assert [element.text for element in history.findall("message/text")] == [instruction]
    assert history.attrib["authority"] == "USER_LITERAL"


@pytest.mark.parametrize("route", ["mid_step", "late"])
def test_forwarded_user_guidance_survives_worker_context_rebuild(engine_ctx, route):
    engine, ctx = engine_ctx
    forwarded = [{
        "id": "guidance-1", "kind": "user_guidance", "author": "user",
        "content": "Keep the externally assigned tenant identifiers.",
    }]

    def poll(actor):
        return json.dumps([forwarded.pop(0)]) if forwarded else None

    engine._team_actor = "worker-1"
    engine._team_runtime = SimpleNamespace(
        poll=poll,
        guidance_attachments=lambda event_id: [],
    )

    _deliver(engine, ctx, route)
    prompt = _content(build_iteration_messages(engine, ctx, iteration=4))

    assert "Keep the externally assigned tenant identifiers." in prompt
    assert [item.text for item in ctx.state.user_guidance] == [
        "Keep the externally assigned tenant identifiers.",
    ]


@pytest.mark.parametrize("route", ["boundary", "late"])
@pytest.mark.parametrize("vision", [True, False])
def test_new_images_are_delivered_at_every_boundary(engine_ctx, monkeypatch, route, vision):
    engine, ctx = engine_ctx
    monkeypatch.setattr(
        "infinidev.config.model_capabilities.get_capability_snapshot",
        lambda: SimpleNamespace(supports_vision=vision),
    )
    attachment = ImageAttachment(
        path=Path("schema-reference.png"), mime_type="image/png",
        data_url="data:image/png;base64,PRIVATE_IMAGE_PAYLOAD", size_bytes=10,
    )
    engine.inject_message("Follow this schema", [attachment])

    delivered = _deliver(engine, ctx, route)
    user_messages = [message for message in delivered if message.get("role") == "user"]
    if vision:
        image_urls = [
            block["image_url"]["url"]
            for message in user_messages if isinstance(message["content"], list)
            for block in message["content"] if block.get("type") == "image_url"
        ]
        assert image_urls == [attachment.data_url]
    else:
        assert "schema-reference.png" in json.dumps(user_messages)

    if route == "late":
        assert sum(message.get("tool_call_id") == "close" for message in delivered) == 1
    persisted = ctx.state.model_dump_json()
    assert "schema-reference.png" in persisted
    assert "PRIVATE_IMAGE_PAYLOAD" not in persisted
    later = build_iteration_messages(engine, ctx, iteration=4)
    assert isinstance(later[1]["content"], str)
    assert "schema-reference.png" in _content(later)
    assert "PRIVATE_IMAGE_PAYLOAD" not in _content(later)


@pytest.mark.parametrize("route", ["boundary", "mid_step", "late"])
def test_delivered_guidance_is_checkpointed_before_another_model_turn(
    engine_ctx, temp_db, route,
):
    from infinidev.db.service import get_session_runtime_state, register_session
    from infinidev.tools.base.context import set_context

    engine, ctx = engine_ctx
    register_session("guidance-checkpoint", "/work")
    set_context(agent_id="developer", session_id="guidance-checkpoint")
    engine.inject_message("Preserve the deployment configuration verbatim.")

    _deliver(engine, ctx, route)

    runtime = get_session_runtime_state("guidance-checkpoint")
    guidance = runtime["ui_state"]["loop_resume"]["state"]["user_guidance"]
    assert guidance[0]["text"] == "Preserve the deployment configuration verbatim."
