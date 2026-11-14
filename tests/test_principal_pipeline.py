"""The default pipeline owns conversation rather than manufacturing Tasks."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from infinidev.engine.orchestration import pipeline
from infinidev.engine.turns.models import TurnResult, TurnState


@pytest.fixture
def boundary(monkeypatch, temp_db, workspace_dir):
    from infinidev.config.settings import settings
    from infinidev.db.service import register_session
    from infinidev.engine.task_runtime import TaskRuntime

    register_session('principal-session')
    monkeypatch.setattr(settings, "TASK_ENGINE_MODE", "task")
    monkeypatch.setattr(settings, "GATHER_ENABLED", False)
    monkeypatch.setattr(settings, "TASK_MAX_ITERATIONS", 10)
    monkeypatch.setattr(settings, "TASK_MAX_TOOL_CALLS", 20)
    for name in ("add_task", "start_next_task", "complete_current_task", "record_step"):
        monkeypatch.setattr(TaskRuntime, name, Mock(side_effect=AssertionError(name)))
    monkeypatch.setattr(pipeline, "_ken_turn_context", lambda *a: "ken context")
    monkeypatch.setattr(pipeline, "_report_turn_end_to_ken", Mock())
    monkeypatch.setattr(pipeline, "_run_task_start_hook", lambda **kw: "start context")
    monkeypatch.setattr(pipeline, "_task_end_hook", lambda *a, **kw: "")
    monkeypatch.setattr("infinidev.engine.session_context.render_session_context",
                        lambda *a: "previous conversation")
    monkeypatch.setattr("infinidev.engine.orchestration.chat_agent.run_chat_agent",
                        Mock(side_effect=AssertionError("preclassification")))
    hooks = Mock()
    hooks.reply_already_shown = False
    agent = SimpleNamespace(agent_id="principal-test", project_id=None,
                            activate_context=Mock(), deactivate=Mock())
    engine = SimpleNamespace(begin_turn=Mock(), is_cancelled=False,
                             run_turn=Mock(return_value=TurnResult(
                                 message="respuesta", status="answered", state=TurnState(),
                                 metrics={"observed_iterations": 2, "observed_tool_calls": 3})))
    return agent, engine, hooks


def call(boundary, text="hola", **kwargs):
    agent, engine, hooks = boundary
    return pipeline.run_task(agent=agent, engine=engine, hooks=hooks, reviewer=None,
                             session_id="principal-session", user_input=text, **kwargs)


@pytest.mark.parametrize("text", ["hola", "investiga sin cambiar archivos",
                                   "modifica solamente los tests", "seguí con eso"])
def test_default_requests_reach_agent_without_task_or_classifier(boundary, text):
    assert call(boundary, text) == "respuesta"
    agent, engine, hooks = boundary
    args, kwargs = engine.run_turn.call_args
    assert args == (agent, text)
    assert "previous conversation" in kwargs["turn_context"]
    assert "ken context" in kwargs["turn_context"]
    assert "start context" in kwargs["turn_context"]
    hooks.notify.assert_called_once_with("Infinidev", "respuesta", "agent")
    hooks.notify_stream_end.assert_called_once()
    hooks.mark_reply_shown.assert_called_once()
    agent.deactivate.assert_called_once()


def test_attachments_remain_separate_from_literal_input(boundary):
    attachments = [object()]
    call(boundary, " literal  ", attachments=attachments)
    assert boundary[1].run_turn.call_args.args[1] == " literal  "
    assert boundary[1].run_turn.call_args.kwargs["initial_attachments"] == attachments


def test_late_cancellation_preempts_answer(boundary, monkeypatch):
    engine = boundary[1]
    def run(*a, **kw):
        engine.is_cancelled = True
        return TurnResult(message="success", status="answered", state=TurnState())
    engine.run_turn.side_effect = run
    statuses = []
    monkeypatch.setattr(pipeline, "_task_end_hook",
                        lambda *a, **kw: statuses.append(kw["status"]) or "")
    assert "success" not in call(boundary)
    assert statuses == ["cancelled", "cancelled"]


def test_hook_continuation_uses_remaining_budget_without_new_turn(boundary, monkeypatch):
    from infinidev.engine.user_hooks import UserHookEvent
    monkeypatch.setattr(pipeline, "_task_end_hook", lambda event, **kw:
                        "check again" if event == UserHookEvent.TASK_END_INSTRUCTION else "")
    call(boundary)
    engine = boundary[1]
    assert engine.run_turn.call_count == 2
    assert engine.run_turn.call_args_list[0].kwargs["preserve_file_tracker"] is False
    assert engine.run_turn.call_args.kwargs["preserve_file_tracker"] is True
    assert engine.run_turn.call_args.kwargs["max_iterations"] == 8
    assert engine.run_turn.call_args.kwargs["max_total_tool_calls"] == 17
    assert engine.run_turn.call_args.args[1] == "hola"
    assert "check again" in engine.run_turn.call_args.kwargs["turn_context"]
    engine.begin_turn.assert_called_once()


@pytest.mark.parametrize("bad", [None, "Done", SimpleNamespace(status="completed", message="ok"),
                                  TurnResult(message="", status="answered", state=TurnState())])
def test_invalid_result_fails_closed(boundary, monkeypatch, bad):
    statuses = []
    boundary[1].run_turn.return_value = bad
    monkeypatch.setattr(pipeline, "_task_end_hook",
                        lambda *a, **kw: statuses.append(kw["status"]) or "")
    assert "failed" in call(boundary).lower()
    assert statuses == ["failed", "failed"]


def test_default_never_infers_repository_from_prose(boundary, monkeypatch):
    monkeypatch.setattr(
        "infinidev.engine.orchestration.request_signals.resolve_referenced_repository",
        Mock(side_effect=AssertionError("repository prose inference")),
    )
    call(boundary, "Read /other/checkout/README.md")


@pytest.mark.parametrize("metrics, expected", [
    ({"observed_iterations": 10, "observed_tool_calls": 1}, "exhausted"),
    ({}, "failed"),
    ({"observed_iterations": True, "observed_tool_calls": 1}, "failed"),
])
def test_hook_cannot_restart_exhausted_or_unaccounted_budget(boundary, monkeypatch,
                                                          metrics, expected):
    from infinidev.engine.user_hooks import UserHookEvent
    statuses = []
    def end_hook(event, **kwargs):
        statuses.append(kwargs["status"])
        return "continue" if event == UserHookEvent.TASK_END_INSTRUCTION else ""
    monkeypatch.setattr(pipeline, "_task_end_hook", end_hook)
    boundary[1].run_turn.return_value = TurnResult.model_construct(
        message="partial response", status="answered", state=TurnState(), metrics=metrics,
    )
    boundary[1].run_turn.return_value.metrics = metrics
    call(boundary)
    assert boundary[1].run_turn.call_count == 1
    assert statuses[-1] == expected


def test_assistant_transcript_is_recorded_once(boundary, monkeypatch):
    from infinidev.engine.task_runtime import TaskRuntime
    original = TaskRuntime.append_chat
    messages = []
    def append(runtime, role, content, **metadata):
        messages.append((role, content))
        original(runtime, role, content, **metadata)
    monkeypatch.setattr(TaskRuntime, "append_chat", append)
    call(boundary)
    assert messages == [("user", "hola"), ("assistant", "respuesta")]


def test_cancellation_in_end_hook_preempts_final_reply(boundary, monkeypatch):
    from infinidev.engine.user_hooks import UserHookEvent
    def end_hook(event, **kwargs):
        if event == UserHookEvent.TASK_END_SUMMARY:
            boundary[1].is_cancelled = True
        return ""
    monkeypatch.setattr(pipeline, "_task_end_hook", end_hook)
    assert call(boundary) == "Conversation cancelled."
    boundary[2].notify.assert_called_once_with("Infinidev", "Conversation cancelled.", "agent")


def test_explicit_gather_is_context_only(boundary, monkeypatch):
    gather = Mock(return_value=("gathered evidence", ""))
    monkeypatch.setattr(pipeline, "_run_gather_phase", gather)
    call(boundary, "hola", force_gather=True)
    gather.assert_called_once()
    assert boundary[1].run_turn.call_args.args[1] == "hola"
    assert "gathered evidence" in boundary[1].run_turn.call_args.kwargs["turn_context"]


def test_runtime_exception_still_finalizes_conversation(boundary, monkeypatch):
    boundary[1].run_turn.side_effect = RuntimeError("provider failed")
    assert "failed" in call(boundary).lower()
    boundary[0].deactivate.assert_called_once()
    boundary[2].notify_stream_end.assert_called_once()
    boundary[2].mark_reply_shown.assert_called_once()


@pytest.mark.parametrize("hook_continuation", [False, True])
def test_real_runtime_edits_only_requested_test_file(boundary, monkeypatch, workspace_dir,
                                                    auto_approve_permissions, hook_continuation):
    import json
    from infinidev.db.service import register_session
    from infinidev.engine.formats.manual_tool_call import ManualToolCall
    from infinidev.engine.loop.engine import LoopEngine
    from infinidev.engine.turns import runtime as turn_runtime
    from infinidev.tools.base.context import clear_agent_context, set_context
    from infinidev.tools.file.edit_file_tool import EditFileTool
    from infinidev.tools.file.read_file_tool import ReadFileTool

    test_file = workspace_dir / "test_example.py"
    source_file = workspace_dir / "example.py"
    test_file.write_text("assert 1 == 2\n")
    source_file.write_text("VALUE = 1\n")
    agent, _, hooks = boundary
    agent.backstory = "Help the user."
    agent.role = "developer"
    agent.workspace_path = str(workspace_dir)
    agent.tools = [ReadFileTool(), EditFileTool()]
    agent.activate_context = lambda session_id: set_context(
        agent_id=agent.agent_id, project_id=None, session_id=session_id,
        workspace_path=str(workspace_dir),
    )
    agent.deactivate = lambda: clear_agent_context(agent.agent_id)
    register_session("principal-session")
    monkeypatch.setattr(turn_runtime, "get_litellm_params", lambda: {"model": "scripted"})
    monkeypatch.setattr(turn_runtime, "get_model_capabilities", lambda: SimpleNamespace(
        supports_function_calling=True, supports_vision=False,
    ))
    monkeypatch.setattr(turn_runtime, "_get_model_max_context", lambda params: 1_000_000)
    def response(text="", name=None, **arguments):
        calls = [ManualToolCall(id=name, name=name, arguments=json.dumps(arguments))] if name else None
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
            content=text, tool_calls=calls, reasoning_content=None,
        ))], usage=SimpleNamespace(prompt_tokens=100, completion_tokens=10))
    replies = [
        response(name="read_file", file_path="test_example.py"),
        response(name="edit_file", file_path="test_example.py", old_string="1 == 2",
                 new_string="1 == 1"),
        response(name="finish_turn", message="Updated the test only."),
    ]
    from infinidev.engine.user_hooks import UserHookEvent
    summaries = []
    def end_hook(event, **kwargs):
        if event == UserHookEvent.TASK_END_SUMMARY:
            summaries.append(kwargs["files_changed"])
        if event == UserHookEvent.TASK_END_INSTRUCTION and hook_continuation:
            return "Read the test once more."
        return ""
    monkeypatch.setattr(pipeline, "_task_end_hook", end_hook)
    if hook_continuation:
        replies.extend([response(name="read_file", file_path="test_example.py"),
                        response(name="finish_turn", message="Updated the test only.")])
    responses = iter(replies)
    seen = []
    def provider(params, messages, tools=None, **kwargs):
        seen.append((list(messages), tools))
        return next(responses)
    monkeypatch.setattr(turn_runtime, "call_llm", provider)
    engine = LoopEngine()
    result = call((agent, engine, hooks), "Change only the test; stop after editing.")
    assert result == "Updated the test only."
    assert test_file.read_text() == "assert 1 == 1\n"
    assert source_file.read_text() == "VALUE = 1\n"
    assert seen[0][0][-1]["content"] == "Change only the test; stop after editing."
    assert "previous conversation" in seen[0][0][-2]["content"]
    assert hooks.notify.call_count == (2 if hook_continuation else 1)
    assert all(item.args == ("Infinidev", result, "agent")
               for item in hooks.notify.call_args_list)

    assert summaries == [True]
    assert "-assert 1 == 2" in engine._last_file_tracker.get_diff(str(test_file))
    monkeypatch.setattr(pipeline, "_task_end_hook", lambda event, **kw:
                        summaries.append(kw["files_changed"])
                        if event == UserHookEvent.TASK_END_SUMMARY else "")
    responses = iter([response("Hello again.")])
    assert call((agent, engine, hooks), "Hi again") == "Hello again."
    assert summaries == [True, False]
    assert not engine.has_file_changes()


def test_failure_detail_is_displayed_once(boundary):
    message = 'Conversation interrupted: LLM stream exceeded safety limit'
    boundary[1].run_turn.return_value = TurnResult(
        message=message, status='failed', state=TurnState())
    assert call(boundary) == message
    hooks = boundary[2]
    hooks.notify.assert_called_once_with('Infinidev', message, 'agent')
    assert all(message not in str(item) for item in hooks.on_status.call_args_list)


def test_public_parts_and_terminal_reply_are_shown_and_persisted_once(boundary, monkeypatch):
    from infinidev.engine.task_runtime import TaskRuntime
    chats = []
    monkeypatch.setattr(TaskRuntime, 'append_chat', lambda self, role, text, **metadata: chats.append((role, text)))
    def run(*args, **kwargs):
        kwargs['on_public_message']('Concrete proposal details')
        return TurnResult(message='Summary of details above', status='answered', state=TurnState())
    boundary[1].run_turn.side_effect = run
    assert call(boundary) == 'Summary of details above'
    assert [c.args[1] for c in boundary[2].notify.call_args_list] == [
        'Concrete proposal details', 'Summary of details above']
    assert [text for role, text in chats if role == 'assistant'] == [
        'Concrete proposal details', 'Summary of details above']
    boundary[2].mark_reply_shown.assert_called_once()


def test_hook_reentry_publishes_prior_reply_before_it_is_referenced(boundary, monkeypatch):
    from infinidev.engine.task_runtime import TaskRuntime
    from infinidev.engine.user_hooks import UserHookEvent
    chats = []
    monkeypatch.setattr(TaskRuntime, 'append_chat', lambda self, role, text, **metadata: chats.append((role, text)))
    monkeypatch.setattr(pipeline, '_task_end_hook', lambda event, **kw:
        'Continue once' if event == UserHookEvent.TASK_END_INSTRUCTION else '')
    boundary[1].run_turn.side_effect = [
        TurnResult(message='First complete reply', status='answered', state=TurnState(),
                   metrics={'observed_iterations': 1, 'observed_tool_calls': 1}),
        TurnResult(message='Followup to the first reply', status='answered', state=TurnState())]
    assert call(boundary) == 'Followup to the first reply'
    assert [c.args[1] for c in boundary[2].notify.call_args_list] == [
        'First complete reply', 'Followup to the first reply']
    assert [text for role, text in chats if role == 'assistant'] == [
        'First complete reply', 'Followup to the first reply']


def test_published_details_survive_cancellation_with_one_interruption(boundary, monkeypatch):
    from infinidev.engine.task_runtime import TaskRuntime
    chats = []
    monkeypatch.setattr(TaskRuntime, 'append_chat', lambda self, role, text, **metadata: chats.append((role, text)))
    def run(*args, **kwargs):
        kwargs['on_public_message']('Already delivered details')
        boundary[1].is_cancelled = True
        return TurnResult(message='stale success', status='answered', state=TurnState())
    boundary[1].run_turn.side_effect = run
    assert call(boundary) == 'Conversation cancelled.'
    assert [c.args[1] for c in boundary[2].notify.call_args_list] == [
        'Already delivered details', 'Conversation cancelled.']
    assert [text for role, text in chats if role == 'assistant'] == [
        'Already delivered details', 'Conversation cancelled.']


@pytest.mark.parametrize('frontend_persists', [False, True])
def test_public_details_enter_both_histories_without_frontend_duplicates(
    boundary, monkeypatch, frontend_persists,
):
    from infinidev.db.service import register_session, store_conversation_turn, store_session_message
    from infinidev.tools.base.db import get_connection
    sid = 'principal-session'
    register_session(sid)
    hooks = boundary[2]
    hooks.persists_session_messages = frontend_persists
    if frontend_persists:
        hooks.notify.side_effect = lambda speaker, text, kind: store_session_message(
            sid, {'role': 'assistant', 'content': text})
    def run(*args, **kwargs):
        kwargs['on_public_message']('Complete public proposal details')
        return TurnResult(message='Summary referring above', status='answered', state=TurnState())
    boundary[1].run_turn.side_effect = run
    result = call(boundary)
    # Existing frontend ownership of the returned final reply remains unchanged.
    store_conversation_turn(sid, 'assistant', result, result[:200])
    if not frontend_persists:
        store_session_message(sid, {'role': 'assistant', 'content': result})
    with get_connection() as conn:
        public = [json.loads(row[0]) for row in conn.execute(
            'SELECT message_json FROM session_messages WHERE session_id=? ORDER BY id', (sid,))]
        context = [row[0] for row in conn.execute(
            "SELECT content FROM conversation_turns WHERE session_id=? AND role='assistant' "
            'ORDER BY id', (sid,))]
    assert [m['content'] for m in public] == [
        'Complete public proposal details', 'Summary referring above']
    assert context == ['Complete public proposal details', 'Summary referring above']
    if not frontend_persists:
        assert public[0]['turn_complete'] is False


def test_direct_pipeline_caller_can_publish_without_frontend_session_setup(boundary):
    from infinidev.db.service import delete_session, get_session_messages
    delete_session('principal-session')
    boundary[2].persists_session_messages = False
    def run(*args, **kwargs):
        kwargs['on_public_message']('Direct caller public details')
        return TurnResult(message='Done', status='answered', state=TurnState())
    boundary[1].run_turn.side_effect = run
    assert call(boundary) == 'Done'
    messages = get_session_messages('principal-session')
    assert any(m.get('content') == 'Direct caller public details' for m in messages)


def test_cancellation_in_end_instruction_hook_does_not_publish_superseded_reply(boundary, monkeypatch):
    from infinidev.engine.user_hooks import UserHookEvent
    def end_hook(event, **kwargs):
        if event == UserHookEvent.TASK_END_INSTRUCTION:
            boundary[1].is_cancelled = True
            return 'Continue once'
        return ''
    monkeypatch.setattr(pipeline, '_task_end_hook', end_hook)
    assert call(boundary) == 'Conversation cancelled.'
    assert [c.args[1] for c in boundary[2].notify.call_args_list] == ['Conversation cancelled.']
    assert boundary[1].run_turn.call_count == 1
