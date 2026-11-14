"""Provider-boundary coverage for agent-owned conversation turns."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from infinidev.engine.loop.engine import LoopEngine
from infinidev.engine.formats.manual_tool_call import ManualToolCall
from infinidev.tools.base.base_tool import BaseTool
from pydantic import BaseModel


class TextInput(BaseModel):
    text: str


class RecordTool(BaseTool):
    name: str = 'record'
    description: str = 'Record literal text.'
    args_schema: type[BaseModel] = TextInput
    def _run(self, text: str) -> str:
        return text


@pytest.fixture
def runtime(monkeypatch, tool_context, temp_db):
    from infinidev.engine.turns import runtime as rt
    from infinidev.db.service import register_session
    register_session(tool_context.session_id)
    from infinidev.engine.working_memory import WorkingMemory
    memory = WorkingMemory(tool_context.session_id, embed=False)
    monkeypatch.setattr('infinidev.engine.working_memory.get_working_memory', lambda sid: memory)
    monkeypatch.setattr(rt, 'get_litellm_params', lambda: {'model': 'scripted'})
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=True, supports_vision=False))
    monkeypatch.setattr(rt, '_get_model_max_context', lambda p: 1_000_000)
    engine = LoopEngine()
    agent = SimpleNamespace(agent_id=tool_context.agent_id, project_id=tool_context.project_id,
        backstory='You help the user.', role='developer', tools=[RecordTool()],
        workspace_path=tool_context.workspace_path)
    seen = []
    def script(*responses):
        replies = iter(responses)
        def call(params, messages, tools=None, **kwargs):
            seen.append((copy.deepcopy(messages), copy.deepcopy(tools), kwargs))
            response = next(replies)
            if callable(response):
                response = response()
            if isinstance(response, Exception):
                raise response
            return response
        monkeypatch.setattr(rt, 'call_llm', call)
    return engine, agent, script, seen


def reply(text='', *calls, tokens=100, finish_reason='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(
        content=text, tool_calls=list(calls) or None, reasoning_content=None))],
        usage=SimpleNamespace(prompt_tokens=tokens, completion_tokens=10))


def tc(name, **args):
    return ManualToolCall(id=name + str(len(str(args))), name=name, arguments=json.dumps(args))


def test_plain_answer_preserves_literal_input_and_separates_context(runtime):
    engine, agent, script, seen = runtime
    script(reply('Hello!'))
    result = engine.run_turn(agent, 'Hi\n literal', turn_context='Prior evidence')
    assert result.status == 'answered'
    assert result.message == 'Hello!'
    assert seen[0][0][-1]['content'] == 'Hi\n literal'
    assert 'Prior evidence' in seen[0][0][-2]['content']
    assert seen[0][2]['tool_choice'] == 'auto'


def test_tools_then_answer_and_full_repeated_catalog(runtime):
    engine, agent, script, seen = runtime
    script(*(reply('', tc('record', text='evidence')) for _ in range(5)), reply('', tc('finish_turn', message='Read it.')))
    result = engine.run_turn(agent, 'Investigate', max_total_tool_calls=6)
    assert result.status == 'answered'
    assert result.metrics['tool_calls'] == 6
    assert all('record' in [s['function']['name'] for s in tools] for _, tools, _ in seen)
    assert all(s['function']['name'] != 'step_complete' for s in seen[0][1])


@pytest.mark.parametrize('calls', [
    [tc('finish_turn', message='Done'), tc('record', text='must not run')],
    [tc('record', text='must not run'), tc('finish_turn', message='Done')],
])
def test_terminal_must_be_sole_call(runtime, calls):
    engine, agent, script, seen = runtime
    script(reply('', *calls))
    result = engine.run_turn(agent, 'Work')
    assert result.status == 'failed'
    assert result.metrics['tool_calls'] == 0
    assert not result.state.finished


def test_finish_turn(runtime):
    engine, agent, script, seen = runtime
    script(reply('', tc('finish_turn', message='Here is an intermediate result.')))
    result = engine.run_turn(agent, 'Work')
    assert result.status == 'answered'
    assert result.message == 'Here is an intermediate result.'


@pytest.mark.parametrize('text', ['<tool_call>{broken}</tool_call>', '{"tool_calls": [broken]}', ''])
def test_malformed_or_empty_manual_reply_is_not_success(runtime, monkeypatch, text):
    engine, agent, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    script(reply(text))
    assert engine.run_turn(agent, 'Work').status == 'failed'


def test_global_budget_stops_inside_batch(runtime):
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='first'), tc('record', text='second')))
    result = engine.run_turn(agent, 'Work', max_total_tool_calls=1)
    assert result.status == 'exhausted'
    assert result.metrics['tool_calls'] == 1


def test_iteration_budget(runtime):
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='evidence')))
    assert engine.run_turn(agent, 'Work', max_iterations=1).status == 'exhausted'


def test_cancellation_during_model_preempts_reply(runtime):
    engine, agent, script, seen = runtime
    def cancel():
        engine.cancel()
        return reply('Success')
    script(cancel)
    assert engine.run_turn(agent, 'Work').status == 'cancelled'


def test_manual_calls(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    from infinidev.engine.turns import runtime as rt
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=False, supports_vision=False))
    script(reply('<tool_call>{"name":"record","arguments":{"text":"hello"}}</tool_call>'),
        reply('', tc('finish_turn', message='Read.')))
    result = engine.run_turn(agent, 'Work')
    assert result.status == 'answered'
    assert result.metrics['tool_calls'] == 2
    assert seen[0][1] is None


def test_late_guidance_preempts_final(runtime):
    engine, agent, script, seen = runtime
    def late():
        engine.inject_message('Actually explain more')
        return reply('stale reply')
    script(late, reply('More explanation'))
    result = engine.run_turn(agent, 'Work')
    assert result.message == 'More explanation'
    assert seen[1][0][-1]['content'] == 'Actually explain more'


def test_actual_file_edit_without_forced_review(runtime, auto_approve_permissions):
    engine, agent, script, seen = runtime
    from pathlib import Path
    from infinidev.tools.file.edit_file_tool import EditFileTool
    target = Path(agent.workspace_path) / 'sample.txt'
    script(reply('', tc('edit_file', file_path=str(target), old_string='line one',
        new_string='changed')), reply('', tc('finish_turn', message='Intermediate delivery.')))
    result = engine.run_turn(agent, 'Change this test', task_tools=[EditFileTool()])
    assert result.status == 'answered'
    assert target.read_text().startswith('changed\n')
    assert engine._last_file_tracker.get_diff(str(target))
    assert len(seen) == 2


def test_native_malformed_arguments_do_not_execute(runtime):
    engine, agent, script, seen = runtime
    script(reply('', ManualToolCall(id='bad', name='record', arguments='{oops')),
           reply('', tc('finish_turn', message='Cannot proceed.')))
    result = engine.run_turn(agent, 'Work')
    assert result.status == 'answered'
    assert result.metrics['observed_tool_calls'] == 2
    assert json.loads(seen[1][0][-1]['content'])['executed'] is False


def test_schema_rejects_inferred_argument_repair(runtime):
    engine, agent, script, seen = runtime
    script(reply('', tc('record', message='invented alias')))
    assert engine.run_turn(agent, 'Work').status == 'failed'


def test_cancellation_during_tool_stops_remaining_batch(runtime):
    engine, agent, script, seen = runtime
    class CancelTool(RecordTool):
        def _run(self, text: str) -> str:
            engine.cancel()
            return 'partial evidence'
    agent.tools = [CancelTool()]
    script(reply('', tc('record', text='first'), tc('record', text='second')))
    result = engine.run_turn(agent, 'Work')
    assert result.status == 'cancelled'
    assert result.metrics['observed_tool_calls'] == 1
    assert any('partial evidence' in str(m['content']) for m in engine._last_turn_messages)


def test_persisted_plan_is_context_without_replay(runtime):
    engine, agent, script, seen = runtime
    script(reply('', tc('update_plan', plan=[dict(id='x', title='Opaque legacy obligation',
        status='pending')])), reply('', tc('finish_turn', message='Paused.')))
    assert engine.run_turn(agent, 'First request').status == 'answered'
    script(reply('Hello again.'))
    result = engine.run_turn(agent, 'Hi')
    assert result.status == 'answered'
    assert result.state.plan[0].title == 'Opaque legacy obligation'
    assert result.metrics['observed_tool_calls'] == 0
    assert seen[-1][0][-1]['content'] == 'Hi'
    assert 'prior_state_context' in seen[-1][0][-2]['content']


def test_actual_pressure_compacts_archived_results_and_preserves_input(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    from infinidev.engine.turns import runtime as rt
    monkeypatch.setattr(rt, '_get_model_max_context', lambda p: 1_000_000)
    evidence = 'important evidence\n' * 6000
    script(reply('', tc('record', text=evidence), tokens=900_000),
        reply('', tc('record', text='new evidence'), tokens=900_000), reply('', tc('finish_turn', message='Analyzed.')))
    result = engine.run_turn(agent, 'literal current input')
    assert result.status == 'answered'
    tool_messages = [m for m in seen[2][0] if m['role'] == 'tool']
    assert len(tool_messages[0]['content']) < len(evidence)
    assert 'Archived evidence' in tool_messages[0]['content']
    assert seen[2][0][1]['content'] == 'literal current input'
    assert result.metrics['observed_prompt_tokens'] == 1_800_100


def test_large_context_is_delivered_intact_without_pressure(runtime):
    engine, agent, script, seen = runtime
    text = 'Full context ' * 10000
    script(reply('Understood.'))
    assert engine.run_turn(agent, 'Hi', turn_context=text).status == 'answered'
    assert text in seen[0][0][1]['content']


def test_all_outcomes_report_observed_usage(runtime):
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='evidence'), tokens=321))
    result = engine.run_turn(agent, 'Work', max_iterations=1)
    assert result.metrics['observed_iterations'] == 1
    assert result.metrics['observed_tool_calls'] == 1
    assert result.metrics['observed_prompt_tokens'] == 321
    assert result.metrics['observed_completion_tokens'] == 10


def test_explicit_empty_tool_grant_does_not_restore_role_tools(runtime):
    engine, agent, script, seen = runtime
    script(reply('Hi'))
    assert engine.run_turn(agent, 'Hi', task_tools=[]).status == 'answered'
    assert {schema['function']['name'] for schema in seen[0][1]} == {'finish_turn', 'update_plan'}


def test_manual_prompt_uses_conversation_protocol(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    from infinidev.engine.turns import runtime as rt
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=False, supports_vision=False))
    script(reply('Hello'))
    assert engine.run_turn(agent, 'Hi').status == 'answered'
    assert 'step_complete' not in seen[0][0][0]['content']
    assert 'mark the step' not in seen[0][0][0]['content']


@pytest.mark.parametrize('text', ['<tool_call>{broken', '<function_call>{broken'])
def test_incomplete_manual_transport_never_becomes_final_answer(runtime, monkeypatch, text):
    engine, agent, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    script(reply(text))
    assert engine.run_turn(agent, 'Work').status == 'failed'


def test_project_instructions_survive_prompt_composition(runtime):
    engine, agent, script, seen = runtime
    from pathlib import Path
    (Path(agent.workspace_path) / 'AGENTS.md').write_text('Use the exact project convention ZORP.')
    script(reply('Hello'))
    assert engine.run_turn(agent, 'Hi').status == 'answered'
    assert 'ZORP' in seen[0][0][0]['content']


def test_archival_reference_survives_repeated_pressure(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='old evidence\n' * 2000), tokens=900_000),
        *(reply('', tc('record', text='new evidence'), tokens=900_000) for _ in range(4)),
        reply('', tc('finish_turn', message='Done')))
    assert engine.run_turn(agent, 'Work').status == 'answered'
    tool_messages = [m for m in seen[-1][0] if m['role'] == 'tool']
    assert 'Archived evidence' in tool_messages[0]['content']


def test_provider_never_receives_internal_transcript_metadata(runtime):
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='data')), reply('', tc('finish_turn', message='Done')))
    assert engine.run_turn(agent, 'Work').status == 'answered'
    assert all(not key.startswith('_') for message in seen[-1][0] for key in message)


def test_manual_results_compact_under_pressure(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    from infinidev.engine.turns import runtime as rt
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=False, supports_vision=False))
    script(reply('<tool_call>{"name":"record","arguments":{"text":"' + 'a' * 4000
        + '"}}</tool_call>', tokens=900_000),
        reply('<tool_call>{"name":"record","arguments":{"text":"new"}}</tool_call>',
            tokens=900_000), reply('', tc('finish_turn', message='Done')))
    assert engine.run_turn(agent, 'literal').status == 'answered'
    result_messages = [m for m in seen[-1][0] if '[Tool: record]' in str(m.get('content'))]
    assert len(result_messages[0]['content']) < 1500
    assert seen[-1][0][1]['content'] == 'literal'


def test_answer_can_explain_tool_protocol_as_prose(runtime):
    engine, agent, script, seen = runtime
    script(reply('The "arguments" field contains the values you supply.'))
    result = engine.run_turn(agent, 'Explain arguments')
    assert result.status == 'answered'


def test_host_permission_denial_preserves_file(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    from pathlib import Path
    from infinidev.config.settings import settings
    from infinidev.tools.file.edit_file_tool import EditFileTool
    monkeypatch.setattr(settings, 'FILE_OPERATIONS_PERMISSION', 'deny')
    target = Path(agent.workspace_path) / 'sample.txt'
    before = target.read_text()
    script(reply('', tc('edit_file', file_path=str(target), old_string='line one',
        new_string='changed')), reply('', tc('finish_turn', message='Permission was denied.')))
    result = engine.run_turn(agent, 'Edit', task_tools=[EditFileTool()])
    assert result.status == 'answered'
    assert target.read_text() == before
    assert 'denied' in str(seen[-1][0]).lower()


def test_post_model_hook_receives_literal_request_and_turn_plan(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    from infinidev.engine.hooks.hooks import HookEvent, hook_manager
    events = []
    monkeypatch.setattr(hook_manager, 'dispatch', lambda context: events.append(context))
    script(reply('Hello'))
    result = engine.run_turn(agent, 'literal input', turn_context='prior context')
    assert result.status == 'answered'
    model_events = [event for event in events if event.event == HookEvent.POST_MODEL_MESSAGE]
    assert len(model_events) == 1
    assert model_events[0].metadata['task'] == 'literal input'
    assert model_events[0].metadata['raw_content'] == 'Hello'


def test_archive_references_survive_session_continuation(runtime):
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='Durable evidence ' * 10)), reply('', tc('finish_turn', message='Paused')))
    assert engine.run_turn(agent, 'Investigate').status == 'answered'
    script(reply('Hello'))
    assert engine.run_turn(agent, 'Hi').status == 'answered'
    assert 'Archived evidence' in str(seen[-1][0])


@pytest.mark.parametrize('manual', [False, True])
def test_provider_reasoning_and_signatures_survive_tool_roundtrip(runtime, monkeypatch, manual):
    engine, agent, script, seen = runtime
    from infinidev.engine.turns import runtime as rt
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=not manual, supports_vision=False))
    first = (reply('<tool_call>{"name":"record","arguments":{"text":"data"}}</tool_call>')
             if manual else reply('', tc('record', text='data')))
    fields = {
        'reasoning_content': 'Visible provider reasoning',
        'thinking_blocks': [{'type': 'thinking', 'thinking': 'text', 'signature': 'opaque'}],
        'reasoning_details': [{'type': 'reasoning.encrypted', 'encrypted_content': 'secret'}],
        'provider_specific_fields': {'thought_signatures': ['signature'],
                                     'unrelated_provider_metadata': 'omit'},
    }
    for key, value in fields.items():
        setattr(first.choices[0].message, key, value)
    script(first, reply('', tc('finish_turn', message='Done')))
    assert engine.run_turn(agent, 'Work').status == 'answered'
    assistant = next(m for m in seen[-1][0] if m['role'] == 'assistant')
    assert assistant['reasoning_content'] == fields['reasoning_content']
    assert assistant['thinking_blocks'] == fields['thinking_blocks']
    assert assistant['reasoning_details'] == fields['reasoning_details']
    assert assistant['provider_specific_fields'] == {'thought_signatures': ['signature']}


@pytest.mark.parametrize('mode', ['prose', 'tool', 'manual'])
def test_provider_length_limit_interrupts_without_executing_truncated_calls(
        runtime, monkeypatch, mode):
    engine, agent, script, seen = runtime
    if mode == 'manual':
        from infinidev.engine.turns import runtime as rt
        monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
            supports_function_calling=False, supports_vision=False))
    calls = [tc('record', text='must not execute')] if mode == 'tool' else []
    text = ('<tool_call>{"name":"record","arguments":{"text":"must not execute"}}</tool_call>'
            if mode == 'manual' else 'Incomplete answer')
    truncated = reply(text, *calls, tokens=321)
    truncated.choices[0].finish_reason = 'length'
    script(reply('', tc('record', text='prior evidence')), truncated)
    result = engine.run_turn(agent, 'Work')
    assert result.status == 'exhausted'
    assert result.metrics['observed_iterations'] == 2
    assert result.metrics['observed_tool_calls'] == 1
    assert result.metrics['observed_prompt_tokens'] == 421
    assert result.metrics['observed_completion_tokens'] == 20
    assert any('prior evidence' in str(m.get('content')) for m in engine._last_turn_messages)
    assert 'Archived evidence' in str(engine._last_turn_messages)


def test_actual_edit_emits_file_change_callback(runtime, auto_approve_permissions):
    engine, agent, script, seen = runtime
    from pathlib import Path
    from infinidev.tools.file.edit_file_tool import EditFileTool
    target = Path(agent.workspace_path) / 'sample.txt'
    changed = []
    engine._hooks = SimpleNamespace(on_file_change=changed.append)
    script(reply('', tc('edit_file', file_path=str(target), old_string='line one',
                           new_string='changed')), reply('', tc('finish_turn', message='Done')))
    result = engine.run_turn(agent, 'Edit', task_tools=[EditFileTool()])
    assert result.status == 'answered'
    assert target.read_text().startswith('changed\n')
    assert changed == [str(target)]


@pytest.mark.parametrize('manual', [False, True])
def test_invalid_schema_returns_tool_feedback_and_agent_repairs(runtime, monkeypatch, manual):
    from infinidev.tools.shell.execute_command_input import ExecuteCommandInput

    class CommandTool(RecordTool):
        name: str = 'execute_command'
        args_schema: type[BaseModel] = ExecuteCommandInput

        def _run(self, command, rationale='', timeout=120, cwd=None, env=None):
            assert env is None
            return 'executed once'

    engine, agent, script, seen = runtime
    if manual:
        monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
            lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    script(reply('', tc('execute_command', command='true', env='null')),
           reply('', tc('execute_command', command='true', env=None)), reply('', tc('finish_turn', message='Done')))
    result = engine.run_turn(agent, 'Work', task_tools=[CommandTool()])
    assert result.status == 'answered'
    content = seen[1][0][-1]['content']
    feedback = json.loads(content.split('Result:\n', 1)[-1])
    assert feedback['executed'] is False
    assert feedback['errors'][0]['loc'] == ['env']
    assert 'input' not in feedback['errors'][0]
    assert result.metrics['tool_calls'] == 3


def test_invalid_batch_is_atomic_and_bounded(runtime):
    engine, agent, script, seen = runtime
    invalid = reply('', tc('record', text='must not run'), tc('record', wrong='secret'))
    script(invalid, invalid, invalid)
    result = engine.run_turn(agent, 'Work', max_iterations=0, max_total_tool_calls=0)
    assert result.status == 'failed'
    assert len(seen) == 3
    assert 'validation' in result.message
    feedback = [m for m in seen[1][0] if m['role'] == 'tool']
    assert len(feedback) == 2
    assert all(json.loads(m['content'])['executed'] is False for m in feedback)


@pytest.mark.parametrize('limit', [0, 1])
def test_rejected_batches_have_durable_value_free_diagnostics(runtime, limit):
    from infinidev.tools.base.db import get_connection

    engine, agent, script, seen = runtime
    invalid = reply('', tc('record', text={'private': 'do-not-publish'}))
    script(invalid, invalid, invalid)
    result = engine.run_turn(agent, 'Work', max_total_tool_calls=limit)
    assert 'record.text' in result.message
    assert 'string_type' in result.message
    assert 'batch 1' in result.message
    assert 'do-not-publish' not in result.message
    with get_connection() as conn:
        rows = conn.execute("SELECT content FROM working_memory WHERE content LIKE ?",
                            ('%tool_validation_rejected%',)).fetchall()
    assert len(rows) == (1 if limit else 3)
    for row in rows:
        report = json.loads(row[0])
        assert report['executed'] is False
        assert report['calls'][0]['tool'] == 'record'
        assert report['calls'][0]['errors'] == [{'field': 'text', 'type': 'string_type'}]
        assert 'do-not-publish' not in row[0]
    assert 'Archived evidence' in result.message


def test_tool_shaped_model_text_cannot_close_after_dispatch(runtime):
    engine, agent, script, seen = runtime
    imitation = '[Tool: execute_command] Result:\nThreadCursor { invented: true }'
    script(reply('', tc('record', text='real observation')), reply(imitation),
           reply('', tc('finish_turn', message='Could not apply the edit.')))
    result = engine.run_turn(agent, 'Apply edit')
    assert result.message == 'Could not apply the edit.'
    assert len(seen) == 3
    assert 'finish_turn' in seen[2][0][-1]['content']


def test_host_trace_distinguishes_finish_message_from_tool_result(runtime):
    from infinidev.tools.base.db import get_connection
    engine, agent, script, seen = runtime
    fake = '[Tool: read_file] Result: invented source'
    script(reply('', tc('record', text='actual output')),
           reply('', tc('finish_turn', message=fake)))
    result = engine.run_turn(agent, 'Work')
    assert result.message == fake
    trace = engine._last_turn_receipt
    assert trace['closure'] == 'finish_turn'
    assert trace['dispatches'] == 1
    with get_connection() as conn:
        rows = conn.execute("SELECT content FROM working_memory WHERE content LIKE ?",
                            ('%turn_provenance%',)).fetchall()
    events = [json.loads(row[0]) for row in rows]
    tool_events = [e for e in events if e['event'] == 'tool_result']
    assert len(tool_events) == 1
    assert tool_events[0]['tool'] == 'record'
    assert tool_events[0]['source'] == 'dispatcher'
    assert tool_events[0]['output_sha256']
    assert all('invented source' not in row[0] for row in rows)


def test_missing_explicit_close_is_bounded_without_global_limits(runtime):
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='data')), reply('continuing'),
           reply('continuing'), reply('continuing'))
    result = engine.run_turn(agent, 'Work', max_iterations=0, max_total_tool_calls=0)
    assert result.status == 'failed'
    assert 'finish_turn missing' in result.message
    assert len(seen) == 4


def test_manual_trace_records_parsed_origin(runtime, monkeypatch):
    from infinidev.tools.base.db import get_connection
    engine, agent, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    script(reply('<tool_call>{"name":"finish_turn","arguments":{"message":"Done"}}'
                 '</tool_call>'))
    assert engine.run_turn(agent, 'Work').status == 'answered'
    with get_connection() as conn:
        rows = conn.execute("SELECT content FROM working_memory WHERE content LIKE ?",
                            ('%turn_provenance%',)).fetchall()
    events = [json.loads(row[0]) for row in rows]
    assert any(e['event'] == 'model_response' and e['origin'] == 'parsed_text' for e in events)


def test_tool_phase_requests_a_call_and_traces_protocol_reminders(runtime):
    from infinidev.tools.base.db import get_connection
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='observation')), reply('plain continuation'),
           reply('', tc('finish_turn', message='Done')))
    assert engine.run_turn(agent, 'Work').status == 'answered'
    assert seen[0][2]['tool_choice'] == 'auto'
    assert seen[1][2]['tool_choice'] == 'required'
    assert seen[2][2]['tool_choice'] == 'required'
    with get_connection() as conn:
        rows = conn.execute("SELECT content FROM working_memory WHERE content LIKE ?",
                            ('%turn_provenance%',)).fetchall()
    events = [json.loads(row[0]) for row in rows]
    reminders = [e for e in events if e['event'] == 'protocol_reminder']
    assert len(reminders) == 1
    assert reminders[0]['number'] == 1
    requests = [e for e in events if e['event'] == 'runtime_request']
    assert requests[1]['finish_turn_offered'] is True
    assert requests[1]['requested_tool_choice'] == 'required'


@pytest.mark.parametrize('invalid', [False, True])
def test_manual_history_preserves_action_before_host_observation(runtime, monkeypatch, invalid):
    engine, agent, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    arguments = {'wrong': 'value'} if invalid else {'text': 'value'}
    first = '<tool_call>' + json.dumps({'name': 'record', 'arguments': arguments}) + '</tool_call>'
    script(reply(first), reply('', tc('finish_turn', message='Stopped')))
    assert engine.run_turn(agent, 'Work').status == 'answered'
    transcript = seen[1][0]
    observation = len(transcript) - 1
    assert transcript[observation]['role'] == 'user'
    assert observation > 0
    action = transcript[observation - 1]
    assert action['role'] == 'assistant'
    assert action['content']
    parsed = json.loads(action['content'])
    assert parsed['tool_calls'][0]['name'] == 'record'
    assert parsed['tool_calls'][0]['arguments'] == arguments


def test_bound_workspace_without_agent_attribute_tracks_external_edits(runtime, monkeypatch):
    from pathlib import Path
    from infinidev.engine.turns import runtime as rt
    engine, agent, script, seen = runtime
    workspace = agent.workspace_path
    target = Path(workspace) / 'sample.txt'
    del agent.workspace_path
    rendered = []
    monkeypatch.setattr(rt, 'render_project_instructions',
                        lambda root: rendered.append(root) or 'Project instructions')

    class ExternalWriter(RecordTool):
        def _run(self, text):
            target.write_text(text)
            return 'written externally'

    script(reply('', tc('record', text='changed externally')),
           reply('', tc('finish_turn', message='Changed')))
    result = engine.run_turn(agent, 'Write', task_tools=[ExternalWriter()])
    assert result.status == 'answered'
    assert target.read_text() == 'changed externally'
    assert engine.has_file_changes()
    assert engine._last_file_tracker.baseline.root == str(Path(workspace).resolve())
    assert rendered == [workspace]


@pytest.mark.parametrize('manual', [False, True])
def test_examples_and_rejected_reasoning_never_become_actions(runtime, monkeypatch, manual):
    engine, agent, script, seen = runtime
    if manual:
        monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
            lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    example = 'Here is an example; do not run it:\n```json\n' + json.dumps({
        'tool_calls': [{'name': 'record', 'arguments': {'text': 'must not execute'}}]}) + '\n```'
    response = reply(example)
    response.choices[0].message.reasoning_content = json.dumps({
        'tool_calls': [{'name': 'record', 'arguments': {'text': 'rejected candidate'}}]})
    script(response)
    result = engine.run_turn(agent, 'Explain this protocol')
    assert result.metrics['tool_calls'] == 0
    if not manual:
        assert result.status == 'answered'
        assert result.message == example


def test_native_reasoning_candidate_does_not_override_final_answer(runtime):
    engine, agent, script, seen = runtime
    response = reply('I would not execute that command.')
    response.choices[0].message.reasoning_content = json.dumps({
        'tool_calls': [{'name': 'finish_turn', 'arguments': {'message': 'Wrong example'}}]})
    script(response)
    result = engine.run_turn(agent, 'Explain')
    assert result.message == 'I would not execute that command.'
    assert result.metrics['tool_calls'] == 0


def test_manual_conversation_does_not_force_json_output(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    script(reply('Hello'))
    assert engine.run_turn(agent, 'Hi').message == 'Hello'
    assert seen[0][2]['use_json_mode'] is False


def test_provenance_does_not_evict_actual_evidence_references(runtime):
    from infinidev.db.service import get_session_runtime_state
    from infinidev.tools.base.context import get_context_for_agent
    engine, agent, script, seen = runtime
    script(*(reply('', tc('record', text=f'Unique actual evidence {i}')) for i in range(12)),
           reply('', tc('finish_turn', message='Done')))
    assert engine.run_turn(agent, 'Work').status == 'answered'
    state = get_session_runtime_state(get_context_for_agent(agent.agent_id).session_id)
    refs = state['ui_state']['conversation']['evidence']
    assert len(refs) >= 12
    assert not any('turn_provenance_' in ref for ref in refs)


def test_archive_references_distinguish_consecutive_turns(runtime):
    from infinidev.db.service import get_session_runtime_state
    from infinidev.tools.base.context import get_context_for_agent
    engine, agent, script, seen = runtime
    sid = get_context_for_agent(agent.agent_id).session_id
    script(reply('', tc('record', text='First content')), reply('', tc('finish_turn', message='One')))
    engine.run_turn(agent, 'First')
    first = get_session_runtime_state(sid)['ui_state']['conversation']['evidence']
    script(reply('', tc('record', text='Second content')), reply('', tc('finish_turn', message='Two')))
    engine.run_turn(agent, 'Second')
    second = get_session_runtime_state(sid)['ui_state']['conversation']['evidence']
    assert len(second) == len(set(second))
    assert set(first) < set(second)


def test_manual_json_capable_backend_keeps_plain_answer_transport(runtime, monkeypatch):
    from infinidev.config import model_capabilities as mc
    from infinidev.engine import llm_client
    from infinidev.engine.turns import runtime as rt
    engine, agent, script, seen = runtime
    caps = mc.ModelCapabilities(supports_function_calling=False, supports_json_mode=True,
                                supports_tool_choice_required=False, probed=True)
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: caps)
    monkeypatch.setattr(mc, 'get_model_capabilities', lambda: caps)
    monkeypatch.setattr(rt, 'call_llm', llm_client.call_llm)
    requests = []
    def completion(litellm, kwargs, **callbacks):
        requests.append(kwargs)
        return reply('Hello in prose')
    monkeypatch.setattr(llm_client, '_stream_and_assemble', completion)
    assert engine.run_turn(agent, 'Hi').message == 'Hello in prose'
    assert 'response_format' not in requests[0]
    assert not requests[0].get('tools')


def test_manual_multiple_full_envelopes_preserve_argument_literals(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    calls = ''.join('<tool_call>' + json.dumps({'name': 'record', 'arguments': {'text': text}})
                    + '</tool_call>\n' for text in ['literal </tool_call> text', 'second'])
    script(reply(calls), reply('', tc('finish_turn', message='Done')))
    assert engine.run_turn(agent, 'Work').status == 'answered'
    results = [m['content'] for m in seen[1][0] if m['role'] == 'user']
    assert any('literal </tool_call> text' in text for text in results)
    assert engine._last_total_tool_calls == 3


@pytest.mark.parametrize('frame', ['json', 'qwen'])
def test_manual_action_arguments_are_opaque_to_thinking_markup(runtime, monkeypatch, frame):
    engine, agent, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    literal = 'literal <think>example</think> content'
    action = {'name': 'record', 'arguments': {'text': literal}}
    content = (json.dumps({'tool_calls': [action]}) if frame == 'json' else
               '<tool_call>' + json.dumps(action) + '</tool_call>')
    script(reply(content), reply('', tc('finish_turn', message='Done')))
    assert engine.run_turn(agent, 'Preserve literal').status == 'answered'
    action_history = next(m for m in seen[1][0] if m['role'] == 'assistant')
    assert json.loads(action_history['content'])['tool_calls'][0]['arguments']['text'] == literal
    assert any(literal in m.get('content', '') for m in seen[1][0] if m['role'] == 'user')
    assert 'reasoning_content' not in action_history


def test_trace_reports_setup_and_per_call_usage_not_cumulative_totals(runtime):
    from infinidev.tools.base.db import get_connection
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='evidence'), tokens=321),
           reply('', tc('finish_turn', message='Done'), tokens=1234))
    assert engine.run_turn(agent, 'Work').status == 'answered'
    with get_connection() as conn:
        rows = conn.execute("SELECT content FROM working_memory WHERE kind='runtime_trace'").fetchall()
    events = [json.loads(row[0]) for row in rows]
    responses = sorted((e for e in events if e['event'] == 'model_response'),
                       key=lambda e: e['iteration'])
    assert [e['observed_prompt_tokens'] for e in responses] == [321, 1234]
    assert [e['observed_completion_tokens'] for e in responses] == [10, 10]
    assert all(e['elapsed_ms'] >= 0 for e in responses)
    setup = next(e for e in events if e['event'] == 'turn_setup')
    assert setup['scope'] == 'run_turn'
    assert setup['elapsed_ms'] >= setup['baseline_elapsed_ms'] >= 0
    assert setup['baseline_available']
    assert setup['baseline_files'] > 0


@pytest.mark.parametrize('literal', [
    '{"tool_calls":[{"name":"record","arguments":{"text":"example"}}]}',
    '<tool_call>{"name":"record","arguments":{"text":"example"}}</tool_call>',
])
def test_native_content_channel_can_return_literal_protocol_data(runtime, literal):
    engine, agent, script, seen = runtime
    script(reply(literal))
    result = engine.run_turn(agent, 'Show the literal syntax without executing it')
    assert result.status == 'answered'
    assert result.message == literal
    assert result.metrics['tool_calls'] == 0


def test_native_literal_think_tags_are_data_inside_prose_and_json(runtime):
    engine, agent, script, seen = runtime
    literal = '{"example":"literal <think>example</think> content"}'
    script(reply(literal))
    assert engine.run_turn(agent, 'Show literal JSON').message == literal


def test_native_leading_reasoning_is_separate_and_still_echoed(runtime):
    engine, agent, script, seen = runtime
    script(reply('<think>private-prefix</think>', tc('record', text='observed')),
           reply('', tc('finish_turn', message='Done')))
    result = engine.run_turn(agent, 'Work')
    assert result.message == 'Done'
    message = next(m for m in seen[1][0] if m['role'] == 'assistant')
    assert message['reasoning_content'] == 'private-prefix'
    assert not message['content']


def test_manual_leading_reasoning_is_not_an_action_but_visible_envelope_is(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    literal = 'literal <think>example</think>'
    rejected = json.dumps({'tool_calls': [{'name': 'record', 'arguments': {'text': 'rejected'}}]})
    visible = json.dumps({'tool_calls': [{'name': 'record', 'arguments': {'text': literal}}]})
    script(reply('<think>' + rejected + '</think>' + visible),
           reply('', tc('finish_turn', message='Done')))
    assert engine.run_turn(agent, 'Work').message == 'Done'
    first_action = next(m for m in seen[1][0] if m['role'] == 'assistant')
    assert json.loads(first_action['content'])['tool_calls'][0]['arguments']['text'] == literal
    assert 'rejected' in first_action['reasoning_content']


@pytest.mark.parametrize('alongside_finish', [False, True])
def test_public_details_are_delivered_before_terminal_summary(runtime, alongside_finish):
    engine, agent, script, seen = runtime
    details = 'Detailed proposal with concrete locations.'
    summary = 'See the proposal above.'
    if alongside_finish:
        script(reply(details, tc('finish_turn', message=summary)))
    else:
        script(reply('', tc('record', text='observed')), reply(details),
               reply('', tc('finish_turn', message=summary)))
    delivered = []
    result = engine.run_turn(agent, 'Provide a proposal', on_public_message=delivered.append)
    assert result.message == summary
    assert delivered == [details]
    assert result.public_messages == [details]


def test_public_delivery_excludes_reasoning_and_manual_action_envelopes(runtime, monkeypatch):
    engine, agent, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=False, supports_vision=False))
    response = reply('<tool_call>{"name":"record","arguments":{"text":"data"}}</tool_call>')
    response.choices[0].message.reasoning_content = 'private deliberation'
    script(response, reply('', tc('finish_turn', message='Done')))
    delivered = []
    result = engine.run_turn(agent, 'Work', on_public_message=delivered.append)
    assert result.message == 'Done'
    assert delivered == result.public_messages == []


def test_cancellation_during_provider_call_does_not_publish_stale_body(runtime):
    engine, agent, script, seen = runtime
    def cancel():
        engine.cancel()
        return reply('stale public body', tc('finish_turn', message='stale close'))
    script(cancel)
    delivered = []
    result = engine.run_turn(agent, 'Work', on_public_message=delivered.append)
    assert result.status == 'cancelled'
    assert delivered == []


def test_published_progress_survives_exhaustion_and_callback_cancellation(runtime):
    engine, agent, script, seen = runtime
    script(reply('Public progress', tc('record', text='data')))
    delivered = []
    result = engine.run_turn(agent, 'Work', max_total_tool_calls=1,
                             on_public_message=delivered.append)
    assert result.status == 'exhausted'
    assert delivered == result.public_messages == ['Public progress']

    engine.begin_turn()
    script(reply('Visible before cancellation', tc('record', text='must not run')))
    def publish_and_cancel(text):
        delivered.append(text)
        engine.cancel()
    result = engine.run_turn(agent, 'Work', on_public_message=publish_and_cancel)
    assert result.status == 'cancelled'
    assert result.public_messages == ['Visible before cancellation']
    assert result.metrics['tool_calls'] == 0


def test_pending_guidance_defers_native_close_and_its_public_body(runtime):
    engine, agent, script, seen = runtime
    def update():
        engine.inject_message('Use the newer request')
        return reply('stale details', tc('finish_turn', message='stale final'))
    script(update, reply('Updated answer'))
    delivered = []
    result = engine.run_turn(agent, 'Work', on_public_message=delivered.append)
    assert result.message == 'Updated answer'
    assert delivered == result.public_messages == []


def test_guidance_after_publication_preserves_accurate_visibility(runtime):
    engine, agent, script, seen = runtime
    script(reply('Already visible details', tc('finish_turn', message='Old close')),
           reply('Updated answer'))
    delivered = []
    def publish_and_update(text):
        delivered.append(text)
        engine.inject_message('Updated request')
    result = engine.run_turn(agent, 'Work', on_public_message=publish_and_update)
    assert delivered == ['Already visible details']
    assert result.message == 'Updated answer'
    feedback = [m['content'] for m in seen[1][0] if m['role'] == 'tool']
    assert feedback
    assert all('not published' not in text for text in feedback)
