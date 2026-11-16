"""Generated memory guidance follows the same grant as provider schemas."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from infinidev.engine.session_context import render_session_context
from tests.test_principal_pipeline import boundary, call
from tests.test_turn_runtime import RecordTool, reply, runtime, tc


@pytest.mark.parametrize('manual', [False, True])
def test_principal_generated_context_matches_read_create_grant(
        boundary, monkeypatch, workspace_dir, manual):
    from infinidev.agents.base import InfinidevAgent
    from infinidev.db.service import store_conversation_turn
    from infinidev.engine.loop.engine import LoopEngine
    from infinidev.engine.orchestration import pipeline
    from infinidev.engine.turns import runtime as rt
    from infinidev.prompts.profiles import EffectivePromptConfiguration
    from infinidev.tools.base.context import clear_agent_context, set_context
    from infinidev.tools.file.create_file_tool import CreateFileTool
    from infinidev.tools.file.read_file_tool import ReadFileTool

    original, _, hooks = boundary
    # Real agent defaults, with tool discovery replaced rather than starting MCP.
    tools = [ReadFileTool(), CreateFileTool()]
    def resolve(self):
        self._tools = tools
        self._mcp_generation = 0
    monkeypatch.setattr(InfinidevAgent, '_resolve_tools', resolve)
    monkeypatch.setattr('infinidev.agents.base._mcp_generation', lambda: 0)
    agent = InfinidevAgent(agent_id=original.agent_id)
    agent.activate_context = lambda session_id: set_context(
        agent_id=agent.agent_id, project_id=None, session_id=session_id,
        workspace_path=str(workspace_dir))
    agent.deactivate = lambda: clear_agent_context(agent.agent_id)
    monkeypatch.setattr('infinidev.engine.session_context.render_session_context',
                        render_session_context)
    monkeypatch.setattr(pipeline, '_ken_turn_context', lambda *a: '')
    monkeypatch.setattr(pipeline, '_run_task_start_hook', lambda **kw: '')
    monkeypatch.setattr(rt, 'get_litellm_params', lambda: {'model': 'scripted'})
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=not manual, supports_vision=False))
    monkeypatch.setattr(rt, '_get_model_max_context', lambda _: 1_000_000)
    monkeypatch.setattr(rt.EffectivePromptConfiguration, 'compile',
                        lambda: EffectivePromptConfiguration({}))
    store_conversation_turn('principal-session', 'assistant', 'Previous decision.', 'Previous')
    seen = []
    def provider(params, messages, schemas=None, **kwargs):
        seen.append((copy.deepcopy(messages), copy.deepcopy(schemas)))
        return reply('Synthetic answer.')
    monkeypatch.setattr(rt, 'call_llm', provider)
    call((agent, LoopEngine(), hooks), 'Read the brief and create a new file.', task_tools=tools)
    messages, schemas = seen[0]
    assert messages[0]['role'] == 'system'
    assert all(m['role'] != 'developer' for m in messages)
    generated = '\n'.join(m['content'] for m in messages[:-1])
    assert 'Previous decision.' in generated
    assert 'chat_history_search' not in generated
    assert 'edit_file' not in generated
    assert 'step_complete' not in generated
    expected = {'read_file', 'create_file', 'finish_turn', 'update_plan'}
    if manual:
        schema_text = messages[0]['content'].split('Available tools and their JSON schemas:\n')[1]
        schemas, _ = json.JSONDecoder().raw_decode(schema_text)
    assert {s['function']['name'] for s in schemas} == expected


@pytest.mark.parametrize('names,expected', [
    (None, True), ((), False), (('read_file', 'create_file'), False),
    (('chat_history_search',), True),
])
def test_history_guidance_uses_explicit_catalog(monkeypatch, names, expected):
    monkeypatch.setattr('infinidev.db.service.get_recent_turns_full',
                        lambda *a, **kw: [('assistant', 'Prior decision.')])
    kwargs = {} if names is None else {'tool_names': names}
    text = render_session_context('synthetic-session', 'Current request', **kwargs)
    assert 'Prior decision.' in text
    assert ('chat_history_search' in text) is expected


@pytest.mark.parametrize('manual', [False, True])
@pytest.mark.parametrize('rejected', [False, True])
def test_result_and_rejection_references_keep_evidence_without_unoffered_guidance(
        runtime, monkeypatch, tool_context, manual, rejected):
    from infinidev.engine.turns import runtime as rt
    engine, agent, script, seen = runtime
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=not manual, supports_vision=False))
    name = 'unoffered' if rejected else 'record'
    first = (reply(json.dumps({'tool_calls': [{'name': name, 'arguments': {'text': 'evidence'}}]}))
             if manual else reply('', tc(name, text='evidence')))
    final = (reply(json.dumps({'tool_calls': [{'name': 'finish_turn',
                                              'arguments': {'message': 'Stopped.'}}]}))
             if manual else reply('', tc('finish_turn', message='Stopped.')))
    script(first, final)
    result = engine.run_turn(agent, 'Synthetic request', task_tools=agent.tools)
    assert result.status == 'answered'
    messages = seen[-1][0]
    _, references = rt._load_state(tool_context.session_id)
    assert references and all('Archived evidence' in ref for ref in references)
    assert all('recall_context' not in ref for ref in references)
    assert 'recall_context' not in json.dumps(messages)
    assert engine._last_turn_receipt['dispatches'] == (0 if rejected else 1)


def test_old_host_reference_projects_for_grant_without_mutating_original(runtime, monkeypatch):
    from infinidev.engine.turns import runtime as rt
    from infinidev.engine.turns.models import TurnState
    engine, agent, script, seen = runtime
    references = ['[Archived evidence; recall_context reference: wm-archive:abc]',
                  '[Archived evidence; recall_context query: exact old title]']
    monkeypatch.setattr(rt, '_load_state', lambda _: (TurnState(), references))
    literal = 'User quote: recall_context and edit_file remain literal user text.'
    script(reply('Synthetic answer.'))
    engine.run_turn(agent, literal, task_tools=agent.tools)
    prior = next(m['content'] for m in seen[0][0] if '<prior_state_context>' in m['content'])
    assert 'wm-archive:abc' in prior and 'exact old title' in prior
    assert 'recall_context' not in prior
    assert references[0] == '[Archived evidence; recall_context reference: wm-archive:abc]'
    assert references[1] == '[Archived evidence; recall_context query: exact old title]'
    assert seen[0][0][-1]['content'] == literal


def test_default_catalog_still_recommends_available_recall(runtime):
    engine, agent, script, seen = runtime
    class Recall(RecordTool):
        name: str = 'recall_context'
    agent.tools.append(Recall())
    script(reply('', tc('record', text='evidence')), reply('', tc('finish_turn', message='Done.')))
    engine.run_turn(agent, 'Synthetic request')
    assert 'recall_context reference:' in seen[-1][0][-1]['content']


def test_cancelled_turn_retains_neutral_archive_for_next_request(runtime):
    engine, agent, script, seen = runtime
    def cancel_after_result():
        engine.cancel()
        return reply('', tc('finish_turn', message='Cancelled.'))
    script(reply('', tc('record', text='evidence')), cancel_after_result)
    result = engine.run_turn(agent, 'Synthetic request', task_tools=agent.tools)
    assert result.status == 'cancelled'
    engine.begin_turn()
    script(reply('Resumed.'))
    engine.run_turn(agent, 'Continue', task_tools=agent.tools)
    request = json.dumps(seen[-1][0])
    assert 'Archived evidence' in request
    assert 'recall_context' not in request


def test_history_preserves_literal_names_in_human_content(monkeypatch):
    literal = 'Use edit_file later; recall_context is a quoted historical name.'
    monkeypatch.setattr('infinidev.db.service.get_recent_turns_full',
                        lambda *a, **kw: [('user', literal)])
    text = render_session_context('synthetic-session', tool_names=())
    assert '<turn role="USER">\n' + literal + '\n</turn>' in text
    assert 'chat_history_search' not in text


def test_history_omission_markers_also_follow_grant(monkeypatch):
    monkeypatch.setattr('infinidev.db.service.get_recent_turns_full',
                        lambda *a, **kw: [('assistant', 'z' * 10000)] * 200)
    monkeypatch.setattr('infinidev.engine.loop.model_context.get_model_context_window',
                        lambda *a: 2048)
    text = render_session_context('synthetic-session', tool_names=())
    assert 'omitted' in text
    assert 'chat_history_search' not in text


@pytest.mark.parametrize('manual', [False, True])
def test_pressure_compaction_never_reintroduces_unoffered_recall(runtime, monkeypatch, manual):
    from infinidev.engine.turns import runtime as rt
    engine, agent, script, seen = runtime
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=not manual, supports_vision=False))
    if manual:
        first = reply(json.dumps({'tool_calls': [{'name': 'record',
                         'arguments': {'text': 'Evidence ' * 2000}}]}), tokens=900_000)
        final = reply(json.dumps({'tool_calls': [{'name': 'finish_turn',
                                                 'arguments': {'message': 'Done.'}}]}))
    else:
        first = reply('', tc('record', text='Evidence ' * 2000), tokens=900_000)
        final = reply('', tc('finish_turn', message='Done.'))
    script(first, final)
    assert engine.run_turn(agent, 'Synthetic request', task_tools=agent.tools).status == 'answered'
    request = json.dumps(seen[-1][0])
    assert 'Archived evidence' in request
    assert 'recall_context' not in request


@pytest.mark.parametrize('field', ['reference', 'query'])
@pytest.mark.parametrize('closed', [False, True])
@pytest.mark.parametrize('recall_offered', [False, True])
def test_long_saved_reference_is_projected_without_rewriting_durable_state(
        runtime, tool_context, field, closed, recall_offered):
    from infinidev.db.service import _persist_session_runtime_field, get_session_runtime_state
    from infinidev.engine.turns import runtime as rt
    engine, agent, script, seen = runtime
    if recall_offered:
        class Recall(RecordTool):
            name: str = 'recall_context'
        agent.tools.append(Recall())
    identifier = 'opaque-identifier-' + 'λ' * 700
    reference = f'[Archived evidence; recall_context {field}: ' + identifier
    if closed:
        reference += ']'
    refs = [reference]
    _persist_session_runtime_field(tool_context.session_id, 'conversation',
                                   {'plan': [], 'notes': [], 'evidence': refs})
    assert rt._load_state(tool_context.session_id)[1] == refs
    script(reply('Synthetic answer.'))
    assert engine.run_turn(agent, 'Continue', task_tools=agent.tools).status == 'answered'
    prior = next(m['content'] for m in seen[-1][0] if '<prior_state_context>' in m['content'])
    payload = json.loads(prior.split('\n', 1)[1].rsplit('\n', 1)[0])
    projected = payload['evidence'][0]
    assert len(projected) <= 500
    assert 'recall_context' not in projected
    assert f'{field} excerpt (incomplete)' in projected
    prefix = projected.split(': ', 1)[1][:-1]
    assert identifier.startswith(prefix)
    saved = get_session_runtime_state(tool_context.session_id)['ui_state']['conversation']
    assert saved['evidence'] == refs
