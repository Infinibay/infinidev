"""Canonical tool grants and literal atomic commands preserve task boundaries."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_principal_pipeline import boundary, call


def catalog():
    from infinidev.engine.turns import catalog as module
    return module


def tool(name):
    return SimpleNamespace(name=name)


def test_effective_catalog_preserves_order_and_filters_runtime_and_legacy():
    values = [tool('read_file'), tool('step_complete'), tool('finish_turn'), tool('edit_file')]
    assert catalog().effective_tools(values) == [values[0], values[3]]


def test_sdk_grant_requires_canonical_identity_and_preserves_empty():
    read = tool('read_file')
    assert catalog().freeze_task_tools([read], None) is None
    assert catalog().freeze_task_tools([read], []).tools == ()
    grant = catalog().freeze_task_tools([read], [read])
    assert grant.tools == (read,)
    assert grant.names == ('read_file',)
    with pytest.raises(ValueError, match='canonical'):
        catalog().freeze_task_tools([read], [tool('read_file')])
    with pytest.raises(ValueError):
        catalog().freeze_task_tools([read], [tool('delete_file')])


def test_selector_rejects_unknown_alias_duplicates_and_excluded_legacy():
    read, legacy = tool('read_file'), tool('step_complete')
    for names in [('write_file',), ('step_complete',), ('read_file', 'read_file')]:
        with pytest.raises(ValueError):
            catalog().select_tool_names([read, legacy], names)
    assert catalog().select_tool_names([read], ('finish_turn', 'update_plan')) == ()
    assert catalog().select_tool_names([read], ('read_file',)) == (read,)


def test_snapshot_detects_renamed_tool():
    read = tool('read_file')
    grant = catalog().freeze_task_tools([read], [read])
    read.name = 'delete_file'
    with pytest.raises(ValueError):
        grant.validate()
    with pytest.raises(ValueError):
        catalog().freeze_task_tools([read], grant)


@pytest.mark.parametrize('names,body', [
    ('read_file,create_file', '  λ\n!literal -- /exit\n'),
    ('', '/init is literal task text'),
])
def test_atomic_command_preserves_opaque_body(names, body):
    from infinidev.cli.tool_commands import parse_tool_request
    prefix = '/tools run' + (' ' + names if names else '')
    request = parse_tool_request(prefix + ' -- ' + body)
    assert request.body == body
    assert request.names == tuple(names.split(',')) if names else request.names == ()


@pytest.mark.parametrize('text', ['/tools run read_file', '/tools run read_file -- ',
                                 '/tools next read_file -- task',
                                 '/tools run read_file,,edit_file -- task'])
def test_invalid_command_does_not_create_request(text):
    from infinidev.cli.tool_commands import parse_tool_request
    with pytest.raises(ValueError):
        parse_tool_request(text)


@pytest.mark.parametrize('empty', [False, True])
def test_principal_forwards_explicit_snapshot_to_both_children(boundary, monkeypatch, empty):
    from infinidev.engine.orchestration import pipeline
    from infinidev.engine.user_hooks import UserHookEvent
    agent, engine, hooks = boundary
    read = tool('read_file')
    agent.tools = [read]
    chosen = [] if empty else [read]
    def end(event, **kwargs):
        if event == UserHookEvent.TASK_END_INSTRUCTION:
            chosen.clear()
            agent.tools = [tool('delete_file')]
            return 'Continue within the same grant.'
        return ''
    monkeypatch.setattr(pipeline, '_task_end_hook', end)
    assert call(boundary, 'Literal task', task_tools=chosen) == 'respuesta'
    assert engine.run_turn.call_count == 2
    for args in engine.run_turn.call_args_list:
        assert tuple(args.kwargs['task_tools']) == (() if empty else (read,))
    engine.run_turn.reset_mock()
    call(boundary, 'Next parent')
    assert engine.run_turn.call_args.kwargs.get('task_tools') is None


@pytest.mark.parametrize('kind', ['replacement', 'legacy', 'mode', 'gather', 'autonomous'])
def test_invalid_grant_fails_before_hooks_or_provider(boundary, monkeypatch, kind):
    from infinidev.config.settings import settings
    from infinidev.engine.orchestration import pipeline
    agent, engine, hooks = boundary
    read, legacy = tool('read_file'), tool('step_complete')
    agent.tools = [read, legacy]
    start = Mock(side_effect=AssertionError('No hook before rejecting grant'))
    monkeypatch.setattr(pipeline, '_run_task_start_hook', start)
    kwargs = {'task_tools': [read]}
    if kind == 'replacement':
        kwargs['task_tools'] = [tool('read_file')]
    elif kind == 'legacy':
        kwargs['task_tools'] = [legacy]
    elif kind == 'mode':
        monkeypatch.setattr(settings, 'TASK_ENGINE_MODE', 'react')
    elif kind == 'gather':
        kwargs['force_gather'] = True
    else:
        kwargs['autonomous'] = True
    with pytest.raises(ValueError):
        call(boundary, **kwargs)
    start.assert_not_called()
    engine.run_turn.assert_not_called()


@pytest.mark.parametrize('transport', ['native', 'manual', 'fallback'])
def test_grant_reaches_provider_and_rejects_entire_ungranted_batch(
        boundary, monkeypatch, workspace_dir, transport):
    import copy
    import json
    from infinidev.engine.loop.engine import LoopEngine
    from infinidev.engine.turns import runtime as rt
    from infinidev.tools.base.context import clear_agent_context, set_context
    from tests.test_turn_runtime import RecordTool, reply, tc

    agent, _, hooks = boundary
    agent.backstory = 'Follow the literal current request.'
    agent.role = 'developer'
    agent.workspace_path = str(workspace_dir)
    record = RecordTool()
    class Forbidden(RecordTool):
        name: str = 'delete_file'
    agent.tools = [record, Forbidden()]
    agent.activate_context = lambda session_id: set_context(
        agent_id=agent.agent_id, project_id=None, session_id=session_id,
        workspace_path=str(workspace_dir))
    agent.deactivate = lambda: clear_agent_context(agent.agent_id)
    monkeypatch.setattr(rt, 'get_litellm_params', lambda: {'model': 'scripted'})
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=transport != 'manual', supports_vision=False))
    monkeypatch.setattr(rt, '_get_model_max_context', lambda _: 1_000_000)
    executed = []
    monkeypatch.setattr(RecordTool, '_run', lambda self, text: executed.append(text) or text)
    calls = [('record', {'text':'must not run'}), ('delete_file', {'text':'forbidden'})]
    manual = transport != 'native'
    first = (reply(json.dumps({'tool_calls':[dict(name=n, arguments=a) for n,a in calls]}))
             if manual else reply('', *(tc(n,**a) for n,a in calls), finish_reason='tool_calls'))
    final = (reply(json.dumps({'tool_calls':[dict(name='finish_turn',arguments={'message':'Rejected.'})]}))
             if manual else reply('',tc('finish_turn',message='Rejected.'),finish_reason='tool_calls'))
    responses = iter(([RuntimeError('does not support tools')] if transport=='fallback' else [])
                     + [first, final])
    seen=[]
    def provider(params, messages, tools=None, **kwargs):
        seen.append((copy.deepcopy(messages), copy.deepcopy(tools)))
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result
    monkeypatch.setattr(rt, 'call_llm', provider)
    granted = catalog().freeze_task_tools(agent.tools, [record])
    assert call((agent, LoopEngine(), hooks), 'Literal task', task_tools=granted) == 'Rejected.'
    assert executed == []
    for messages, schemas in seen:
        if schemas is not None:
            assert {s['function']['name'] for s in schemas} == {'record','finish_turn','update_plan'}
        else:
            system=messages[0]['content']
            payload=system.split('Available tools and their JSON schemas:\n',1)[1]
            offered,_=json.JSONDecoder().raw_decode(payload)
            assert {s['function']['name'] for s in offered} == {'record','finish_turn','update_plan'}


def test_start_hook_cannot_rename_an_explicit_tool_before_first_child(boundary, monkeypatch):
    from infinidev.engine.orchestration import pipeline
    agent, engine, hooks = boundary
    read = tool('read_file')
    agent.tools = [read]
    def rename(**kwargs):
        read.name = 'delete_file'
        return 'Hook context'
    monkeypatch.setattr(pipeline, '_run_task_start_hook', rename)
    assert 'failed' in call(boundary, task_tools=[read]).lower()
    engine.run_turn.assert_not_called()


def test_canonical_selection_does_not_bypass_effect_permission(monkeypatch, tool_context, temp_db):
    from infinidev.config.settings import settings
    from infinidev.engine.turns import runtime as rt
    from infinidev.engine.loop.engine import LoopEngine
    from infinidev.tools.base.tool_effects import ToolEffects
    from tests.test_turn_runtime import RecordTool, reply, tc
    from infinidev.tools import permission

    class Sensitive(RecordTool):
        name: str = 'sensitive_probe'
        effects: ToolEffects = ToolEffects(destructive=True)
    selected = Sensitive()
    agent = SimpleNamespace(agent_id=tool_context.agent_id, project_id=tool_context.project_id,
                            workspace_path=tool_context.workspace_path, role='developer',
                            backstory='Help the user.', tools=[selected])
    monkeypatch.setattr(settings, 'TOOL_EFFECTS_PERMISSION', 'auto')
    monkeypatch.setattr(permission, 'is_permission_handler_registered', lambda: True)
    deny = Mock(return_value=False)
    monkeypatch.setattr(permission, 'request_permission', deny)
    executed = Mock(side_effect=AssertionError('No approval was granted'))
    monkeypatch.setattr(Sensitive, '_run', executed)
    monkeypatch.setattr(rt, 'get_litellm_params', lambda: {'model':'scripted'})
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=True, supports_vision=False))
    monkeypatch.setattr(rt, '_get_model_max_context', lambda _: 1_000_000)
    replies=iter([reply('',tc('sensitive_probe',text='synthetic')),
                  reply('',tc('finish_turn',message='Denied.'))])
    monkeypatch.setattr(rt, 'call_llm', lambda *a, **kw: next(replies))
    grant = catalog().freeze_task_tools(agent.tools, [selected])
    assert LoopEngine().run_turn(agent, 'Synthetic request', task_tools=grant.tools).status=='answered'
    deny.assert_called_once()
    executed.assert_not_called()


@pytest.mark.parametrize('empty', [False, True])
def test_explicit_grant_does_not_enable_gather_changed_by_start_hook(
        boundary, monkeypatch, empty):
    from infinidev.config.settings import settings
    from infinidev.engine.orchestration import pipeline
    from infinidev.engine.user_hooks import UserHookEvent
    agent, engine, _ = boundary
    read = tool('read_file')
    agent.tools = [read]
    def start(**kwargs):
        settings.GATHER_ENABLED = True
        return 'Hook enabled gathering for later requests'
    monkeypatch.setattr(pipeline, '_run_task_start_hook', start)
    gather = Mock(side_effect=AssertionError('No gather outside accepted grant'))
    monkeypatch.setattr(pipeline, '_run_gather_phase', gather)
    monkeypatch.setattr(pipeline, '_task_end_hook', lambda event, **kw:
                        'Continue' if event == UserHookEvent.TASK_END_INSTRUCTION else '')
    assert call(boundary, task_tools=[] if empty else [read]) == 'respuesta'
    gather.assert_not_called()
    assert engine.run_turn.call_count == 2


def test_default_none_keeps_existing_hook_gather_behavior(boundary, monkeypatch):
    from infinidev.config.settings import settings
    from infinidev.engine.orchestration import pipeline
    def start(**kwargs):
        settings.GATHER_ENABLED = True
        return ''
    monkeypatch.setattr(pipeline, '_run_task_start_hook', start)
    gather = Mock(return_value=('Gathered', []))
    monkeypatch.setattr(pipeline, '_run_gather_phase', gather)
    assert call(boundary) == 'respuesta'
    gather.assert_called_once()


def test_default_none_keeps_mode_selection_after_start_hook(boundary, monkeypatch):
    from infinidev.config.settings import settings
    from infinidev.engine.orchestration import pipeline
    def start(**kwargs):
        settings.TASK_ENGINE_MODE = 'react'
        return ''
    monkeypatch.setattr(pipeline, '_run_task_start_hook', start)
    with pytest.raises(AssertionError, match='add_task'):
        call(boundary)
    boundary[1].run_turn.assert_not_called()


def test_default_none_reads_settings_replaced_by_start_hook(boundary, monkeypatch):
    import importlib
    from infinidev.engine.orchestration import pipeline
    settings_module = importlib.import_module('infinidev.config.settings')
    replacement = settings_module.settings.model_copy(update={'TASK_ENGINE_MODE':'react'})
    def start(**kwargs):
        monkeypatch.setattr(settings_module, 'settings', replacement)
        return ''
    monkeypatch.setattr(pipeline, '_run_task_start_hook', start)
    with pytest.raises(AssertionError, match='add_task'):
        call(boundary)
    boundary[1].run_turn.assert_not_called()
