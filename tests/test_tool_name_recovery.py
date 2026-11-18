"""Exact-name failures expose the actual grant without expanding dispatch."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_turn_runtime import reply, runtime, tc


def action(manual, calls):
    if manual:
        return reply(json.dumps({'tool_calls':[{'name':n,'arguments':a} for n,a in calls]}))
    return reply('', *(tc(n,**a) for n,a in calls), finish_reason='tool_calls')


def observations(request, manual):
    result=[]
    for message in request[0]:
        body=message.get('content')
        if message['role']=='tool':
            result.append(json.loads(body))
        elif manual and message['role']=='user' and isinstance(body,str) and body.startswith('[Tool:'):
            result.append(json.loads(body.split('\n',1)[1]))
    return result


@pytest.mark.parametrize('transport', ['native','manual','fallback'])
def test_ungranted_name_is_typed_and_catalog_is_actual(runtime, monkeypatch, transport):
    from infinidev.engine.turns import runtime as rt
    engine,agent,script,seen=runtime
    manual=transport!='native'
    monkeypatch.setattr(rt,'get_model_capabilities',lambda:SimpleNamespace(
        supports_function_calling=transport!='manual',supports_vision=False))
    responses=([RuntimeError('does not support tools')] if transport=='fallback' else [])
    responses += [action(manual,[('devkit:record',{'text':'PRIVATE_INPUT'})]),
                  action(manual,[('finish_turn',{'message':'Stopped.'})])]
    script(*responses)
    assert engine.run_turn(agent,'Synthetic request',task_tools=agent.tools).status=='answered'
    body=observations(seen[-1],manual)[0]
    assert body['executed'] is False
    assert body['errors'][0]['type']=='tool_not_offered'
    assert body['errors'][0]['loc']==['tool_name']
    assert body['requested_name']=='devkit:record'
    assert body['requested_name_complete'] is True
    assert body['offered_names']==['record','finish_turn','update_plan']
    assert body['offered_names_complete'] is True
    assert body['offered_name_count']==3
    assert 'PRIVATE_INPUT' not in json.dumps(body)
    assert engine._last_turn_receipt['dispatches']==0


def test_argument_failure_keeps_actual_pydantic_type(runtime):
    engine,agent,script,seen=runtime
    script(action(False,[('record',{})]),action(False,[('finish_turn',{'message':'Missing input.'})]))
    engine.run_turn(agent,'Synthetic request')
    body=observations(seen[1],False)[0]
    assert body['errors'][0]['type']=='missing'
    assert body['errors'][0]['loc']==['text']
    assert body['executed'] is False
    assert 'tool_not_offered' not in json.dumps(body)


@pytest.mark.parametrize('manual', [False, True])
def test_mixed_batch_feedback_preserves_valid_siblings(runtime, monkeypatch, manual):
    from infinidev.engine.turns import runtime as rt
    from tests.test_turn_runtime import RecordTool

    engine, agent, script, seen = runtime
    executed = Mock(return_value='Recorded')
    monkeypatch.setattr(RecordTool, '_run', executed)
    if manual:
        monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
            supports_function_calling=False, supports_vision=False))

    valid = ('record', {'text': 'valid'})
    invalid_name = ('devkit:record', {'text': 'PRIVATE_INPUT'})
    invalid_arguments = ('record', {'text': 24})

    def retry():
        executed.assert_not_called()
        bodies = observations(seen[-1], manual)
        assert len(bodies) == 3 and all(body['executed'] is False for body in bodies)
        assert bodies[0]['errors'] == []
        assert 'unchanged' in bodies[0]['recovery']
        assert 'offered_names' not in bodies[0]
        assert 'requested_name' not in bodies[0]
        assert bodies[1]['errors'][0]['type'] == 'tool_not_offered'
        assert bodies[1]['requested_name'] == 'devkit:record'
        assert bodies[1]['offered_names'] == ['record', 'finish_turn', 'update_plan']
        assert bodies[2]['errors'][0]['type'] == 'string_type'
        assert 'offered_names' not in bodies[2]
        assert 'arguments' in bodies[2]['recovery']
        return action(manual, [valid, ('record', {'text': 'fixed arguments'})])

    script(action(manual, [valid, invalid_name, invalid_arguments]), retry,
           action(manual, [('finish_turn', {'message': 'Recorded.'})]))
    assert engine.run_turn(agent, 'Record both valid observations.').status == 'answered'
    assert [entry.kwargs['text'] for entry in executed.call_args_list] == ['valid', 'fixed arguments']


def test_empty_grant_reports_only_runtime_names(runtime):
    engine,agent,script,seen=runtime
    script(action(False,[('record',{'text':'PRIVATE_INPUT'})]),
           action(False,[('finish_turn',{'message':'No workspace tools.'})]))
    engine.run_turn(agent,'Synthetic request',task_tools=[])
    body=observations(seen[1],False)[0]
    assert body['offered_names']==['finish_turn','update_plan']
    assert body['offered_name_count']==2
    assert body['errors'][0]['type']=='tool_not_offered'


def test_three_rejected_name_batches_keep_fuse_and_zero_dispatch(runtime):
    engine,agent,script,seen=runtime
    script(*(action(False,[('devkit:record',{'text':'PRIVATE_INPUT'})]) for _ in range(3)))
    result=engine.run_turn(agent,'Synthetic request')
    assert result.status=='failed'
    assert len(seen)==3
    assert engine._last_turn_receipt['dispatches']==0
    assert 'tool validation failed in 3 batches' in result.message
    assert 'tool_not_offered' in result.message


def test_structural_archive_has_no_argument_value(runtime, tool_context):
    from infinidev.tools.base.db import execute_with_retry
    engine,agent,script,seen=runtime
    script(action(False,[('devkit:record',{'text':'PRIVATE_INPUT'})]),
           action(False,[('finish_turn',{'message':'Rejected.'})]))
    engine.run_turn(agent,'Synthetic request')
    rows=execute_with_retry(lambda c:c.execute(
        "SELECT content FROM working_memory WHERE session_id=? AND kind='tool_output' "
        "AND title LIKE '% tool_validation_rejected part %'",(tool_context.session_id,)).fetchall())
    reports=[json.loads(r[0]) for r in rows]
    assert len(reports)==1
    report=reports[0]
    assert report['executed'] is False
    assert report['calls'][0]['errors'][0]=={'field':'tool_name','type':'tool_not_offered'}
    assert report['calls'][0]['requested_name']=='devkit:record'
    assert report['calls'][0]['offered_names']==['record','finish_turn','update_plan']
    assert 'PRIVATE_INPUT' not in json.dumps(report)


def test_exact_retry_after_rejection_dispatches_only_the_offered_name(runtime):
    engine, agent, script, seen = runtime
    script(action(False, [('devkit:record', {'text': 'candidate'})]),
           action(False, [('record', {'text': 'candidate'})]),
           action(False, [('finish_turn', {'message': 'Recorded.'})]))
    result = engine.run_turn(agent, 'Synthetic request')
    assert result.status == 'answered'
    assert engine._last_turn_receipt['dispatches'] == 1
    assert observations(seen[1], False)[0]['errors'][0]['type'] == 'tool_not_offered'
    assert seen[2][0][-1]['content'].startswith('candidate')


def test_name_diagnostics_are_bounded_and_mark_incomplete():
    from infinidev.engine.turns.runtime import _tool_name_details
    names = tuple(('名前' * 60) + str(i) for i in range(100))
    details = _tool_name_details('接頭辞:' + 'x' * 200, names)
    assert len(details['requested_name']) == 100
    assert details['requested_name_complete'] is False
    assert len(json.dumps(details['offered_names'], ensure_ascii=False).encode()) <= 8192
    assert details['offered_names_complete'] is False
    assert details['offered_name_count'] == len(names)
    assert tuple(details['offered_names']) == names[:len(details['offered_names'])]


def test_single_oversized_name_is_not_published_as_complete_catalog():
    from infinidev.engine.turns.runtime import _tool_name_details
    details = _tool_name_details('record', ('界' * 4000, 'record'))
    assert details['offered_names'] == []
    assert details['offered_names_complete'] is False
    assert details['offered_name_count'] == 2
