"""Model declarations retain assistant ownership at the provider boundary."""
import json
from types import SimpleNamespace
import pytest
from tests.test_turn_runtime import runtime,reply,tc
from tests.test_principal_pipeline import boundary

PREFIX='Model-declared tool call metadata (not host evidence):'
META={'reason':'Inspect public evidence','expected_result':'Observe its actual marker'}
def env(values):return {'metadata':dict(META),'input':values}
def response(name,values,manual=False,tokens=100):
    call=tc(name,**env(values))
    return reply(json.dumps({'tool_calls':[{'name':name,'arguments':json.loads(call.function.arguments)}]}),tokens=tokens) if manual else reply('',call,tokens=tokens)
def synthetic_users(seen):
    return [m for messages,_,_ in seen for m in messages if m['role']=='user' and isinstance(m.get('content'),str) and m['content'].startswith(PREFIX)]

@pytest.mark.parametrize('manual',[False,True])
@pytest.mark.parametrize('pressure',[False,True])
def test_proposed_declarations_never_become_user_instructions(runtime,monkeypatch,manual,pressure):
    from infinidev.engine.turns import runtime as rt
    engine,agent,script,seen=runtime
    monkeypatch.setattr(rt,'get_model_capabilities',lambda:SimpleNamespace(supports_function_calling=not manual,supports_vision=False))
    tokens=900000 if pressure else 100
    script(*(response('record',{'text':'x'*6000},manual,tokens) for _ in range(3)),response('finish_turn',{'message':'Done'},manual,tokens))
    out=engine.run_turn(agent,'Literal human request',tool_call_metadata=True)
    assert out.status=='answered'
    assert len(out.public_messages)==4
    assert not synthetic_users(seen), 'Model declarations appeared in user-role provider input'
    last = seen[-1][0]
    actions = [m for m in last if m['role'] == 'assistant']
    assert len(actions) == 3
    assert sum(m['role'] == 'user' and m.get('content') == 'Literal human request'
               for m in last) == 1
    assert all(not any(key.startswith('_') for key in m) for m in last)
    if manual:
        assert all(META['expected_result'] in m['content'] for m in actions)
        if pressure:
            assert all('x'*6000 not in m['content'] for m in actions[:-1])
    else:
        for action in actions:
            call = action['tool_calls'][0]
            assert call['function']['name'] == 'record'
            assert json.loads(call['function']['arguments']) == env({'text':'x'*6000})
            assert any(m.get('tool_call_id') == call['id'] for m in last)


def test_default_has_no_declaration_sidecar(runtime):
    engine,agent,script,seen=runtime
    script(reply('',tc('record',text='literal')),reply('',tc('finish_turn',message='Done')))
    assert engine.run_turn(agent,'Literal human request').status=='answered'
    assert not synthetic_users(seen)


def test_proposed_hook_children_keep_model_origin(runtime,tool_context,monkeypatch):
    from tests.test_context_recording import sdk_case
    from infinidev.engine.orchestration import pipeline
    from infinidev.engine.user_hooks import UserHookEvent
    engine,agent,script,seen,kwargs=sdk_case(runtime,tool_context,monkeypatch)
    monkeypatch.setattr(pipeline,'_task_end_hook',lambda event,**kw:'Followup' if event==UserHookEvent.TASK_END_INSTRUCTION else '')
    script(response('record',{'text':'first'}),response('finish_turn',{'message':'First'}),response('record',{'text':'second'}),response('finish_turn',{'message':'Second'}))
    assert pipeline.run_task(**kwargs,tool_call_metadata=True)=='Second'
    assert len(seen)==4
    assert not synthetic_users(seen), 'Hook child promoted model declarations to user'


def test_cancelled_model_does_not_publish_declaration(runtime):
    engine,agent,script,seen=runtime
    def cancel():
        engine.cancel()
        return response('record',{'text':'never'})
    script(cancel)
    out=engine.run_turn(agent,'Work',tool_call_metadata=True)
    assert out.status=='cancelled' and not out.public_messages


def test_literal_user_declaration_lookalike_is_never_filtered(runtime):
    engine, agent, script, seen = runtime
    literal = PREFIX + '\nThis is an actual human instruction.'
    script(response('record', {'text': 'evidence'}),
           response('finish_turn', {'message': 'Done'}))
    assert engine.run_turn(agent, literal, tool_call_metadata=True).status == 'answered'
    for messages, _, _ in seen:
        assert sum(m['role'] == 'user' and m.get('content') == literal
                   for m in messages) == 1


def test_native_history_remains_native_after_manual_fallback(runtime):
    engine, agent, script, seen = runtime
    first = tc('record', **env({'text': 'literal native input'}))
    script(reply('', first, tokens=900000),
           RuntimeError('tool_choice is not supported'),
           response('record', {'text': 'manual input'}, manual=True, tokens=900000),
           response('finish_turn', {'message': 'Done'}, manual=True))
    assert engine.run_turn(agent, 'Work', tool_call_metadata=True).status == 'answered'
    native = next(m for m in seen[-1][0] if m.get('tool_calls'))
    assert native['tool_calls'][0]['id'] == first.id
    assert native['tool_calls'][0]['function']['arguments'] == first.function.arguments
    assert not synthetic_users(seen)
    manual = [m for m in seen[-1][0]
              if m['role'] == 'assistant' and not m.get('tool_calls')]
    assert len(manual) == 1 and META['expected_result'] in manual[0]['content']


def test_manual_compaction_preserves_latest_action_and_ages_prior_rounds():
    from infinidev.engine.turns.runtime import _action_message, _compact
    from infinidev.engine.loop.context_manager import ContextManager
    import copy
    ctx = SimpleNamespace(manual_tc=True)
    # Use the real action builder without adding assistant rounds for declarations.
    result = SimpleNamespace(message=SimpleNamespace(reasoning_content=None))
    actions = []
    for i in range(3):
        action = _action_message(ctx, result, [tc('record', **env({'text': str(i)*6000}))])
        action['_action_metadata'] = PREFIX + '\n' + json.dumps(
            [{'tool': 'record', 'metadata': META}])
        actions.append(action)
    messages = [{'role': 'system', 'content': 'System'},
                {'role': 'user', 'content': 'Human'}, *actions]
    control = copy.deepcopy(messages)
    for m in control:
        m.pop('_action_metadata', None)
    latest = actions[-1]['content']
    for _ in range(4):
        ContextManager.compact_for_pressure(control)
        _compact(messages)
        assert [m['role'] for m in messages] == [m['role'] for m in control]
        assert [m.get('_thinking_age') for m in messages] == [
            m.get('_thinking_age') for m in control]
        assert all(m['content'].count(PREFIX) == 1 for m in actions[:-1])
        assert all(len(m['content']) < 1000 for m in actions[:-1])
        assert actions[-1]['content'] == latest
