"""Opt-in declarations are transport, not executable tool parameters."""
from __future__ import annotations
import json
from types import SimpleNamespace
import pytest
from tests.test_turn_runtime import runtime, reply, tc
from tests.test_principal_pipeline import boundary

def envelope(values, **changes):
    body={'metadata':{'reason':'Inspect the fixture','expected_result':'Observe its actual signature'},'input':values}
    body.update(changes)
    return body

@pytest.mark.parametrize('manual',[False,True])
def test_metadata_transport_preserves_literals(runtime,monkeypatch,manual):
    from infinidev.engine.turns import runtime as rt
    engine,agent,script,seen=runtime
    monkeypatch.setattr(rt,'get_model_capabilities',lambda:SimpleNamespace(supports_function_calling=not manual,supports_vision=False))
    value='literal metadata input \n <think> untouched'
    calls=[tc('record',**envelope({'text':value})),tc('finish_turn',**envelope({'message':'Done'}))]
    def response(call):
        return reply(json.dumps({'tool_calls':[{'name':call.function.name,'arguments':json.loads(call.function.arguments)}]})) if manual else reply('',call)
    script(*(response(c) for c in calls))
    out=engine.run_turn(agent,'Work',tool_call_metadata=True)
    assert out.status=='answered' and out.message=='Done'
    assert any('Observe its actual signature' in p for p in out.public_messages)
    assert any(value in str(m.get('content')) for m in seen[1][0])
    if not manual:
        schema=seen[0][1][0]['function']['parameters']
        assert set(schema['required'])=={'metadata','input'}
        assert 'reason' in schema['properties']['metadata']['properties']
        assert json.loads(seen[1][0][-2]['tool_calls'][0]['function']['arguments'])==envelope({'text':value})

@pytest.mark.parametrize('bad',[None,{}, {'reason':' ','expected_result':'x'}, {'reason':1,'expected_result':'x'}, {'reason':'x'*501,'expected_result':'x'}])
def test_bad_metadata_rejects_entire_batch(runtime,bad):
    engine,agent,script,seen=runtime
    good=tc('record',**envelope({'text':'NEVER_EXECUTE'}))
    broken=tc('record',**envelope({'text':'bad'},metadata=bad))
    script(*(reply('',good,broken) for _ in range(3)))
    out=engine.run_turn(agent,'Work',tool_call_metadata=True)
    assert out.status=='failed'
    assert engine._last_turn_receipt['dispatches']==0
    assert not out.public_messages


def test_sdk_forwards_explicit_opt_in_to_both_children(boundary,monkeypatch):
    from tests.test_principal_pipeline import call
    from infinidev.engine.orchestration import pipeline
    from infinidev.engine.user_hooks import UserHookEvent
    monkeypatch.setattr(pipeline,'_task_end_hook',lambda event,**kw:
        'Check' if event==UserHookEvent.TASK_END_INSTRUCTION else '')
    call(boundary,tool_call_metadata=True)
    assert boundary[1].run_turn.call_count==2
    assert all(c.kwargs['tool_call_metadata'] is True for c in boundary[1].run_turn.call_args_list)


def test_default_sdk_does_not_forward_new_protocol(boundary):
    from tests.test_principal_pipeline import call
    call(boundary)
    assert 'tool_call_metadata' not in boundary[1].run_turn.call_args.kwargs


@pytest.mark.parametrize('use_phase_engine',[True,False])
def test_unsupported_routes_reject_before_work(boundary,monkeypatch,use_phase_engine):
    from tests.test_principal_pipeline import call
    from infinidev.config.settings import settings
    if not use_phase_engine: monkeypatch.setattr(settings,'GATHER_ENABLED',True)
    with pytest.raises(ValueError,match='metadata requires'):
        call(boundary,tool_call_metadata=True,use_phase_engine=use_phase_engine)
    boundary[1].run_turn.assert_not_called()


def test_schema_and_literal_legacy_input_preserved():
    from infinidev.engine.turns.action_metadata import transport_schema
    from infinidev.engine.schema_sanitizer import tool_to_openai_schema
    from infinidev.tools.shell.execute_command_tool import ExecuteCommandTool
    from infinidev.tools.shell.execute_command_input import ExecuteCommandInput
    from infinidev.engine.turns.runtime import _validated_arguments
    original=tool_to_openai_schema(ExecuteCommandTool())
    import copy
    before=copy.deepcopy(original)
    wrapped=transport_schema(original)
    assert original==before
    assert wrapped['function']['parameters']['properties']['input']==original['function']['parameters']
    values={'command':'printf "literal"','rationale':' actual reason ','env':{'A':'<think>literal'}}
    ctx=SimpleNamespace(tool_call_metadata=True,tool_dispatch={'execute_command':SimpleNamespace(args_schema=ExecuteCommandInput)})
    call=tc('execute_command',**envelope(values))
    assert _validated_arguments(ctx,call)==ExecuteCommandInput.model_validate(values).model_dump()


def test_manual_metadata_survives_pressure(runtime,monkeypatch):
    from infinidev.engine.turns import runtime as rt
    engine,agent,script,seen=runtime
    monkeypatch.setattr(rt,'get_model_capabilities',lambda:SimpleNamespace(supports_function_calling=False,supports_vision=False))
    def response(name,values):
        return reply(json.dumps({'tool_calls':[{'name':name,'arguments':envelope(values)}]}),tokens=900000)
    script(response('record',{'text':'x'*6000}),response('record',{'text':'y'*6000}),response('finish_turn',{'message':'Done'}))
    assert engine.run_turn(agent,'Work',tool_call_metadata=True).status=='answered'
    assert sum('Observe its actual signature' in str(m.get('content')) for m in seen[-1][0])>=2


def test_active_wrapper_obeys_provider_sanitizer(runtime,monkeypatch):
    from infinidev.engine.turns import runtime as rt
    from infinidev.engine.schema_sanitizer import _sanitize_tool_schema
    engine,agent,script,seen=runtime
    monkeypatch.setattr(rt,'get_model_capabilities',lambda:SimpleNamespace(supports_function_calling=True,supports_vision=False,needs_schema_sanitization=True))
    script(reply('',tc('finish_turn',**envelope({'message':'Done'}))))
    assert engine.run_turn(agent,'Work',tool_call_metadata=True).status=='answered'
    assert all(s==_sanitize_tool_schema(s) for s in seen[0][1])


def test_metadata_close_deferred_does_not_publish(runtime):
    engine,agent,script,seen=runtime
    def update():
        engine.inject_message('Use the newer request')
        return reply('stale details',tc('finish_turn',**envelope({'message':'stale'})))
    script(update,reply('Updated'))
    out=engine.run_turn(agent,'Work',tool_call_metadata=True)
    assert out.message=='Updated' and out.public_messages==[]


def test_metadata_cancelled_model_does_not_publish(runtime):
    engine,agent,script,seen=runtime
    def cancel():
        engine.cancel()
        return reply('',tc('record',**envelope({'text':'NEVER'})))
    script(cancel)
    out=engine.run_turn(agent,'Work',tool_call_metadata=True)
    assert out.status=='cancelled' and not out.public_messages
    assert engine._last_turn_receipt['dispatches']==0


def test_transport_hoists_real_nested_model_refs():
    from pydantic import BaseModel
    from infinidev.engine.turns.action_metadata import transport_schema
    class Item(BaseModel):
        value:str
    class Input(BaseModel):
        items:list[Item]
    original=Input.model_json_schema()
    schema=transport_schema({'function':{'parameters':original}})['function']['parameters']
    assert schema['$defs']['Item']==original['$defs']['Item']
    assert '$defs' not in schema['properties']['input']
    assert schema['properties']['input']['properties']['items']['items']['$ref']=='#/$defs/Item'
