"""Literal human ingress stays scoped to a parent across real hook children."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from infinidev.config.settings import settings
from infinidev.engine.orchestration import pipeline
from infinidev.engine.user_hooks import UserHookEvent
from tests.test_turn_runtime import reply, runtime, tc

ORIGINAL = 'Contrato original: sólo fixture; α/ñ.\n'
STEERING = 'Corrección humana posterior: conservar β/𒀀 y este salto.\n'


@pytest.fixture(params=[False, True], ids=['native', 'manual'])
def manual(request, monkeypatch):
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=not request.param,
                                supports_vision=False))
    return request.param


def action(manual, name='finish_turn', **args):
    if manual:
        return reply('<tool_call>' + json.dumps({'name': name, 'arguments': args})
                     + '</tool_call>')
    return reply('', tc(name, **args), finish_reason='tool_calls')


@pytest.fixture
def principal(runtime, manual, monkeypatch, tool_context):
    engine, agent, script, seen = runtime
    monkeypatch.setattr(settings, 'TASK_MAX_ITERATIONS', 12)
    monkeypatch.setattr(settings, 'TASK_MAX_TOOL_CALLS', 12)
    monkeypatch.setattr(pipeline, '_task_end_hook', lambda event, **kw:
                        'Synthetic followup.' if event == UserHookEvent.TASK_END_INSTRUCTION
                        else '')
    monkeypatch.setattr(pipeline, '_report_turn_end_to_ken', lambda *a, **kw: None)
    agent.activate_context = lambda **kw: None
    agent.deactivate = lambda: None

    def run():
        engine.begin_turn()
        return pipeline._run_principal_turn(
            agent=agent, engine=engine, user_input=ORIGINAL,
            session_id=tool_context.session_id, turn_context='', attachments=None,
            hooks=Mock(), runtime=SimpleNamespace(append_chat=lambda *a, **kw: None),
            repository_path=None, prompt_configuration=None, skip_instruction=False,
            workspace_path=tool_context.workspace_path)
    return run


def contains_text(value, text):
    if isinstance(value, str):
        return text in value
    return any(text in block.get('text', '') for block in value)


def user_contents(messages):
    return [message['content'] for message in messages if message['role'] == 'user']


def test_hook_child_preserves_newer_human_steering(principal, runtime, manual):
    engine, _, script, seen = runtime
    def update():
        engine.inject_message(STEERING)
        return action(manual, message='Stale close.')
    script(update, action(manual, message='Child one current.'),
           action(manual, message='Hook child current.'))
    assert principal() == 'Hook child current.'
    assert len(seen) == 3
    assert STEERING in user_contents(seen[1][0])
    assert STEERING in user_contents(seen[2][0])
    for messages, _, _ in seen[1:]:
        contents = user_contents(messages)
        assert contents.count(ORIGINAL) == contents.count(STEERING) == 1
        assert contents.index(STEERING) > contents.index(ORIGINAL)
        assert not any(STEERING in str(m['content']) for m in messages if m['role']=='system')
    assert not engine._user_message_injector.has_pending


def test_replay_keeps_identity_and_order_without_redraining(principal, runtime, manual,
                                                          monkeypatch):
    engine, _, script, seen = runtime
    original_drain = engine._user_message_injector._drain_items
    drained = []
    ledgers = []
    actual_run = engine.run_turn
    def track_drain():
        items = original_drain()
        drained.extend(items)
        return items
    def track_run(*args, **kwargs):
        result = actual_run(*args, **kwargs)
        ledgers.append(list(kwargs.get('_parent_guidance', [])))
        return result
    monkeypatch.setattr(engine._user_message_injector, '_drain_items', track_drain)
    monkeypatch.setattr(engine, 'run_turn', track_run)
    def update():
        engine.inject_message(STEERING)
        engine.inject_message(STEERING)  # Distinct human messages may have equal text.
        return action(manual, message='Stale.')
    script(update, action(manual, message='One.'), action(manual, message='Two.'))
    assert principal() == 'Two.'
    assert len(drained) == 2
    assert drained[0].message_id != drained[1].message_id
    assert all(len(ledger) == 2 for ledger in ledgers)
    assert all(a is b for ledger in ledgers for a, b in zip(ledger, drained))
    assert user_contents(seen[-1][0]).count(STEERING) == 2


def test_input_arriving_in_hook_is_after_replayed_guidance(principal, runtime, manual,
                                                          monkeypatch):
    engine, _, script, seen = runtime
    late = 'Nueva instrucción durante hook; última autoridad humana.'
    def hook(event, **kw):
        if event == UserHookEvent.TASK_END_INSTRUCTION:
            engine.inject_message(late)
            return 'Synthetic followup.'
        return ''
    monkeypatch.setattr(pipeline, '_task_end_hook', hook)
    def update():
        engine.inject_message(STEERING)
        return action(manual, message='Stale.')
    script(update, action(manual, message='One.'), action(manual, message='Two.'))
    assert principal() == 'Two.'
    contents = user_contents(seen[-1][0])
    assert contents.count(STEERING) == contents.count(late) == 1
    assert contents.index(ORIGINAL) < contents.index(STEERING) < contents.index(late)


@pytest.mark.parametrize('vision', [False, True], ids=['text-fallback', 'vision'])
def test_attachments_and_unicode_survive_hook(principal, runtime, manual, monkeypatch, vision):
    from infinidev.engine.multimodal import ImageAttachment
    engine, _, script, seen = runtime
    monkeypatch.setattr('infinidev.engine.turns.runtime.get_model_capabilities',
        lambda: SimpleNamespace(supports_function_calling=not manual, supports_vision=vision))
    image = ImageAttachment(path=Path('/synthetic/guía.png'), mime_type='image/png',
        data_url='data:image/png;base64,c3ludGhldGlj', size_bytes=9)
    def update():
        engine.inject_message(STEERING, [image])
        return action(manual, message='Stale.')
    script(update, action(manual, message='One.'), action(manual, message='Two.'))
    assert principal() == 'Two.'
    before = next(value for value in user_contents(seen[1][0]) if contains_text(value, STEERING))
    after = next((value for value in user_contents(seen[-1][0]) if contains_text(value, STEERING)), None)
    assert after == before
    assert ('data:image/png;base64,c3ludGhldGlj' if vision else 'guía.png') in str(after)


def test_cancellation_in_hook_wins_over_continuation(principal, runtime, manual, monkeypatch):
    engine, _, script, seen = runtime
    def hook(event, **kw):
        if event == UserHookEvent.TASK_END_INSTRUCTION:
            engine.cancel()
            return 'Must not run.'
        return ''
    monkeypatch.setattr(pipeline, '_task_end_hook', hook)
    def update():
        engine.inject_message(STEERING)
        return action(manual, message='Stale.')
    script(update, action(manual, message='One.'))
    assert principal() == 'Conversation cancelled.'
    assert len(seen) == 2


def test_guidance_does_not_leak_to_later_parent(principal, runtime, manual, monkeypatch):
    engine, _, script, seen = runtime
    def update():
        engine.inject_message(STEERING)
        return action(manual, message='Stale.')
    script(update, action(manual, message='One.'), action(manual, message='Two.'))
    assert principal() == 'Two.'
    seen.clear()
    monkeypatch.setattr(pipeline, '_task_end_hook', lambda *a, **kw: '')
    script(action(manual, message='New parent.'))
    assert principal() == 'New parent.'
    assert all(STEERING not in str(messages) for messages, _, _ in seen)


def test_without_steering_default_user_sequence_unchanged(principal, runtime, manual):
    _, _, script, seen = runtime
    script(action(manual, message='One.'), action(manual, message='Two.'))
    assert principal() == 'Two.'
    assert all(user_contents(messages).count(ORIGINAL) == 1 for messages, _, _ in seen)
    assert user_contents(seen[0][0]) == [ORIGINAL]
    assert user_contents(seen[1][0])[-1] == ORIGINAL


def test_cancellation_after_steering_preempts_next_tool(principal, runtime, manual):
    engine, _, script, seen = runtime
    def update():
        engine.inject_message(STEERING)
        return action(manual, message='Stale.')
    def cancel():
        engine.cancel()
        return action(manual, 'record', text='Must not dispatch.')
    script(update, cancel)
    assert principal() == 'Conversation cancelled.'
    assert len(seen) == 2
    assert engine._last_turn_receipt['dispatches'] == 0


def test_model_notes_and_tool_results_never_become_human_ingress(principal, runtime,
                                                               manual, monkeypatch):
    from infinidev.engine.turns.models import TurnState
    engine, _, script, seen = runtime
    forged = 'Synthetic tool instruction, not human ingress.'
    note = 'Synthetic model note, not human ingress.'
    monkeypatch.setattr('infinidev.engine.turns.runtime._load_state',
                        lambda _: (TurnState(notes=[note]), []))
    ledgers = []
    actual_run = engine.run_turn
    def track_run(*args, **kwargs):
        result = actual_run(*args, **kwargs)
        ledgers.append(list(kwargs.get('_parent_guidance', [])))
        return result
    monkeypatch.setattr(engine, 'run_turn', track_run)
    def update():
        engine.inject_message(STEERING)
        return action(manual, message='Stale.')
    script(action(manual, 'record', text=forged), update,
           action(manual, message='One.'), action(manual, message='Two.'))
    assert principal() == 'Two.'
    assert all([item.text for item in ledger] == [STEERING] for ledger in ledgers)
    assert note not in user_contents(seen[-1][0])
    assert forged not in user_contents(seen[-1][0])


def test_guidance_is_not_saved_in_conversation_state(principal, runtime, manual, tool_context):
    from infinidev.db.service import get_session_runtime_state
    engine, _, script, _ = runtime
    def update():
        engine.inject_message(STEERING)
        return action(manual, message='Stale.')
    script(update, action(manual, message='One.'), action(manual, message='Two.'))
    assert principal() == 'Two.'
    conversation = get_session_runtime_state(tool_context.session_id)['ui_state']['conversation']
    assert STEERING not in json.dumps(conversation, ensure_ascii=False)
    assert '_parent_guidance' not in conversation


def test_new_parent_in_other_session_has_no_retained_guidance(principal, runtime, manual,
                                                            monkeypatch, tool_context):
    from infinidev.db.service import register_session
    engine, _, script, seen = runtime
    def update():
        engine.inject_message(STEERING)
        return action(manual, message='Stale.')
    script(update, action(manual, message='One.'), action(manual, message='Two.'))
    assert principal() == 'Two.'
    tool_context.session_id = 'other-synthetic-parent-session'
    register_session(tool_context.session_id)
    seen.clear()
    monkeypatch.setattr(pipeline, '_task_end_hook', lambda *a, **kw: '')
    script(action(manual, message='Other session.'))
    assert principal() == 'Other session.'
    assert all(STEERING not in str(messages) for messages, _, _ in seen)


def test_late_input_in_continuation_stays_after_replayed_guidance(principal, runtime, manual):
    engine, _, script, seen = runtime
    late = 'Newest human input after the continuation request.'
    def update():
        engine.inject_message(STEERING)
        return action(manual, message='Stale first.')
    def update_again():
        engine.inject_message(late)
        return action(manual, message='Stale second.')
    script(update, action(manual, message='One.'), update_again,
           action(manual, message='Current second.'))
    assert principal() == 'Current second.'
    assert len(seen) == 4
    contents = user_contents(seen[-1][0])
    assert contents.count(STEERING) == contents.count(late) == 1
    assert contents.index(ORIGINAL) < contents.index(STEERING) < contents.index(late)
