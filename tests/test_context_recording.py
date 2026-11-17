"""Passive context diagnostics keep content out of per-invocation artifacts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests.test_principal_pipeline import boundary
from tests.test_turn_runtime import runtime


def records(path: Path) -> list[dict]:
    return [json.loads(line) for line in (path / 'metadata.jsonl').read_text().splitlines()]


def test_disabled_recording_has_no_io(tmp_path):
    from infinidev.engine.context_recording import recording, current_recorder
    with recording(None, workspace=tmp_path):
        assert current_recorder() is None
    assert list(tmp_path.iterdir()) == []


def test_private_metadata_only_and_redacted_hash(tmp_path):
    from infinidev.engine.context_recording import recording, current_recorder
    from infinidev.engine.history.redaction import redact_text
    path = tmp_path / 'capture'
    body = 'password=very-private-value body-not-to-save'
    with recording(path, workspace=tmp_path / 'workspace') as capture:
        assert current_recorder() is capture
        identity = capture.source('history', 'assistant', body, source_id=79,
                                  loader_clipped=False, renderer_clipped=True)
    text = (path / 'metadata.jsonl').read_text()
    assert 'very-private-value' not in text and 'body-not-to-save' not in text
    row = next(r for r in records(path) if r['kind'] == 'source')
    assert row['id'] == identity and row['source_id'] == 79
    assert row['renderer_clipped'] is True
    assert row['hash_scope'] == 'redacted_public'
    assert row['content_sha256'] == hashlib.sha256(redact_text(body).encode()).hexdigest()
    assert path.stat().st_mode & 0o777 == 0o700
    assert (path / 'metadata.jsonl').stat().st_mode & 0o777 == 0o600
    assert json.loads((path / 'manifest.json').read_text())['complete'] is True
    assert current_recorder() is None


def test_nonpublic_and_unknown_content_omit_digest(tmp_path):
    from infinidev.engine.context_recording import recording
    path = tmp_path / 'capture'
    with recording(path, workspace=tmp_path / 'workspace') as capture:
        capture.source('history', 'work_summary', 'PRIVATE_INTERNAL')
        capture.source('literal_input', 'user', [{'type': 'unknown', 'data': 'SECRET_ASSET'}])
    text = (path / 'metadata.jsonl').read_text()
    assert 'PRIVATE_INTERNAL' not in text and 'SECRET_ASSET' not in text
    assert all(row['content_sha256'] is None for row in records(path) if row['kind'] == 'source')
    assert hashlib.sha256(b'PRIVATE_INTERNAL').hexdigest() not in text


def test_directory_collision_and_symlink_fail_before_recording(tmp_path):
    from infinidev.engine.context_recording import recording
    existing = tmp_path / 'existing'
    existing.mkdir()
    link = tmp_path / 'linked'
    link.symlink_to(existing, target_is_directory=True)
    for path in (existing, link, link / 'child'):
        with pytest.raises((OSError, ValueError)):
            with recording(path, workspace=tmp_path / 'workspace'):
                pytest.fail('Invalid capture path was admitted')
    assert list(existing.iterdir()) == []


def test_capture_inside_workspace_is_rejected(tmp_path):
    from infinidev.engine.context_recording import recording
    with pytest.raises(ValueError):
        with recording(tmp_path / 'capture', workspace=tmp_path):
            pytest.fail('Recording would alter the experiment baseline')


def test_duplicate_text_has_distinct_provenance_and_nested_scope_isolated(tmp_path):
    from infinidev.engine.context_recording import recording, current_recorder
    with recording(tmp_path / 'one', workspace=tmp_path / 'workspace') as first:
        a = first.source('history', 'user', 'Same', source_id=1)
        b = first.source('history', 'user', 'Same', source_id=2)
        assert a != b
        with recording(tmp_path / 'two', workspace=tmp_path / 'workspace') as second:
            assert current_recorder() is second
        assert current_recorder() is first
    assert current_recorder() is None


def test_budget_keeps_whole_records_and_marks_incomplete(tmp_path):
    from infinidev.engine.context_recording import recording
    path = tmp_path / 'capture'
    with recording(path, workspace=tmp_path / 'workspace', max_bytes=300) as capture:
        for n in range(20):
            capture.source('history', 'assistant', str(n), source_id=n)
    assert (path / 'metadata.jsonl').stat().st_size <= 300
    rows = records(path)
    assert all(isinstance(row, dict) for row in rows)
    manifest = json.loads((path / 'manifest.json').read_text())
    assert manifest['complete'] is False and manifest['omitted_records'] > 0


def test_explicit_sdk_recording_option_reaches_real_principal(boundary, tmp_path, monkeypatch):
    from tests.test_principal_pipeline import call
    from infinidev.tools.base.context import clear_agent_context, set_context
    agent = boundary[0]
    monkeypatch.setenv('INFINIDEV_WORKSPACE', str(tmp_path))
    set_context(agent_id=agent.agent_id, project_id=None, session_id='principal-session',
                workspace_path=str(tmp_path))
    path = tmp_path.parent / (tmp_path.name + '-capture')
    try:
        assert call(boundary, 'Literal request', record_context_dir=path) == 'respuesta'
    finally:
        clear_agent_context(agent.agent_id)
    assert (path / 'manifest.json').is_file()


def test_history_records_keep_ids_and_legacy_projection(temp_db, tool_context):
    from infinidev.db.service import (get_recent_turn_records, get_recent_turns_full,
                                     store_conversation_turn)
    store_conversation_turn(tool_context.session_id, 'assistant', 'X' * 100, 'X')
    records = get_recent_turn_records(tool_context.session_id, max_chars_per_turn=10)
    assert records[0]['id'] > 0
    assert records[0]['loader_clipped'] is True
    assert records[0]['original_bytes'] == 100
    assert [(r['role'], r['content']) for r in records] == get_recent_turns_full(
        tool_context.session_id, max_chars_per_turn=10)


def sdk_case(runtime, tool_context, monkeypatch):
    from unittest.mock import Mock
    from infinidev.config.settings import settings
    from infinidev.db.service import store_conversation_turn
    from infinidev.engine.orchestration import pipeline
    from infinidev.prompts.profiles import EffectivePromptConfiguration
    from tests.test_turn_runtime import reply
    engine, agent, script, seen = runtime
    monkeypatch.setenv('INFINIDEV_WORKSPACE', str(tool_context.workspace_path))
    agent.activate_context = Mock()
    agent.deactivate = Mock()
    monkeypatch.setattr(settings, 'TASK_ENGINE_MODE', 'task')
    monkeypatch.setattr(settings, 'GATHER_ENABLED', False)
    monkeypatch.setattr(pipeline, '_ken_turn_context', lambda *a: '')
    monkeypatch.setattr(pipeline, '_run_task_start_hook', lambda **kw: '')
    monkeypatch.setattr(pipeline, '_task_end_hook', lambda *a, **kw: '')
    monkeypatch.setattr(pipeline, '_report_turn_end_to_ken', Mock())
    kwargs = dict(agent=agent, engine=engine, hooks=Mock(), reviewer=None,
                  session_id=tool_context.session_id, user_input='NEW_PRIVATE_BODY',
                  prompt_configuration=EffectivePromptConfiguration({}), task_tools=agent.tools)
    return engine, agent, script, seen, kwargs


def test_real_principal_records_selected_history_without_changing_provider_input(
        runtime, tool_context, monkeypatch, tmp_path):
    from infinidev.db.service import store_conversation_turn
    from infinidev.engine.orchestration import pipeline
    from tests.test_turn_runtime import reply
    engine, agent, script, seen, kwargs = sdk_case(runtime, tool_context, monkeypatch)
    store_conversation_turn(tool_context.session_id, 'assistant', 'OLD_PUBLIC_CLAIM', 'Old')
    script(reply('Done.'))
    assert pipeline.run_task(**kwargs) == 'Done.'
    baseline = seen[-1]
    path = tmp_path.parent / (tmp_path.name + '-capture')
    script(reply('Done.'))
    assert pipeline.run_task(**kwargs, record_context_dir=path) == 'Done.'
    assert seen[-1][:2] == baseline[:2]
    assert {k: v for k, v in seen[-1][2].items() if not callable(v)} == {
        k: v for k, v in baseline[2].items() if not callable(v)}
    assert {k for k, v in seen[-1][2].items() if callable(v)} == {
        k for k, v in baseline[2].items() if callable(v)}
    rows = records(path)
    history = [r for r in rows if r['kind'] == 'source' and r['origin'] == 'history']
    assert history and all(r['source_id'] is not None for r in history)
    request = next(r for r in rows if r['kind'] == 'request')
    assert request['child_run_id'] == engine._last_turn_receipt['run_id']
    assert request['tool_names'] == ['record', 'finish_turn', 'update_plan']
    assert any(m['mapping_valid'] and m['source_id'] for m in request['messages'])
    dump = ''.join(f.read_text() for f in path.iterdir())
    assert 'OLD_PUBLIC_CLAIM' not in dump and 'NEW_PRIVATE_BODY' not in dump
    assert all(not key.startswith('_recording') for m in seen[-1][0] for key in m)


def test_unknown_origin_string_does_not_publish_digest(tmp_path):
    from infinidev.engine.context_recording import recording
    path = tmp_path / 'capture'
    with recording(path, workspace=tmp_path / 'workspace') as capture:
        capture.source('unrecognized', 'assistant', 'UNCLASSIFIED_PRIVATE')
    text = (path / 'metadata.jsonl').read_text()
    assert hashlib.sha256(b'UNCLASSIFIED_PRIVATE').hexdigest() not in text
    assert next(r for r in records(path) if r['kind'] == 'source')['content_sha256'] is None


def test_short_write_retains_whole_old_records_and_marks_incomplete(tmp_path, monkeypatch):
    import os
    from infinidev.engine.context_recording import recording
    path = tmp_path / 'capture'
    with recording(path, workspace=tmp_path / 'workspace') as capture:
        capture.source('history', 'assistant', 'First', source_id=1)
        previous = (path / 'metadata.jsonl').read_bytes()
        real_write = os.write
        with monkeypatch.context() as patch:
            patch.setattr(os, 'write', lambda fd, data: real_write(fd, data[:5]))
            capture.source('history', 'assistant', 'Second', source_id=2)
    assert (path / 'metadata.jsonl').read_bytes() == previous
    assert json.loads((path / 'manifest.json').read_text())['complete'] is False


def test_request_ignores_reasoning_arguments_and_unknown_roles(tmp_path):
    from infinidev.engine.context_recording import recording
    path = tmp_path / 'capture'
    with recording(path, workspace=tmp_path / 'workspace') as capture:
        capture.request('a' * 32, 1, [
            {'role': 'assistant', 'content': None, 'reasoning_content': 'PRIVATE_THOUGHTS',
             'tool_calls': [{'function': {'name': 'record', 'arguments': 'PRIVATE_ARGUMENT'}}]},
            {'role': {'password': 'ROLE_SECRET'}, 'content': 'PRIVATE_UNKNOWN'},
        ], [], mode='function_calling')
    text = (path / 'metadata.jsonl').read_text()
    assert all(value not in text for value in (
        'PRIVATE_THOUGHTS', 'PRIVATE_ARGUMENT', 'ROLE_SECRET', 'PRIVATE_UNKNOWN'))


def test_existing_event_projection_drops_bodies_and_arguments(tmp_path):
    from infinidev.engine.context_recording import recording
    path = tmp_path / 'capture'
    with recording(path, workspace=tmp_path / 'workspace') as capture:
        capture.event('a' * 32, 2, 'model_response', {
            'origin': 'native', 'finish_reason': 'tool_calls',
            'calls': [{'tool': 'Read', 'arguments': 'PRIVATE_ARGUMENT'}],
            'reasoning_content': 'PRIVATE_THOUGHTS', 'raw_content': 'PRIVATE_BODY'})
        capture.event('a' * 32, 2, 'tool_validation_rejected', {
            'executed': False, 'calls': [{'tool': 'Read',
                'errors': [{'field': 'tool_name', 'type': 'tool_not_offered'}]}]})
    text = (path / 'metadata.jsonl').read_text()
    assert all(value not in text for value in ('PRIVATE_ARGUMENT', 'PRIVATE_THOUGHTS', 'PRIVATE_BODY'))
    rejected = next(r for r in records(path) if r.get('event') == 'tool_validation_rejected')
    assert rejected['executed'] is False
    assert rejected['calls'][0]['errors'][0]['type'] == 'tool_not_offered'


@pytest.mark.parametrize('manual', [False, True])
def test_runtime_modes_record_request_without_leaking_metadata(runtime, monkeypatch, tmp_path, manual):
    from types import SimpleNamespace
    from infinidev.engine.context_recording import recording
    from infinidev.engine.turns import runtime as rt
    from tests.test_turn_runtime import reply
    engine, agent, script, seen = runtime
    monkeypatch.setattr(rt, 'get_model_capabilities', lambda: SimpleNamespace(
        supports_function_calling=not manual, supports_vision=False))
    path = tmp_path.parent / (tmp_path.name + '-capture')
    script(reply('Done.'))
    with recording(path, workspace=tmp_path):
        assert engine.run_turn(agent, 'Literal').status == 'answered'
    request = next(r for r in records(path) if r['kind'] == 'request')
    assert request['mode'] == ('manual' if manual else 'function_calling')
    assert request['messages'][0]['mapping_valid'] is True
    assert all('_recording_source' not in m for m in seen[-1][0])


def test_two_children_are_linked_and_next_sdk_invocation_is_not_recorded(
        runtime, tool_context, monkeypatch, tmp_path):
    from infinidev.engine.context_recording import current_recorder
    from infinidev.engine.orchestration import pipeline
    from infinidev.engine.user_hooks import UserHookEvent
    from tests.test_turn_runtime import reply
    engine, agent, script, seen, kwargs = sdk_case(runtime, tool_context, monkeypatch)
    monkeypatch.setattr(pipeline, '_task_end_hook', lambda event, **kw:
                        'Follow-up' if event == UserHookEvent.TASK_END_INSTRUCTION else '')
    path = tmp_path.parent / (tmp_path.name + '-capture')
    script(reply('First.'), reply('Second.'))
    assert pipeline.run_task(**kwargs, record_context_dir=path) == 'Second.'
    requests = [r for r in records(path) if r['kind'] == 'request']
    assert len(requests) == 2
    assert requests[0]['child_run_id'] != requests[1]['child_run_id']
    assert requests[1]['messages'][1]['mapping_valid'] is True
    assert current_recorder() is None
    prior = (path / 'metadata.jsonl').read_bytes()
    script(reply('Third.'), reply('Fourth.'))
    assert pipeline.run_task(**kwargs) == 'Fourth.'
    assert (path / 'metadata.jsonl').read_bytes() == prior


def test_rejection_and_cancellation_are_host_facts_not_source_bodies(runtime, tmp_path):
    from infinidev.engine.context_recording import recording, current_recorder
    from tests.test_turn_runtime import reply, tc
    engine, agent, script, seen = runtime
    def cancel():
        engine.cancel()
        return reply('Cancelled.')
    script(reply('', tc('unknown', text='PRIVATE_ARGUMENT')), cancel)
    path = tmp_path.parent / (tmp_path.name + '-capture')
    with recording(path, workspace=tmp_path):
        assert engine.run_turn(agent, 'PRIVATE_TASK').status == 'cancelled'
    assert current_recorder() is None
    rejected = next(r for r in records(path) if r.get('event') == 'tool_validation_rejected')
    assert rejected['executed'] is False
    assert rejected['calls'][0]['errors'][0]['type'] == 'tool_not_offered'
    closed = next(r for r in records(path) if r.get('event') == 'turn_closed')
    assert closed['status'] == 'cancelled'
    data = (path / 'metadata.jsonl').read_text()
    assert 'PRIVATE_TASK' not in data and 'PRIVATE_ARGUMENT' not in data


def test_changed_message_is_not_attributed_to_old_source(tmp_path):
    from infinidev.engine.context_recording import recording
    path = tmp_path / 'capture'
    with recording(path, workspace=tmp_path / 'workspace') as capture:
        message = {'role': 'user', 'content': 'Before'}
        source = capture.source('literal_input', 'user', message['content'])
        capture.attach(message, source)
        message['content'] = 'After'
        capture.request('a' * 32, 2, [message], [], mode='manual')
    request = next(r for r in records(path) if r['kind'] == 'request')
    assert request['messages'][0]['mapping_valid'] is False
    assert request['messages'][0]['source_id'] is None
    assert request['messages'][0]['assembly_ancestor'] == source


def test_recording_close_failure_preserves_original_error(tmp_path, monkeypatch):
    import os
    from infinidev.engine.context_recording import recording, current_recorder
    path = tmp_path / 'capture'
    original_close = os.close
    recorder = None
    with pytest.raises(ValueError, match='original task error'):
        with monkeypatch.context() as patch:
            with recording(path, workspace=tmp_path / 'workspace') as recorder:
                def fail_metadata_close(fd):
                    if fd == recorder._fd:
                        raise OSError('simulated recording error')
                    return original_close(fd)
                patch.setattr(os, 'close', fail_metadata_close)
                raise ValueError('original task error')
    original_close(recorder._fd)  # The simulated close intentionally left this descriptor open.
    assert current_recorder() is None


def test_native_fallback_records_actual_manual_system(runtime, tmp_path):
    from infinidev.engine.context_recording import recording
    from tests.test_turn_runtime import reply
    engine, agent, script, seen = runtime
    script(RuntimeError('does not support tools'), reply('Done.'))
    path = tmp_path.parent / (tmp_path.name + '-capture')
    with recording(path, workspace=tmp_path):
        assert engine.run_turn(agent, 'Literal').status == 'answered'
    requests = [r for r in records(path) if r['kind'] == 'request']
    assert [r['mode'] for r in requests] == ['function_calling', 'manual']
    assert requests[-1]['messages'][0]['mapping_valid'] is True
    assert '_recording_source' not in seen[-1][0][0]


def test_history_selected_view_reports_clipping_stage(runtime, monkeypatch, tmp_path, tool_context):
    from infinidev.db.service import store_conversation_turn
    from infinidev.engine.context_recording import recording
    from infinidev.engine.session_context import render_session_context
    store_conversation_turn(tool_context.session_id, 'assistant', 'λ' * 2000, 'Old')
    monkeypatch.setattr('infinidev.engine.loop.model_context.get_model_context_window', lambda _: 2048)
    path = tmp_path.parent / (tmp_path.name + '-capture')
    with recording(path, workspace=tmp_path):
        rendered = render_session_context(tool_context.session_id, tool_names=())
    row = next(r for r in records(path) if r['kind'] == 'source' and r['origin'] == 'history')
    assert row['phase'] == 'history_selected_view'
    assert row['loader_clipped'] is True and row['renderer_clipped'] is True
    assert row['clipping']['loader_limit_chars'] == 512
    assert row['clipping']['renderer_head_bytes'] > 0
    assert row['original_content_sha256'] is None
    assert 'omitted' in rendered


def test_real_pressure_keeps_recording_keys_out_of_provider(runtime, tmp_path):
    from infinidev.engine.context_recording import recording
    from tests.test_turn_runtime import reply, tc
    engine, agent, script, seen = runtime
    script(reply('', tc('record', text='PUBLIC_TOOL_DATA ' * 2000), tokens=900_000),
           reply('', tc('finish_turn', message='Done.')))
    path = tmp_path.parent / (tmp_path.name + '-capture')
    with recording(path, workspace=tmp_path):
        assert engine.run_turn(agent, 'Literal').status == 'answered'
    assert all(not k.startswith('_recording') for message in seen[-1][0] for k in message)
    assert 'PUBLIC_TOOL_DATA' not in (path / 'metadata.jsonl').read_text()


def test_copied_context_cannot_reuse_closed_recorder(tmp_path, monkeypatch):
    import contextvars
    import os
    from infinidev.engine.context_recording import recording, current_recorder
    path = tmp_path / 'capture'
    with recording(path, workspace=tmp_path / 'workspace') as capture:
        saved_context = contextvars.copy_context()
    assert saved_context.run(current_recorder) is None
    writes = []
    monkeypatch.setattr(os, 'write', lambda fd, body: writes.append(fd) or len(body))
    capture.source('literal_input', 'user', 'Late callback')
    assert writes == []


def test_copied_context_on_another_thread_is_not_an_implicit_opt_in(tmp_path):
    import contextvars
    import threading
    from infinidev.engine.context_recording import recording, current_recorder
    with recording(tmp_path / 'capture', workspace=tmp_path / 'workspace'):
        copied = contextvars.copy_context()
        observed = []
        worker = threading.Thread(target=lambda: observed.append(copied.run(current_recorder)))
        worker.start()
        worker.join(timeout=2)
        assert not worker.is_alive()
        assert observed == [None]


def test_manual_append_with_unmapped_prefix_never_hashes_unknown_system(tmp_path):
    from infinidev.engine.context_recording import recording
    from infinidev.engine.turns.runtime import _append_manual_tools
    path = tmp_path / 'capture'
    private_system = 'UNCLASSIFIED_PRIVATE_SYSTEM'
    with recording(path, workspace=tmp_path / 'workspace'):
        rendered = _append_manual_tools(private_system, [])
    data = (path / 'metadata.jsonl').read_text()
    assert hashlib.sha256(rendered.encode()).hexdigest() not in data
    bundles = [r for r in records(path) if r['kind'] == 'bundle']
    assert bundles[-1]['parts_known'] is False
    assert bundles[-1]['content_sha256'] is None


def test_sdk_checks_workspace_environment_override_before_start(boundary, tmp_path, monkeypatch):
    from tests.test_principal_pipeline import call
    monkeypatch.setenv('INFINIDEV_WORKSPACE', str(tmp_path.parent))
    path = tmp_path.parent / (tmp_path.name + '-capture')
    with pytest.raises(ValueError, match='outside the workspace'):
        call(boundary, 'Literal request', record_context_dir=path)
    assert not path.exists()
    boundary[1].run_turn.assert_not_called()


def test_sdk_ignores_stale_context_workspace_before_activation(boundary, tmp_path, monkeypatch):
    from infinidev.agents.base import InfinidevAgent
    from infinidev.tools.base.context import clear_agent_context, set_context
    from tests.test_principal_pipeline import call
    agent = boundary[0]
    previous = tmp_path / 'previous'
    current = tmp_path / 'current'
    previous.mkdir()
    current.mkdir()
    monkeypatch.delenv('INFINIDEV_WORKSPACE', raising=False)
    monkeypatch.chdir(current)
    set_context(agent_id=agent.agent_id, project_id=None, session_id='principal-session',
                workspace_path=str(previous))
    agent.activate_context.side_effect = lambda **kw: InfinidevAgent.activate_context(agent, **kw)
    path = current / 'capture'
    try:
        with pytest.raises(ValueError, match='outside the workspace'):
            call(boundary, 'Literal request', record_context_dir=path)
        assert not path.exists()
        agent.activate_context.assert_not_called()
        boundary[1].run_turn.assert_not_called()
    finally:
        clear_agent_context(agent.agent_id)
