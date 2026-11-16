"""Explicit tool controls stay attached to literal tasks through frontend admission."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tests.test_cli_single_prompt import cli_main


def bare_app():
    from infinidev.ui.app import InfinidevApp
    from infinidev.ui.attachments import AttachmentTray
    app = object.__new__(InfinidevApp)
    app._autocomplete = Mock()
    app._permission_waiting = False
    app._permission_event = None
    app._engine_running = False
    app._context_admission_pending = None
    app._context_admission_state = {}
    app._context_admission_compact_button = SimpleNamespace(handler=None)
    app._context_admission_switch_button = SimpleNamespace(handler=None)
    app._context_admission_cancel_button = SimpleNamespace(handler=None)
    app._analysis_waiting = False
    app._plan_review_waiting = False
    app.active_dialog = None
    app.agent = SimpleNamespace(tools=[SimpleNamespace(name='read_file')])
    app.app = SimpleNamespace(layout=Mock())
    app.attachment_tray = AttachmentTray()
    app.add_message = Mock()
    app.invalidate = Mock()
    app._start_engine_task = Mock()
    app._resolve_attachments = Mock(side_effect=AssertionError('Do not rewrite opaque body'))
    app.handle_command = Mock(side_effect=AssertionError('Do not reroute literal body'))
    app._execute_shell_command = Mock(side_effect=AssertionError('No local shell'))
    return app


def admission(monkeypatch):
    monkeypatch.setattr('infinidev.config.llm.get_litellm_params', lambda: {'model':'small'})
    monkeypatch.setattr('infinidev.config.providers.get_provider',
                        lambda _: SimpleNamespace(prefix='test/', static_models=[]))
    monkeypatch.setattr('infinidev.ui.context_admission.find_context_admission',
                        lambda **kwargs: SimpleNamespace(active_window=80_000,
                                                         replacement_model=None))


def test_classic_command_returns_one_atomic_literal_request():
    from infinidev.cli.commands import handle_command
    from infinidev.cli.tool_commands import ToolRequest
    body = '  λ\n!opaque -- /exit\n'
    request = handle_command('/tools run read_file -- '+body)
    assert isinstance(request, ToolRequest)
    assert request.names == ('read_file',)
    assert request.body == body
    assert handle_command('/tools') == 'tools'


@pytest.mark.parametrize('body', ['!literal command', '/init is opaque', 'image.png\nλ -- x'])
def test_tui_routes_grant_without_shell_slash_or_attachment_rewrite(body):
    app = bare_app()
    app._request_context_admission = Mock(return_value=False)
    app._handle_submit('/tools run read_file -- '+body)
    assert app._start_engine_task.call_args.args == (body, [])
    assert tuple(app._start_engine_task.call_args.kwargs['task_tools']) == tuple(app.agent.tools)
    app._resolve_attachments.assert_not_called()
    app.handle_command.assert_not_called()
    app._execute_shell_command.assert_not_called()
    app.add_message.assert_any_call('You', body, 'user')


def test_pending_admission_cannot_be_overwritten_and_cancel_discards_grant(monkeypatch):
    admission(monkeypatch)
    app = bare_app()
    app._handle_submit('/tools run read_file -- first body')
    pending = app._context_admission_pending
    assert pending[0] == 'first body'
    assert tuple(pending[2]) == tuple(app.agent.tools)
    old_handler = app._context_admission_compact_button.handler
    app._handle_submit('/tools run -- second body')
    app._handle_submit('ordinary second body')
    assert app._context_admission_pending is pending
    app._start_engine_task.assert_not_called()
    app._cancel_context_admission(expected_request=pending)
    assert app._context_admission_pending is None
    old_handler()
    app._start_engine_task.assert_not_called()
    app._request_context_admission = Mock(return_value=False)
    app._resolve_attachments = Mock(return_value=('next ordinary', []))
    app._handle_submit('next ordinary')
    app._start_engine_task.assert_called_once_with('next ordinary', [])


def test_old_admission_callback_cannot_start_a_new_pending_request(monkeypatch):
    admission(monkeypatch)
    app = bare_app()
    app._handle_submit('/tools run read_file -- first')
    first = app._context_admission_pending
    old_handler = app._context_admission_compact_button.handler
    app._cancel_context_admission(expected_request=first)
    app._handle_submit('/tools run -- second')
    second = app._context_admission_pending
    old_handler()
    assert app._context_admission_pending is second
    app._start_engine_task.assert_not_called()
    app._context_admission_compact_button.handler()
    assert app._start_engine_task.call_args.args[:2] == ('second', [])
    assert tuple(app._start_engine_task.call_args.args[2]) == ()


def test_invalid_tool_name_does_not_consume_staged_attachments():
    app = bare_app()
    item = object()
    app.attachment_tray.add(item)
    app._handle_submit('/tools run write_file -- literal task')
    assert app.attachment_tray.items == [item]
    app._start_engine_task.assert_not_called()


def test_grant_cannot_change_active_task():
    app = bare_app()
    app._engine_running = True
    app._handle_submit('/tools run read_file -- task')
    app._start_engine_task.assert_not_called()
    app.handle_command.assert_not_called()


@pytest.mark.parametrize('names', ['read_file', ''])
def test_classic_repl_consumes_grant_for_exactly_one_submission(
        monkeypatch, temp_db, cli_main, names):
    cli = cli_main
    from infinidev.cli import classic_renderer as renderer
    from infinidev.db.service import register_session
    from infinidev.engine import orchestration
    from infinidev.tools import permission

    register_session('classic-grant')
    read = SimpleNamespace(name='read_file')
    agent = SimpleNamespace(tools=[read])
    command = '/tools run' + (' '+names if names else '') + ' -- /literal λ -- body'
    inputs = iter([command, 'next ordinary'])
    class Session:
        def prompt(self, *args):
            try:
                return next(inputs)
            except StopIteration:
                raise EOFError
    monkeypatch.setattr(cli, 'PromptSession', lambda **kwargs: Session())
    monkeypatch.setattr(cli, '_bootstrap_single_prompt_runtime', lambda: None)
    monkeypatch.setattr(cli, '_cleanup_classic_runtime', lambda *a, **kw: None)
    monkeypatch.setattr(cli, '_resolve_classic_session', lambda *args: ('classic-grant', [], 'test'))
    monkeypatch.setattr(cli, 'InfinidevAgent', lambda **kwargs: agent)
    monkeypatch.setattr(cli, 'LoopEngine', lambda: SimpleNamespace(is_cancelled=False))
    monkeypatch.setattr(cli, 'ReviewEngine', lambda: object())
    monkeypatch.setattr(cli, '_render_final', Mock())
    monkeypatch.setattr(cli, '_record_classic_transcript', Mock())
    monkeypatch.setattr('infinidev.db.service.store_conversation_turn', Mock())
    monkeypatch.setattr(renderer, 'ClassicRenderer', lambda status: SimpleNamespace(
        subscribe=lambda: None, unsubscribe=lambda: None))
    monkeypatch.setattr(renderer, 'run_with_live_status', lambda engine, fn, status, **kwargs: fn())
    monkeypatch.setattr(permission, 'set_permission_handler', lambda handler: None)
    run = Mock(return_value='Done')
    monkeypatch.setattr(orchestration, 'run_task', run)
    cli._run_main(False, True, None, False, False)
    assert run.call_count == 2
    assert run.call_args_list[0].kwargs['user_input'] == '/literal λ -- body'
    assert tuple(run.call_args_list[0].kwargs['task_tools']) == ((read,) if names else ())
    assert run.call_args_list[1].kwargs['user_input'] == 'next ordinary'
    assert 'task_tools' not in run.call_args_list[1].kwargs


def test_headless_control_command_is_not_silently_sent_as_unrestricted_prompt(monkeypatch, cli_main):
    cli = cli_main
    monkeypatch.setattr(cli, '_bootstrap_single_prompt_runtime',
                        Mock(side_effect=AssertionError('No bootstrap for unsupported control')))
    with pytest.raises(ValueError, match='interactive'):
        cli._run_single_prompt('/tools run read_file -- task')


def test_tool_name_change_during_admission_cannot_widen_frozen_selection(monkeypatch):
    admission(monkeypatch)
    app = bare_app()
    app._handle_submit('/tools run read_file -- private candidate')
    pending = app._context_admission_pending
    app.agent.tools[0].name = 'delete_file'
    app._context_admission_compact_button.handler()
    app._start_engine_task.assert_not_called()
    assert app._context_admission_pending is None
