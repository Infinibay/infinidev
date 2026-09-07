"""Browser authentication, durable sessions and the headless execution contract."""

from __future__ import annotations

import threading
import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture
def web_client(temp_db, tmp_path, monkeypatch):
    from infinidev.config.settings import settings
    from infinidev.server.app import create_app
    from infinidev.tools import permission

    for key, value in settings.model_dump().items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(permission, "_permission_handler", permission._permission_handler)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("infinidev.config.settings.SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.setattr("infinidev.server.bootstrap.bootstrap_runtime", lambda: None)
    from infinidev.db.service import init_db

    init_db()
    app = create_app(token="test-access-token", initialize=False)
    with TestClient(app) as client:
        client.headers["Authorization"] = "Bearer test-access-token"
        yield client, app


def test_api_authenticates_and_never_returns_model_credentials(web_client, monkeypatch):
    from infinidev.config.settings import settings

    client, _ = web_client
    monkeypatch.setattr(settings, "LLM_API_KEY", "sk-test-credential-that-must-stay-private")
    response = client.get("/api/settings")
    assert response.status_code == 200
    assert settings.LLM_API_KEY not in response.text
    assert response.json()["values"]["THINKING_BUDGET_TOKENS"] == settings.THINKING_BUDGET_TOKENS
    assert "THINKING_BUDGET_TOKENS" not in response.json()["secrets"]
    assert client.get("/api/info", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.get("/api/info", headers={"Origin": "https://untrusted.example"}).status_code == 403


def test_session_survives_disconnect_and_reconnect_replays_current_state(web_client):
    client, app = web_client
    session_id = client.post("/api/sessions", json={}).json()["session_id"]
    session = app.state.runtime.session(session_id)
    session.add_message("Infinidev", "Persisted answer", "agent")
    with client.websocket_connect(f"/ws?session_id={session_id}",
                                  subprotocols=["infinidev", "token.test-access-token"]) as ws:
        snapshot = ws.receive_json()
        assert snapshot["type"] == "snapshot"
        assert snapshot["session"]["messages"][-1]["text"] == "Persisted answer"
    assert not session.closed
    with client.websocket_connect(f"/ws?session_id={session_id}",
                                  subprotocols=["infinidev", "token.test-access-token"]) as ws:
        assert ws.receive_json()["session"]["messages"][-1]["text"] == "Persisted answer"


def test_permission_waits_for_explicit_answer_and_cancel_denies(web_client):
    client, app = web_client
    session_id = client.post("/api/sessions", json={}).json()["session_id"]
    session = app.state.runtime.session(session_id)
    results = []
    worker = threading.Thread(target=lambda: results.append(session.ask(
        "Run command?", "permission", details="pytest", tool="execute_command")))
    worker.start()
    deadline = time.monotonic() + 2
    while not session.snapshot()["pending"] and time.monotonic() < deadline:
        time.sleep(0.01)
    request_id = session.snapshot()["pending"][0]["request_id"]
    assert worker.is_alive()
    assert client.post(f"/api/sessions/{session_id}/answers", json={
        "request_id": request_id, "text": "allow",
    }).status_code == 200
    worker.join(2)
    assert results == ["allow"]
    assert client.post(f"/api/sessions/{session_id}/answers", json={
        "request_id": request_id, "text": "allow",
    }).status_code == 409


def test_files_are_scoped_and_saved_with_conflict_detection(web_client, tmp_path):
    client, _ = web_client
    path = tmp_path / "source.py"
    path.write_text("old\n")
    original = client.get("/api/files/read", params={"path": "source.py"}).json()
    path.write_text("external edit\n")
    response = client.put("/api/files/write", json={
        "path": "source.py", "text": "new\n", "revision": original["revision"],
    })
    assert response.status_code == 409
    assert path.read_text() == "external edit\n"
    assert client.get("/api/files/read", params={"path": "../outside"}).status_code == 400


def test_model_effort_options_come_from_selected_model(web_client, monkeypatch):
    from infinidev.config.settings import settings

    client, _ = web_client
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_MODEL", "openai/gpt-6-astra")
    response = client.get("/api/models").json()
    assert response["effort"]["choices"] == ["low", "medium", "high", "xhigh", "max"]


def test_conversation_thread_keeps_replies_and_is_scoped_to_the_session(web_client):
    import hashlib
    import json

    from infinidev.engine.team.store import TeamStore

    client, app = web_client
    session_id = client.post("/api/sessions", json={}).json()["session_id"]
    other = client.post("/api/sessions", json={}).json()["session_id"]
    key = json.dumps([1, str(app.state.runtime.root), session_id])
    store = TeamStore(hashlib.sha256(key.encode()).hexdigest())

    def conversation(state, emit):
        question = emit("message", "a", "Question", recipient="b", message_type="request")
        emit("message", "b", "Answer", recipient="a", reply_to=question,
             thread_id=question, message_type="reply")
        return question

    question = store.update(conversation)
    response = client.get(f"/api/sessions/{session_id}/threads/{question}")
    assert response.status_code == 200
    assert [e["content"] for e in response.json()["events"]] == ["Question", "Answer"]
    assert client.get(f"/api/sessions/{other}/threads/{question}").status_code == 404


def _wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def test_cancel_resolves_all_questions_without_approving(web_client):
    client, app = web_client
    session_id = client.post('/api/sessions', json={}).json()['session_id']
    session = app.state.runtime.session(session_id)
    results = []
    worker = threading.Thread(target=lambda: results.append(session.ask('Allow?', 'permission')))
    worker.start()
    _wait_until(lambda: session.snapshot()['pending'])
    client.post(f'/api/sessions/{session_id}/cancel', json={})
    worker.join(2)
    assert results == [None]
    assert session.snapshot()['pending'] == []
    assert app.state.runtime.permission('shell', 'Execute?', '') is False


def test_websocket_rejects_missing_token_and_foreign_session(web_client, tmp_path):
    from starlette.websockets import WebSocketDisconnect
    from infinidev.db.service import register_session

    client, _ = web_client
    register_session('foreign', str(tmp_path.parent))
    assert client.get('/api/sessions/foreign').status_code == 404
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/ws?session_id=foreign'):
            pass
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect('/ws?session_id=foreign', subprotocols=[
            'infinidev', 'token.test-access-token',
        ]):
            pass
    assert client.get('/api/info', headers={'Host': 'attacker.example'}).status_code == 403


def test_stream_keeps_agent_attribution_and_persists_final_text(web_client):
    from types import SimpleNamespace
    from infinidev.db.service import get_session_messages

    client, app = web_client
    session = app.state.runtime.session(client.post('/api/sessions').json()['session_id'])
    session.engine = SimpleNamespace(_team_runtime=SimpleNamespace(store=SimpleNamespace(
        snapshot=lambda: {'agents': {'w_alice': {'name': 'Alice'}}})))
    session.on_engine_event('loop_thinking_chunk', 1, 'w_alice', {'text': 'Evidence '})
    session.on_engine_event('loop_thinking_chunk', 1, 'w_alice', {'text': 'first.'})
    session.on_engine_event('loop_stream_status', 1, 'w_alice', {'phase': 'done'})
    session.engine = None
    messages = get_session_messages(session.session_id)
    assert len(messages) == 1
    assert messages[0]['sender'] == 'Alice'
    assert messages[0]['text'] == 'Evidence first.'
    assert messages[0]['streaming'] is False


def test_turns_are_serialized_idempotent_and_cancel_engine(web_client, monkeypatch):
    from types import SimpleNamespace
    import infinidev.agents.base
    import infinidev.engine.loop
    import infinidev.engine.orchestration
    import infinidev.engine.analysis.review_engine

    monkeypatch.setattr('infinidev.config.settings.reload_all', lambda: None)
    monkeypatch.setattr('infinidev.cli.session_resume.begin_resumed_session', lambda _: None)
    monkeypatch.setattr(infinidev.agents.base, 'InfinidevAgent', lambda **_: SimpleNamespace(
        activate_context=lambda **_: None, deactivate=lambda: None))
    monkeypatch.setattr(infinidev.engine.analysis.review_engine, 'ReviewEngine', lambda: None)

    class Engine:
        def __init__(self):
            self.stopped = threading.Event()

        def cancel(self):
            self.stopped.set()

    monkeypatch.setattr(infinidev.engine.loop, 'LoopEngine', Engine)
    entered = []
    second_release = threading.Event()

    def run_task(**kwargs):
        entered.append(kwargs['user_input'])
        kwargs['hooks'].on_phase('execute')
        if len(entered) == 1:
            assert kwargs['engine'].stopped.wait(3)
        else:
            assert second_release.wait(3)
            kwargs['hooks'].notify_stream_chunk('Infinidev', 'One answer')
            kwargs['hooks'].notify_stream_end('Infinidev')
            kwargs['hooks'].mark_reply_shown()
        return 'One answer'

    monkeypatch.setattr(infinidev.engine.orchestration, 'run_task', run_task)
    client, app = web_client
    first = client.post('/api/sessions').json()['session_id']
    second = client.post('/api/sessions').json()['session_id']
    payload = {'text': 'first task', 'client_id': 'request-1'}
    assert client.post(f'/api/sessions/{first}/messages', json=payload).status_code == 200
    _wait_until(lambda: entered)
    duplicate = client.post(f'/api/sessions/{first}/messages', json=payload).json()
    assert duplicate['duplicate']
    client.post(f'/api/sessions/{second}/messages', json={
        'text': 'second task', 'client_id': 'request-2',
    })
    assert entered == ['first task']
    assert client.patch('/api/settings', json={'updates': {'TEAM_MAX_WORKERS': 2}}).status_code == 409
    client.post(f'/api/sessions/{first}/cancel')
    _wait_until(lambda: len(entered) == 2)
    second_release.set()
    _wait_until(lambda: not app.state.runtime.session(second).busy)
    messages = app.state.runtime.session(second).snapshot()['messages']
    assert [m['text'] for m in messages if m['speaker'] == 'Infinidev'] == ['One answer']


def test_notes_preserve_user_authorship_and_do_not_claim_worker_liveness(web_client):
    import hashlib
    import json
    from infinidev.engine.team.store import TeamStore

    client, app = web_client
    session_id = client.post('/api/sessions').json()['session_id']
    key = json.dumps([1, str(app.state.runtime.root), session_id])
    store = TeamStore(hashlib.sha256(key.encode()).hexdigest())
    store.update(lambda state, emit: state.update(agents={
        'w_test': {'id': 'w_test', 'name': 'Mara', 'role': 'Researcher', 'status': 'running'},
    }, tickets={}))
    response = client.post(f'/api/sessions/{session_id}/notes', json={'content': 'Check the baseline.'})
    assert response.json()['author'] == 'user'
    snapshot = client.get(f'/api/sessions/{session_id}/team').json()
    assert snapshot['live'] is False
    assert snapshot['events'][0]['author_label'] == 'You'
    assert snapshot['events'][0]['content'] == 'Check the baseline.'


def test_file_scope_follows_symlinks_and_permission_denials(web_client, tmp_path, monkeypatch):
    client, _ = web_client
    (tmp_path / 'escape').symlink_to(tmp_path.parent, target_is_directory=True)
    assert client.get('/api/files/read?path=escape/private.txt').status_code == 400
    assert client.get('/api/files/read?path=.infinidev/settings.json').status_code == 403
    (tmp_path / 'safe.txt').write_text('unchanged')
    original = client.get('/api/files/read?path=safe.txt').json()
    monkeypatch.setattr('infinidev.tools.base.permissions.check_file_permission', lambda *_: 'Denied')
    response = client.put('/api/files/write', json={**original, 'text': 'overwrite'})
    assert response.status_code == 403
    assert (tmp_path / 'safe.txt').read_text() == 'unchanged'


def test_tool_history_round_trips_between_web_and_terminal(web_client):
    from infinidev.db.service import get_session_messages, store_session_message

    client, app = web_client
    session = app.state.runtime.create_session()
    store_session_message(session.session_id, {
        'sender': 'Tool', 'type': 'tool_call', 'text': 'Read project notes',
        'tool_name': 'read_file', 'args': {'path': 'CONTINUE.md'},
        'result': 'Prior experiment results', 'running': False, 'error': '',
    })
    del app.state.runtime.sessions[session.session_id]
    session = app.state.runtime.session(session.session_id)
    restored = session.snapshot()['messages'][0]
    assert restored['kind'] == 'tool'
    assert restored['data']['tool_arguments'] == {'path': 'CONTINUE.md'}
    assert restored['data']['tool_result_full'] == 'Prior experiment results'
    session.on_engine_event('loop_tool_start', 1, 'root', {
        'tool_run_id': 'foreground', 'tool_name': 'execute_command',
        'tool_arguments': {'command': 'pytest'},
    })
    session.on_engine_event('loop_tool_output', 1, 'root', {
        'tool_run_id': 'foreground', 'chunk': 'Collected 10 tests\n',
    })
    assert session.snapshot()['messages'][-1]['data']['tool_result_full'] == 'Collected 10 tests\n'
    session.on_engine_event('loop_tool_call', 1, 'root', {
        'tool_run_id': 'foreground', 'tool_name': 'execute_command',
        'tool_arguments': {'command': 'pytest'}, 'tool_result_full': '10 passed',
    })
    ledger = get_session_messages(session.session_id)
    assert len(ledger) == 2
    assert ledger[-1]['type'] == 'tool_call'
    assert ledger[-1]['result'] == '10 passed'
    assert ledger[-1]['running'] is False


def test_non_utf8_files_cannot_be_silently_corrupted_in_the_editor(web_client, tmp_path):
    client, _ = web_client
    (tmp_path / 'legacy.txt').write_bytes(b'caf\xe9')
    response = client.get('/api/files/read?path=legacy.txt').json()
    assert response['binary'] is True
    assert response['text'] == ''


def test_retry_is_idempotent_even_when_request_is_outside_browser_history(web_client):
    client, app = web_client
    session = app.state.runtime.create_session()
    session.add_message('You', 'Already accepted', 'user', client_id='old-request')
    session.messages.clear()
    result = client.post(f'/api/sessions/{session.session_id}/messages', json={
        'text': 'Already accepted', 'client_id': 'old-request',
    }).json()
    assert result['duplicate'] is True
    assert session.future is None


def test_provider_switch_clears_old_key_and_uses_native_effort(web_client, monkeypatch):
    from infinidev.config.settings import settings

    client, _ = web_client
    monkeypatch.setattr(settings, 'LLM_PROVIDER', 'openai')
    monkeypatch.setattr(settings, 'LLM_API_KEY', 'sk-old-provider-private-key')
    response = client.patch('/api/settings', json={'updates': {
        'LLM_PROVIDER': 'anthropic', 'LLM_MODEL': 'claude-sonnet-4-6',
        'THINKING_BUDGET': 'max',
    }})
    assert response.status_code == 200
    assert settings.LLM_API_KEY == ''
    assert settings.LLM_MODEL == 'anthropic/claude-sonnet-4-6'
    assert settings.THINKING_BUDGET == 'max'
    assert client.patch('/api/settings', json={'updates': {
        'THINKING_BUDGET': 'ultra',
    }}).status_code == 400
