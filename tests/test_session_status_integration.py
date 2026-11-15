"""Host lifecycle observations use real principal boundaries with scripted providers."""

from __future__ import annotations

import hashlib
import json
import time
from unittest.mock import Mock

import pytest

from infinidev.engine import session_status as status
from infinidev.engine.orchestration import pipeline
from infinidev.engine.turns.models import TurnResult, TurnState
from tests.test_principal_pipeline import boundary, call
from tests.test_turn_runtime import reply, runtime, tc


@pytest.fixture(autouse=True)
def principal_workspace(monkeypatch, workspace_dir):
    monkeypatch.setenv("INFINIDEV_WORKSPACE", str(workspace_dir))


def test_parent_spans_hook_children_and_hashes_published_final(boundary, monkeypatch, workspace_dir):
    from infinidev.engine.user_hooks import UserHookEvent

    agent, engine, hooks = boundary
    agent.workspace_path = str(workspace_dir)
    parents = []

    def run(*args, **kwargs):
        pub = status.current_observer()
        assert pub is not None
        parents.append(pub)
        pub.record("working", "child_run_closed", child_run_id=str(len(parents)) * 32)
        assert pub.flush(2)
        observed = status.read_status(workspace_dir, "principal-session")
        assert observed["state"] == "working"
        return TurnResult(message="secret-final", status="answered", state=TurnState(),
                          metrics={"observed_iterations": 1, "observed_tool_calls": 1})

    engine.run_turn.side_effect = run
    monkeypatch.setattr(pipeline, "_task_end_hook", lambda event, **kw:
                        "continue" if event == UserHookEvent.TASK_END_INSTRUCTION else "")
    monkeypatch.setattr("infinidev.config.secrets.redact", lambda text: "redacted-final")
    assert call(boundary) == "secret-final"
    assert len(parents) == 2 and parents[0] is parents[1]
    result = status.read_status(workspace_dir, "principal-session",
                                expected_turn=parents[0].host_turn_id)
    assert result["state"] == "closed"
    assert result["observation"]["final_message_sha256"] == hashlib.sha256(
        b"redacted-final").hexdigest()
    hooks.notify.assert_called_with("Infinidev", "redacted-final", "agent")
    assert status.current_observer() is None


def test_publication_failure_never_manufactures_parent_close(boundary, workspace_dir):
    boundary[0].workspace_path = str(workspace_dir)
    seen = []
    def run(*args, **kwargs):
        seen.append(status.current_observer())
        return TurnResult(message="answer", status="answered", state=TurnState())
    boundary[1].run_turn.side_effect = run
    boundary[2].notify.side_effect = RuntimeError("PRIVATE_ERROR")
    with pytest.raises(RuntimeError, match="PRIVATE_ERROR"):
        call(boundary)
    result = status.read_status(workspace_dir, "principal-session",
                                expected_turn=seen[0].host_turn_id)
    assert result["state"] == "unknown"
    assert result["last_known"]["final_message_sha256"] is None
    assert "PRIVATE_ERROR" not in json.dumps(result)


@pytest.mark.parametrize("failure", [False, True])
def test_observer_io_failure_preserves_principal_result(boundary, workspace_dir, monkeypatch, failure):
    boundary[0].workspace_path = str(workspace_dir)
    monkeypatch.setattr(status.StatusPublisher, "_write", Mock(side_effect=OSError("PRIVATE")))
    if failure:
        boundary[1].run_turn.side_effect = ValueError("runtime failure")
    result = call(boundary)
    assert result == ("Conversation failed while running the agent." if failure else "respuesta")
    assert status.read_status(workspace_dir, "principal-session")["state"] == "unknown"


@pytest.mark.parametrize("kind", ["permission", "stdin"])
def test_blocking_input_exposes_only_wait_metadata(tmp_path, kind):
    from infinidev.tools import permission, stdin_prompt

    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    def handler(*args):
        assert pub.flush(2)
        result = status.read_status(tmp_path, "session")
        assert result["state"] == "awaiting_input"
        assert "PRIVATE" not in json.dumps(result)
        return True if kind == "permission" else "PRIVATE_REPLY"
    try:
        with status.bind_observer(pub):
            pub.record("tool", "tool_started", tool_name="execute_command")
            if kind == "permission":
                permission.set_permission_handler(handler)
                assert permission.request_permission("execute_command", "PRIVATE", "PRIVATE")
            else:
                stdin_prompt.set_stdin_input_handler(handler)
                assert stdin_prompt.request_stdin_input("PRIVATE", "PRIVATE") == "PRIVATE_REPLY"
            assert pub.flush(2)
            assert status.read_status(tmp_path, "session")["state"] == "tool"
    finally:
        permission.set_permission_handler(None)
        stdin_prompt.set_stdin_input_handler(None)
        pub.close()


def test_runtime_emits_child_tool_and_usage_without_payloads(runtime, tmp_path):
    engine, agent, script, _ = runtime
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    observed = []
    def provider():
        assert pub.flush(2)
        observed.append(status.read_status(tmp_path, "session"))
        return reply("PRIVATE_MODEL", tc("record", text="PRIVATE_ARGUMENT"), tokens=123)
    script(provider, reply("", tc("finish_turn", message="PRIVATE_FINAL")))
    try:
        with status.bind_observer(pub):
            result = engine.run_turn(agent, "PRIVATE_USER")
        assert result.status == "answered"
        assert observed[0]["state"] == "inference"
        assert pub.flush(2)
        snapshot = status.read_status(tmp_path, "session")
        assert snapshot["state"] == "working"
        assert snapshot["observation"]["child_run_id"] == engine._last_turn_receipt["run_id"]
        assert snapshot["observation"]["prompt_tokens"] == 100
        assert snapshot["observation"]["completion_tokens"] == 10
        assert snapshot["observation"]["last_tool"] == "finish_turn"
        assert "PRIVATE" not in json.dumps(snapshot)
    finally:
        pub.close()


def test_missing_usage_remains_unknown(runtime, tmp_path):
    engine, agent, script, _ = runtime
    response = reply("answer")
    response.usage = None
    script(response)
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    try:
        with status.bind_observer(pub):
            assert engine.run_turn(agent, "hi").status == "answered"
        assert pub.flush(2)
        row = status.read_status(tmp_path, "session")["observation"]
        assert row["prompt_tokens"] is None and row["completion_tokens"] is None
        assert row["usage_source"] is None
    finally:
        pub.close()


def test_real_principal_runtime_child_closure_and_continuation(runtime, boundary, monkeypatch,
                                                              workspace_dir):
    from infinidev.engine.user_hooks import UserHookEvent

    engine, agent, script, _ = runtime
    agent.activate_context = Mock()
    agent.deactivate = Mock()
    parents, children = [], []
    def provider():
        pub = status.current_observer()
        assert pub.flush(2)
        row = status.read_status(workspace_dir, "principal-session")
        assert row["state"] == "inference"
        parents.append(row["observation"]["host_turn_id"])
        children.append(row["observation"]["child_run_id"])
        return reply("", tc("finish_turn", message="same final"))
    script(provider, provider)
    monkeypatch.setattr(pipeline, "_task_end_hook", lambda event, **kw:
                        "continue" if event == UserHookEvent.TASK_END_INSTRUCTION else "")
    assert call((agent, engine, boundary[2])) == "same final"
    assert len(set(parents)) == 1 and len(set(children)) == 2
    result = status.read_status(workspace_dir, "principal-session", expected_turn=parents[0])
    assert result["state"] == "closed"
    assert result["observation"]["child_run_id"] == children[-1]


def test_cancellation_closes_parent_as_cancelled(boundary, workspace_dir):
    boundary[1].is_cancelled = True
    assert call(boundary) == "Conversation cancelled."
    pub = boundary[1]._session_status_publisher
    row = status.read_status(workspace_dir, "principal-session", expected_turn=pub.host_turn_id)
    assert row["state"] == "closed"
    assert row["observation"]["terminal_status"] == "cancelled"


def test_post_publication_hook_failure_is_not_a_completed_parent(boundary, workspace_dir):
    def on_phase(phase):
        if phase == "idle":
            raise RuntimeError("post publication failed")
    boundary[2].on_phase.side_effect = on_phase
    with pytest.raises(RuntimeError, match="post publication"):
        call(boundary)
    pub = boundary[1]._session_status_publisher
    result = status.read_status(workspace_dir, "principal-session", expected_turn=pub.host_turn_id)
    assert result["state"] == "unknown"
    assert result["last_known"]["final_message_sha256"] is None


def test_unencodable_message_does_not_make_observation_change_the_result(boundary, workspace_dir):
    message = "unpaired-surrogate-\ud800"
    boundary[1].run_turn.return_value = TurnResult(message=message, status="answered",
                                                 state=TurnState())
    assert call(boundary) == message
    pub = boundary[1]._session_status_publisher
    result = status.read_status(workspace_dir, "principal-session", expected_turn=pub.host_turn_id)
    assert result["state"] == "unknown"
    assert result["last_known"]["final_message_sha256"] is None


def test_blocked_provider_keeps_heartbeat_without_claiming_progress(runtime, tmp_path, monkeypatch):
    monkeypatch.setattr(status, "_HEARTBEAT", 0.01)
    engine, agent, script, _ = runtime
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    def provider():
        assert pub.flush(2)
        before = status.read_status(tmp_path, "session")["observation"]
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            after = status.read_status(tmp_path, "session")["observation"]
            if after["sequence"] > before["sequence"]:
                break
            time.sleep(0.005)
        assert after["sequence"] > before["sequence"]
        assert after["event_sequence"] == before["event_sequence"]
        assert after["last_progress_at"] == before["last_progress_at"]
        assert after["phase"] == "inference"
        return reply("answer")
    script(provider)
    try:
        with status.bind_observer(pub):
            assert engine.run_turn(agent, "question").status == "answered"
    finally:
        pub.close()
