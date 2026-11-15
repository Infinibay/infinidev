"""Metadata-only host observations, without an engine, database or model."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time

import pytest

from infinidev.engine import session_status as status


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0, 100.0]
    monkeypatch.setattr(status, "_clocks", lambda: tuple(now))
    monkeypatch.setattr(status, "_process_identity", lambda pid: ("boot", "start"))
    return now


def observe(root, publisher, **kwargs):
    assert publisher.flush(timeout=2)
    return status.read_status(root, "session", **kwargs)


def test_reader_does_not_create_state(tmp_path):
    result = status.read_status(tmp_path, "missing")
    assert result["state"] == "unknown"
    assert not (tmp_path / ".infinidev").exists()


def test_host_phases_and_named_parent_closure(tmp_path, clock):
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    try:
        pub.record("inference", "model_request", child_run_id="b" * 32)
        assert observe(tmp_path, pub)["state"] == "inference"
        pub.record("tool", "tool_started", tool_name="read_file")
        assert observe(tmp_path, pub)["state"] == "tool"
        pub.record("working", "tool_completed", tool_name="read_file", progress=True)
        result = observe(tmp_path, pub)
        assert result["observation"]["last_tool"] == "read_file"
        assert result["observation"]["tool_completed_at"] == clock[0]
        pub.finish("answered", "c" * 64)
        assert observe(tmp_path, pub)["state"] == "unknown"
        result = observe(tmp_path, pub, expected_turn=pub.host_turn_id)
        assert result["state"] == "closed"
        assert result["scope"] == "named_turn"
        assert result["observation"]["final_message_sha256"] == "c" * 64
    finally:
        pub.close()


def test_heartbeat_is_not_progress_and_stale_is_unknown(tmp_path, clock):
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    try:
        pub.record("inference", "model_request")
        before = observe(tmp_path, pub)["observation"]
        clock[0] += 1
        clock[1] += 1
        pub.heartbeat()
        after = observe(tmp_path, pub)["observation"]
        assert after["last_event_at"] == before["last_event_at"]
        assert after["last_progress_at"] == before["last_progress_at"]
        assert after["event_sequence"] == before["event_sequence"]
        clock[1] += 20
        assert status.read_status(tmp_path, "session")["state"] == "unknown"
    finally:
        pub.close()


def test_repeated_message_hash_does_not_reuse_turn_identity(tmp_path, clock):
    first = status.StatusPublisher(tmp_path, "session", "a" * 32)
    first.start()
    first.finish("answered", "c" * 64)
    assert first.flush(timeout=2)
    first.close()
    second = status.StatusPublisher(tmp_path, "session", "a" * 32)
    second.start()
    try:
        second.record("inference", "model_request", child_run_id="b" * 32)
        result = observe(tmp_path, second, expected_turn=first.host_turn_id)
        assert result["state"] == "unknown"
        assert first.host_turn_id != second.host_turn_id
        second.finish("answered", "c" * 64)
        result = observe(tmp_path, second, expected_turn=first.host_turn_id)
        assert result["state"] == "unknown"
        assert observe(tmp_path, second, expected_turn=second.host_turn_id)["state"] == "closed"
    finally:
        second.close()


def test_child_close_is_not_parent_close(tmp_path, clock):
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    try:
        pub.record("working", "child_run_closed", child_run_id="b" * 32)
        result = observe(tmp_path, pub, expected_turn=pub.host_turn_id)
        assert result["state"] != "closed"
        assert result["observation"]["terminal_status"] is None
        pub.record("inference", "model_request", child_run_id="d" * 32)
        assert observe(tmp_path, pub, expected_run="b" * 32)["state"] == "unknown"
    finally:
        pub.close()


def test_multiple_live_owners_never_choose_a_closed_or_idle_record(tmp_path, clock):
    one = status.StatusPublisher(tmp_path, "session", "a" * 32)
    two = status.StatusPublisher(tmp_path, "session", "b" * 32)
    one.start()
    two.start()
    try:
        one.record("inference", "model_request")
        two.record("tool", "tool_started", tool_name="read_file")
        assert one.flush(timeout=2) and two.flush(timeout=2)
        result = status.read_status(tmp_path, "session")
        assert result["state"] == "unknown"
        assert result["reason"] == "multiple_active_owners"
    finally:
        one.close()
        two.close()


@pytest.mark.parametrize("identity", [("other-boot", "start"), ("boot", "other-start"), None])
def test_missing_process_or_pid_reuse_cannot_validate_a_record(tmp_path, clock, monkeypatch, identity):
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    try:
        pub.finish("answered", "c" * 64)
        assert pub.flush(timeout=2)
        monkeypatch.setattr(status, "_process_identity", lambda pid: identity)
        assert status.read_status(tmp_path, "session", expected_turn=pub.host_turn_id)["state"] == "unknown"
    finally:
        pub.close()


def test_unknown_or_sensitive_fields_are_not_returned(tmp_path, clock):
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    try:
        assert pub.flush(timeout=2)
        path = next((tmp_path / ".infinidev/session-status/v1").rglob("*.json"))
        data = json.loads(path.read_text())
        data["reasoning"] = "SENSITIVE_PRIVATE_BODY"
        path.write_text(json.dumps(data))
        result = status.read_status(tmp_path, "session")
        assert result["state"] == "unknown"
        assert "SENSITIVE_PRIVATE_BODY" not in json.dumps(result)
    finally:
        pub.close()


def test_interrupted_scope_is_unknown_not_closed(tmp_path, clock):
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    pub.record("tool", "tool_started", tool_name="read_file")
    pub.close()
    result = status.read_status(tmp_path, "session", expected_turn=pub.host_turn_id)
    assert result["state"] == "unknown"
    assert result["last_known"]["final_message_sha256"] is None


def test_reader_rejects_symlink_and_scan_saturation(tmp_path, clock):
    bucket = tmp_path / ".infinidev/session-status/v1" / hashlib.sha256(b"session").hexdigest()
    bucket.mkdir(parents=True)
    external = tmp_path / "not-status"
    external.write_text('{"reasoning":"PRIVATE"}')
    (bucket / "link.json").symlink_to(external)
    result = status.read_status(tmp_path, "session")
    assert result["state"] == "unknown" and "PRIVATE" not in json.dumps(result)
    (bucket / "link.json").unlink()
    for i in range(33):
        (bucket / f"{i}.json").write_text("{}")
    assert status.read_status(tmp_path, "session")["reason"] == "record_limit"


@pytest.mark.parametrize("payload", [b"{bad", b" " * 8193, b"[" * 2000 + b"]" * 2000])
def test_malformed_or_oversized_record_is_unknown(tmp_path, payload):
    bucket = status._bucket(tmp_path, "session", create=True)
    (bucket / ("a" * 32 + ".json")).write_bytes(payload)
    assert status.read_status(tmp_path, "session")["state"] == "unknown"


def test_future_heartbeat_is_not_fresh(tmp_path, clock):
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    try:
        assert pub.flush(2)
        clock[1] -= 1
        assert status.read_status(tmp_path, "session")["state"] == "unknown"
    finally:
        pub.close()


def test_atomic_reads_never_expose_partial_records(tmp_path):
    pub = status.StatusPublisher(tmp_path, "session", "a" * 32)
    pub.start()
    try:
        assert pub.flush(2)
        for index in range(30):
            pub.record("working", "model_response", prompt_tokens=index)
            row = status.read_status(tmp_path, "session")
            assert row["state"] == "working" or row["state"] == "preparing"
            assert row["observation"]["host_turn_id"] == pub.host_turn_id
        assert pub.flush(2)
    finally:
        pub.close()


def test_blocked_writer_does_not_block_main_or_overwrite_next_turn(tmp_path, monkeypatch):
    from types import SimpleNamespace

    entered, release = threading.Event(), threading.Event()
    original = status.StatusPublisher._write
    blocked = []
    def write(self, *args):
        if not blocked:
            blocked.append(self)
            entered.set()
            release.wait(5)
        return original(self, *args)
    monkeypatch.setattr(status.StatusPublisher, "_write", write)
    seen = []
    @status.observed_principal
    def principal(**kwargs):
        pub = status.current_observer()
        seen.append(pub)
        if len(seen) == 1:
            assert entered.wait(2)
        pub.finish("answered", "b" * 64)
        return "answer"
    engine = SimpleNamespace()
    try:
        started = time.monotonic()
        assert principal(engine=engine, workspace_path=tmp_path, session_id="session") == "answer"
        assert time.monotonic() - started < 1
        assert principal(engine=engine, workspace_path=tmp_path, session_id="session") == "answer"
        assert seen[0]._data["owner_instance_id"] != seen[1]._data["owner_instance_id"]
    finally:
        release.set()
        for pub in seen:
            pub._thread.join(2)
    records = list(status._bucket(tmp_path, "session").glob("*.json"))
    assert {json.loads(path.read_bytes())["host_turn_id"] for path in records} == {
        pub.host_turn_id for pub in seen}
