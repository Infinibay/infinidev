"""Ken has to watch the session, and must never be able to break it.

The ranker's reactive, predictive and explicit-mention channels are computed
from a stream of events, not from a query string — which is why asking Ken
questions without reporting the session left three of its channels dark.
These tests cover the reporting client: that it speaks the daemon's
protocol, that it stays silent when there is no daemon, and above all that
nothing it does can take down a coding session.
"""

from __future__ import annotations

import json

import pytest

from infinidev.engine.ken_session import KenSession, get_ken_session, reset_ken_sessions


@pytest.fixture(autouse=True)
def _enabled(monkeypatch):
    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "KEN_SESSION_ENABLED", True, raising=False)
    reset_ken_sessions()
    yield
    reset_ken_sessions()


@pytest.fixture
def workspace(tmp_path):
    """A workspace with a `.ken` directory and a live-looking daemon."""
    ken = tmp_path / ".ken"
    ken.mkdir()
    (ken / "daemon.port").write_text("54321")
    (ken / "meta.json").write_text(json.dumps({"auth_token": "s3cret"}))
    return tmp_path


@pytest.fixture
def posted(monkeypatch):
    """Capture every POST instead of making it."""
    calls: list[tuple[str, dict, dict]] = []

    class _Response:
        def __init__(self, body: bytes):
            self._body = body

        def read(self):
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def fake_urlopen(request, timeout=None):
        calls.append((
            request.full_url,
            json.loads(request.data.decode()),
            dict(request.headers),
        ))
        return _Response(b'{"ok": true, "context_block": "<context-rank/>"}')

    monkeypatch.setattr(
        "infinidev.engine.ken_session.urllib.request.urlopen", fake_urlopen
    )
    return calls


# ── protocol ─────────────────────────────────────────────────────────────


def test_it_speaks_the_daemons_protocol(workspace, posted):
    session = KenSession(workspace, "sess-1")
    session.start()

    url, body, headers = posted[0]
    assert url == "http://127.0.0.1:54321/sessions/start"
    assert body["session_id"] == "sess-1"
    assert body["cwd"] == str(workspace)
    assert headers["Authorization"] == "Bearer s3cret"


def test_every_event_carries_the_session_id(workspace, posted):
    session = KenSession(workspace, "sess-1")
    session.start()
    session.prompt("why is the scroll sticky")
    session.tool_pre("read_file", '{"file_path": "a.py"}')
    session.tool_post("read_file", success=True)
    session.turn_end()
    session.end()

    paths = [url.rsplit("/", 1)[-1] for url, _, _ in posted]
    assert paths == ["start", "prompts", "pre", "post", "turn-end", "end"]
    assert all(body["session_id"] == "sess-1" for _, body, _ in posted)


def test_tool_arguments_are_sent_as_a_mapping(workspace, posted):
    """Ken classifies read-vs-edit by finding a path inside the input. Tool
    calls arrive as a JSON *string* in function-calling mode, and a string
    would silently produce a target-less event the reactive channel ignores."""
    session = KenSession(workspace, "sess-1")
    session.tool_pre("read_file", '{"file_path": "src/a.py"}')

    _, body, _ = posted[0]
    assert body["input"] == {"file_path": "src/a.py"}


def test_malformed_tool_arguments_do_not_raise(workspace, posted):
    session = KenSession(workspace, "sess-1")
    session.tool_pre("read_file", "{not json at all")
    session.tool_pre("read_file", None)
    assert [body["input"] for _, body, _ in posted] == [{}, {}]


def test_a_failed_tool_is_reported_so_ken_can_retract_it(workspace, posted):
    """A broken read must not push a file up the ranking: the agent looked
    at it, but learned nothing from it."""
    session = KenSession(workspace, "sess-1")
    session.tool_post("read_file", success=False, arguments={"file_path": "gone.py"})

    _, body, _ = posted[0]
    assert body["success"] is False
    assert body["input"] == {"file_path": "gone.py"}


def test_start_returns_the_resume_brief(workspace, posted):
    assert KenSession(workspace, "sess-1").start() == "<context-rank/>"


def test_an_empty_prompt_is_not_recorded(workspace, posted):
    """A blank user_prompt row would consume a slot in a window that is
    shared across every agent using this index."""
    KenSession(workspace, "sess-1").prompt("   ")
    assert posted == []


# ── it must never break the host ─────────────────────────────────────────


def test_no_ken_directory_means_silence(tmp_path, posted):
    session = KenSession(tmp_path, "sess-1")
    assert session.available is False
    session.start()
    session.prompt("hello")
    session.tool_pre("read_file", {})
    session.end()
    assert posted == []


def test_a_daemon_that_is_not_running_is_not_contacted(tmp_path, posted):
    """The port file is how ken advertises a live daemon. Without it the
    client stays quiet rather than spawning one — a coding session must not
    pay for a subprocess launch and a model load at startup."""
    (tmp_path / ".ken").mkdir()
    assert KenSession(tmp_path, "sess-1").available is False


def test_a_refusing_daemon_never_raises(workspace, monkeypatch):
    def explode(*_a, **_k):
        raise OSError("connection refused")

    monkeypatch.setattr(
        "infinidev.engine.ken_session.urllib.request.urlopen", explode
    )
    session = KenSession(workspace, "sess-1")
    assert session.start() is None
    assert session.prompt("hello") is None
    session.tool_pre("read_file", {})
    session.turn_end()
    session.end()


def test_a_dead_daemon_stops_being_retried(workspace, monkeypatch):
    """Otherwise every tool call in the session pays the full timeout."""
    attempts = []

    def explode(*_a, **_k):
        attempts.append(1)
        raise OSError("connection refused")

    monkeypatch.setattr(
        "infinidev.engine.ken_session.urllib.request.urlopen", explode
    )
    session = KenSession(workspace, "sess-1")
    for _ in range(20):
        session.tool_pre("read_file", {})

    assert len(attempts) <= 3, "the client must go quiet after repeated failures"
    assert session.available is False


def test_garbled_daemon_metadata_is_survivable(workspace, posted):
    (workspace / ".ken" / "meta.json").write_text("{not json")
    session = KenSession(workspace, "sess-1")
    assert session.start() is None
    assert posted == []


# ── the accessor ─────────────────────────────────────────────────────────


def test_disabled_by_default_returns_nothing(monkeypatch, workspace):
    from infinidev.config import settings as settings_mod

    monkeypatch.setattr(settings_mod.settings, "KEN_SESSION_ENABLED", False)
    assert get_ken_session(workspace, "sess-1") is None


def test_the_same_session_is_shared_across_call_sites(workspace):
    """The engine, the pipeline and the tool runner all report to one
    session, and none of them holds a reference to the others."""
    first = get_ken_session(workspace, "sess-1")
    second = get_ken_session(workspace, "sess-1")
    assert first is second
    assert get_ken_session(workspace, "sess-2") is not first
