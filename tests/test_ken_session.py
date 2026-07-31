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


def test_the_reply_is_what_makes_turn_end_worth_posting(workspace, posted):
    """Ken scans it for cited paths — the strongest multiplier it has (2.5×).
    An empty turn-end leaves that channel dark and stores a blank context
    that future sessions cannot semantic-match against."""
    KenSession(workspace, "sess-1").turn_end("Fixed the leak in ui/app.py")

    _, body, _ = posted[0]
    assert body["assistant_text"] == "Fixed the leak in ui/app.py"


def test_a_long_reply_is_capped_before_it_is_posted(workspace, posted):
    """The daemon stores text[:8000] regardless, so anything past that is a
    larger POST bought for nothing."""
    KenSession(workspace, "sess-1").turn_end("x" * 50_000)

    _, body, _ = posted[0]
    assert len(body["assistant_text"]) == 8_000


# ── the session lifecycle ────────────────────────────────────────────────


def test_start_is_idempotent(workspace, posted):
    """Every turn opens the session so no host owns the "is this the first
    one?" bookkeeping. Only the first call may reach the daemon: each
    /sessions/start INSERTs a fresh cr_sessions row, so one per turn would
    shred a conversation into a row per task with the per-turn decay
    counter restarting each time."""
    session = KenSession(workspace, "sess-1")
    session.start()
    session.start()
    session.start()

    assert len(posted) == 1


def test_the_resume_brief_is_handed_over_exactly_once(workspace, posted):
    """It says where you left off. Re-injecting it on turn nine would be
    telling the model that about a conversation it is already having."""
    session = KenSession(workspace, "sess-1")
    assert session.start() == "<context-rank/>"
    assert session.start() is None


def test_a_daemon_that_comes_up_mid_session_is_still_picked_up(workspace, monkeypatch):
    """A failed open is not remembered — otherwise starting infinidev before
    the daemon costs the ranker the whole session."""
    attempts: list[int] = []

    def flaky(request, timeout=None):
        attempts.append(1)
        raise OSError("connection refused")

    monkeypatch.setattr(
        "infinidev.engine.ken_session.urllib.request.urlopen", flaky
    )
    session = KenSession(workspace, "sess-1")
    session.start()
    session.start()

    assert len(attempts) == 2


def test_ending_a_session_that_never_opened_posts_nothing(workspace, posted):
    KenSession(workspace, "sess-1").end()
    assert posted == []


def test_the_host_closes_every_session_it_opened(workspace, posted):
    """Hosts do not track which workspace/session pairs were opened — they
    just know the conversation is over. /sessions/end is what snapshots the
    productivity scores the predictive channel reads NEXT time, so a missed
    close costs the next session, not this one."""
    from infinidev.engine.ken_session import end_ken_sessions

    for sid in ("sess-1", "sess-2"):
        session = get_ken_session(workspace, sid)
        assert session is not None
        session.start()
    posted.clear()

    end_ken_sessions()

    assert sorted(body["session_id"] for _, body, _ in posted) == ["sess-1", "sess-2"]
    assert all(url.endswith("/sessions/end") for url, _, _ in posted)


def test_closing_twice_is_harmless(workspace, posted):
    """Nothing stops a host from having two exit paths."""
    from infinidev.engine.ken_session import end_ken_sessions

    session = get_ken_session(workspace, "sess-1")
    assert session is not None
    session.start()
    end_ken_sessions()
    posted.clear()

    end_ken_sessions()
    assert posted == []


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


# ── the pipeline reads the answers ───────────────────────────────────────
#
# Posting to Ken is the expensive half. These cover the half that pays for
# it: both endpoints answer with prompt text, and infinidev used to discard
# both.


@pytest.fixture
def fake_ken(monkeypatch):
    """Stand in for the daemon at the pipeline's seam."""

    class _Session:
        def __init__(self):
            self.brief: str | None = "<ken-session-brief>where you left off</ken-session-brief>"
            self.ranked: str | None = "<context-rank>src/a.py</context-rank>"
            self.prompts: list[str] = []
            self.turn_ends: list[str] = []
            self.starts = 0

        def start(self, workspace=None):
            self.starts += 1
            return self.brief if self.starts == 1 else None

        def prompt(self, text):
            self.prompts.append(text)
            return self.ranked

        def turn_end(self, assistant_text=""):
            self.turn_ends.append(assistant_text)

    session = _Session()
    monkeypatch.setattr(
        "infinidev.engine.ken_session.get_ken_session",
        lambda workspace=None, session_id=None: session,
    )
    return session


def test_both_of_kens_blocks_reach_the_prompt(fake_ken):
    """The resume brief and the ranked block are exactly what Ken's own
    hooks print to stdout for Claude Code to inject. Infinidev asked for
    both and threw both away."""
    from infinidev.engine.orchestration.pipeline import _ken_turn_context

    block = _ken_turn_context("why is the scroll sticky", "sess-1")

    assert "<ken-session-brief>" in block
    assert "<context-rank>" in block
    assert fake_ken.prompts == ["why is the scroll sticky"]


def test_the_blocks_are_not_re_wrapped(fake_ken):
    """They arrive tagged. A second wrapper would be a tag the model has
    never seen in any other host."""
    from infinidev.engine.orchestration.pipeline import _ken_turn_context

    block = _ken_turn_context("hi", "sess-1")
    assert block.count("<context-rank>") == 1
    assert block.startswith("<ken-session-brief>")


def test_only_the_first_turn_carries_the_brief(fake_ken):
    from infinidev.engine.orchestration.pipeline import _ken_turn_context

    first = _ken_turn_context("hi", "sess-1")
    second = _ken_turn_context("and now this", "sess-1")

    assert "<ken-session-brief>" in first
    assert "<ken-session-brief>" not in second
    assert "<context-rank>" in second


def test_a_silent_ken_contributes_nothing(fake_ken):
    """No daemon, no index, nothing ranked — the turn must not grow a blank
    context block or a stray pair of newlines."""
    from infinidev.engine.orchestration.pipeline import _ken_turn_context

    fake_ken.brief = None
    fake_ken.ranked = "   "
    assert _ken_turn_context("hi", "sess-1") == ""


def test_a_ken_that_explodes_does_not_take_the_turn_with_it(monkeypatch):
    from infinidev.engine.orchestration.pipeline import _ken_turn_context

    def explode(**_kwargs):
        raise RuntimeError("daemon on fire")

    monkeypatch.setattr(
        "infinidev.engine.ken_session.get_ken_session", explode
    )
    assert _ken_turn_context("hi", "sess-1") == ""


def test_the_turn_ends_with_the_reply_ken_needs(fake_ken):
    from infinidev.engine.orchestration.pipeline import _report_turn_end_to_ken

    _report_turn_end_to_ken("Patched ui/app.py", "sess-1")
    assert fake_ken.turn_ends == ["Patched ui/app.py"]
