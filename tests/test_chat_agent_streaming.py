"""Tests for chat-agent streaming (`hooks.notify_stream_chunk`).

When the caller passes ``hooks`` to ``run_chat_agent``, the LLM call
runs in streaming mode. The chat agent accumulates tool_call argument
chunks, extracts the ``message`` field of a forming ``respond`` tool
call from the partial JSON, and emits the diff of new characters via
``hooks.notify_stream_chunk`` as they arrive.

The tests mock ``litellm.completion`` to return an iterator of chunk
objects that mirror what LiteLLM would produce. They verify:
  * notify_stream_chunk is called with the right progressive content
  * ChatAgentResult.streamed is True when chunks were emitted
  * ChatAgentResult.streamed is False when hooks is None (existing path)
  * Plain-text content (no tool_calls) also streams
  * JSON escapes (``\\"``, ``\\n``) are unescaped incrementally
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import pytest

from infinidev.engine.orchestration.chat_agent import (
    run_chat_agent,
    _extract_partial_message,
)


# ─────────────────────────────────────────────────────────────────────────
# Minimal shapes mirroring LiteLLM's streaming response
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class _DeltaFn:
    name: str | None = None
    arguments: str | None = None


@dataclass
class _DeltaToolCall:
    index: int
    id: str | None = None
    function: _DeltaFn = field(default_factory=_DeltaFn)


@dataclass
class _Delta:
    content: str | None = None
    tool_calls: list[_DeltaToolCall] = field(default_factory=list)


@dataclass
class _Choice:
    delta: _Delta


@dataclass
class _Chunk:
    choices: list[_Choice]


def _tc_chunk(
    idx: int, *, id_: str | None = None, name: str | None = None,
    args_piece: str | None = None,
) -> _Chunk:
    return _Chunk(choices=[_Choice(delta=_Delta(
        tool_calls=[_DeltaToolCall(
            index=idx, id=id_,
            function=_DeltaFn(name=name, arguments=args_piece),
        )],
    ))])


def _content_chunk(text: str) -> _Chunk:
    return _Chunk(choices=[_Choice(delta=_Delta(content=text))])


class _RecordingHooks:
    """Captures notify_stream_chunk and notify_stream_end calls verbatim."""

    def __init__(self) -> None:
        self.chunks: list[tuple[str, str, str]] = []
        self.messages: list[tuple[str, str, str]] = []
        self.stream_ends: list[tuple[str, str]] = []

    def on_phase(self, phase: str) -> None: pass
    def on_status(self, level: str, msg: str) -> None: pass
    def ask_user(self, prompt: str, kind: str = "text") -> str | None: return None
    def on_step_start(self, *a, **kw) -> None: pass
    def on_file_change(self, path: str) -> None: pass

    def notify(self, speaker: str, msg: str, kind: str = "agent") -> None:
        self.messages.append((speaker, msg, kind))

    def notify_stream_chunk(
        self, speaker: str, chunk: str, kind: str = "agent",
    ) -> None:
        self.chunks.append((speaker, chunk, kind))

    def notify_stream_end(
        self, speaker: str, kind: str = "agent",
    ) -> None:
        self.stream_ends.append((speaker, kind))


@pytest.fixture
def patch_stream(monkeypatch):
    def _install(chunks: Iterable[_Chunk]):
        # The chat agent calls `litellm.completion(**kwargs)` — with
        # stream=True it expects an iterable. We return a list (iterable)
        # of _Chunk objects. A second call (rare in these tests) would
        # try to consume the same list again and yield empty.
        chunks_list = list(chunks)
        def _fake(**kwargs):
            assert kwargs.get("stream") is True, "streaming test must set stream=True"
            return iter(chunks_list)
        import litellm as _lit
        monkeypatch.setattr(_lit, "completion", _fake)

    monkeypatch.setattr(
        "infinidev.db.service.get_recent_turns_full",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "infinidev.engine.orchestration.chat_agent.get_litellm_params_for_behavior",
        lambda: {"model": "test/mock", "api_base": "http://localhost"},
    )
    return _install


# ─────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────


class TestPartialMessageExtractor:
    def test_empty_args_returns_empty(self):
        assert _extract_partial_message("") == ""
        assert _extract_partial_message("{") == ""

    def test_key_not_started(self):
        assert _extract_partial_message('{"not_message":"x"}') == ""

    def test_value_in_progress(self):
        assert _extract_partial_message('{"message":"hel') == "hel"

    def test_value_complete_before_close(self):
        assert _extract_partial_message('{"message":"hello"') == "hello"

    def test_escaped_quote(self):
        assert _extract_partial_message('{"message":"say \\"hi\\""') == 'say "hi"'

    def test_escaped_newline(self):
        assert _extract_partial_message('{"message":"line1\\nline2"') == "line1\nline2"

    def test_backslash_escape(self):
        assert _extract_partial_message('{"message":"C:\\\\path"') == "C:\\path"


class TestStreamingPath:
    def test_respond_streams_character_chunks(self, patch_stream):
        # LLM emits tool_call chunks: first the name, then pieces of
        # arguments that together form {"message": "Hola mundo"}.
        patch_stream([
            _tc_chunk(0, id_="tc-1", name="respond"),
            _tc_chunk(0, args_piece='{"message":"'),
            _tc_chunk(0, args_piece="Hola "),
            _tc_chunk(0, args_piece="mundo"),
            _tc_chunk(0, args_piece='"}'),
        ])
        hooks = _RecordingHooks()
        result = run_chat_agent("hola", hooks=hooks)

        assert result.kind == "respond"
        assert result.reply == "Hola mundo"
        assert result.streamed is True

        # The emitted chunks, concatenated, equal the final reply.
        emitted_text = "".join(c[1] for c in hooks.chunks)
        assert emitted_text == "Hola mundo"
        # Speaker is always "Infinidev" for chat agent output.
        assert all(c[0] == "Infinidev" for c in hooks.chunks)

    def test_no_duplicate_emission_when_re_extracting(self, patch_stream):
        # Two chunks where the second re-extends the message field.
        patch_stream([
            _tc_chunk(0, name="respond", args_piece='{"message":"'),
            _tc_chunk(0, args_piece="abc"),
            _tc_chunk(0, args_piece="def"),
            _tc_chunk(0, args_piece='"}'),
        ])
        hooks = _RecordingHooks()
        result = run_chat_agent("hi", hooks=hooks)
        emitted = "".join(c[1] for c in hooks.chunks)
        assert emitted == "abcdef"
        assert result.reply == "abcdef"

    def test_plain_text_also_streams(self, patch_stream):
        """When the model emits plain text (no tool call at all), we
        still stream it and terminate with kind=respond."""
        patch_stream([
            _content_chunk("hel"),
            _content_chunk("lo"),
        ])
        hooks = _RecordingHooks()
        result = run_chat_agent("hi", hooks=hooks)
        assert result.kind == "respond"
        assert result.reply == "hello"
        assert result.streamed is True
        assert hooks.chunks == [
            ("Infinidev", "hel", "agent"),
            ("Infinidev", "lo", "agent"),
        ]

    def test_escalate_does_not_stream(self, patch_stream):
        """Escalate is a handoff, not user-facing text. No chunks should
        be emitted for its JSON args — only the respond tool streams."""
        patch_stream([
            _tc_chunk(0, id_="tc-1", name="escalate"),
            _tc_chunk(0, args_piece='{"understanding":"fix auth bug",'),
            _tc_chunk(0, args_piece='"user_visible_preview":"Voy a arreglarlo"}'),
        ])
        hooks = _RecordingHooks()
        result = run_chat_agent("arreglá el bug", hooks=hooks)
        assert result.kind == "escalate"
        # No streaming chunks — escalate's args are internal.
        assert hooks.chunks == []

    def test_read_tool_then_respond_streams_only_respond(self, patch_stream):
        """First iteration: a list_directory call (no streaming). The
        agent would need a second iteration to terminate, but the test
        stream cuts off here — we verify the interim tool call doesn't
        leak text to the user."""
        patch_stream([
            _tc_chunk(0, id_="tc-1", name="list_directory"),
            _tc_chunk(0, args_piece='{"file_path":"."}'),
        ])
        hooks = _RecordingHooks()
        # The agent may go to a second iteration; that's fine — we just
        # assert the first-iteration read didn't emit user-facing chunks.
        try:
            run_chat_agent("hola", hooks=hooks, max_iterations=1)
        except Exception:
            pass
        assert hooks.chunks == []


class TestNonStreamingPathPreserved:
    """When hooks is None, behaviour is identical to before the streaming
    change. Existing tests in test_chat_agent.py already cover this; we
    add one explicit assertion that streamed=False in that path."""

    def test_streamed_false_when_no_hooks(self, monkeypatch):
        from dataclasses import dataclass as _dc, field as _field

        @_dc
        class _F: name: str; arguments: str
        @_dc
        class _TC: id: str; function: _F; type: str = "function"
        @_dc
        class _M: content: str = ""; tool_calls: list = _field(default_factory=list)
        @_dc
        class _C: message: _M
        @_dc
        class _R: choices: list

        def _fake(**kwargs):
            assert kwargs.get("stream") is False, "no-hooks path must not stream"
            tc = _TC(id="t1", function=_F(name="respond", arguments='{"message":"hi"}'))
            return _R(choices=[_C(message=_M(tool_calls=[tc]))])

        import litellm as _lit
        monkeypatch.setattr(_lit, "completion", _fake)
        monkeypatch.setattr(
            "infinidev.db.service.get_recent_turns_full", lambda *a, **kw: [],
        )
        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.get_litellm_params_for_behavior",
            lambda: {"model": "test/mock", "api_base": "http://localhost"},
        )
        result = run_chat_agent("hola")
        assert result.kind == "respond"
        assert result.reply == "hi"
        assert result.streamed is False


class TestChatAgentStreamingEndsOnTerminator:
    """The chat agent produces ChatAgentResult(streamed=True) when a
    stream path fired. Pipeline tests above cover the downstream; this
    one pins the upstream contract: the respond tool's full reply must
    match the concatenation of emitted chunks."""

    def test_respond_reply_equals_concatenated_chunks(self, patch_stream):
        patch_stream([
            _tc_chunk(0, name="respond", args_piece='{"message":"'),
            _tc_chunk(0, args_piece="# Header\\n"),
            _tc_chunk(0, args_piece="with **bold**"),
            _tc_chunk(0, args_piece='"}'),
        ])
        hooks = _RecordingHooks()
        result = run_chat_agent("hi", hooks=hooks)
        assert result.kind == "respond"
        assert result.streamed is True
        # Partial markdown markers flowed through the stream — the final
        # reply still equals the un-escaped content.
        assert result.reply == "# Header\nwith **bold**"
        # And the concatenation of emitted chunks matches the final
        # reply. The TUI will render these as plain text, then the
        # stream_end flip triggers a re-render with markdown applied.
        assert "".join(c[1] for c in hooks.chunks) == "# Header\nwith **bold**"


class TestPipelineDoesNotDoubleRender:
    """When streamed=True, pipeline.run_task must NOT call hooks.notify
    for the reply (the UI already saw it chunk-by-chunk) but SHOULD call
    hooks.notify_stream_end so the UI can flip the markdown-deferred
    flag and re-render with full styling."""

    def test_streamed_result_skips_final_notify_and_ends_stream(self, monkeypatch):
        from infinidev.engine.orchestration.chat_agent_result import ChatAgentResult
        from infinidev.engine.orchestration.pipeline import run_task

        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            lambda *a, **kw: ChatAgentResult(
                kind="respond", reply="shown via stream", streamed=True,
            ),
        )

        hooks = _RecordingHooks()

        class _FakeAgent:
            agent_id = "x"
            def activate_context(self, session_id): pass
            def deactivate(self): pass

        class _FakeEngine:
            def execute(self, **kw): return "unused"
            def has_file_changes(self): return False

        result = run_task(
            agent=_FakeAgent(),
            user_input="hola",
            session_id="s",
            engine=_FakeEngine(),
            reviewer=object(),
            hooks=hooks,
        )
        assert result == "shown via stream"
        # streamed=True ⇒ the reply must NOT appear via atomic notify.
        assert not any(
            speaker == "Infinidev" and msg == "shown via stream"
            for speaker, msg, _ in hooks.messages
        )
        # But the stream MUST be signalled as ended so the UI re-renders
        # with markdown applied.
        assert ("Infinidev", "agent") in hooks.stream_ends

    def test_unstreamed_result_still_emits_notify(self, monkeypatch):
        from infinidev.engine.orchestration.chat_agent_result import ChatAgentResult
        from infinidev.engine.orchestration.pipeline import run_task

        monkeypatch.setattr(
            "infinidev.engine.orchestration.chat_agent.run_chat_agent",
            lambda *a, **kw: ChatAgentResult(
                kind="respond", reply="atomic reply", streamed=False,
            ),
        )

        hooks = _RecordingHooks()

        class _FakeAgent:
            agent_id = "x"
            def activate_context(self, session_id): pass
            def deactivate(self): pass

        class _FakeEngine:
            def execute(self, **kw): return "unused"
            def has_file_changes(self): return False

        run_task(
            agent=_FakeAgent(),
            user_input="hi",
            session_id="s",
            engine=_FakeEngine(),
            reviewer=object(),
            hooks=hooks,
        )
        assert any(
            speaker == "Infinidev" and msg == "atomic reply"
            for speaker, msg, _ in hooks.messages
        )
