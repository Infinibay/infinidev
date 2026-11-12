"""ASCII-escaped provider text must reach the user as text, not as escapes.

A backend that serialises its JSON with ``ensure_ascii=True`` and then
re-wraps the payload hands the escapes over as *text*: the reported symptom
was the TUI painting ``\\u00a1Hola! \\ud83d\\udc4b`` instead of ``¡Hola! 👋``.
Nothing on the inbound path decoded them, so every renderer showed them
verbatim.

The repair lives in ``normalize_message_text`` — the single funnel every
consumer shares (developer loop, guardrails, chat agent, stream deltas).
The guards matter as much as the decode: code samples quote ``\\uXXXX`` on
purpose, and a reply that already contains real UTF-8 must not be
reinterpreted.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from infinidev.engine.loop.llm_caller import (
    decode_literal_unicode_escapes,
    normalize_message_text,
    promote_embedded_think,
)
from infinidev.engine.orchestration.chat_agent import (
    _build_respond,
    _consume_stream,
)

# The exact string the user reported, as it arrives over the wire: literal
# backslash-u sequences, not decoded characters.
REPORTED = r"\u00a1Hola! \ud83d\udc4b \u00bfEn qu\u00e9 te ayudo hoy?"
REPORTED_TEXT = "¡Hola! 👋 ¿En qué te ayudo hoy?"


def test_the_reported_message_renders_as_text():
    assert normalize_message_text(REPORTED) == REPORTED_TEXT


def test_surrogate_pairs_become_one_character():
    """``\ud83d\udc4b`` is one emoji carried as a UTF-16 pair; decoding the
    halves separately would leave unencodable lone surrogates."""
    decoded = normalize_message_text(r"\ud83d\ude80 rocket")
    assert decoded == "🚀 rocket"
    assert decoded.isascii() is False


def test_lone_surrogates_are_left_as_written():
    """A dangling high surrogate has no character to decode to — inventing
    one would corrupt the message."""
    raw = r"tail \ud83d end"
    assert normalize_message_text(raw) == raw


def test_broken_escapes_are_left_as_written():
    for raw in (r"\uZZZZ", r"\u00", r"cost is 5\u00a", r"backslash u \u"):
        assert normalize_message_text(raw) == raw


def test_already_decoded_text_is_untouched():
    """The no-regression case: a normal reply, and a reply that mixes real
    UTF-8 with a literal escape, are both returned verbatim."""
    for raw in (
        "¡Hola! 👋 ¿En qué te ayudo hoy?",
        r"already ¡decoded \u00e9",
        "plain ascii only",
        "",
    ):
        assert normalize_message_text(raw) == raw


def test_escapes_inside_code_are_preserved():
    """A model quoting Python or JSON is *supposed* to show ``\\u00e9``."""
    fenced = 'Use the literal:\\n```python\\ns = "\\u00e9"\\n```\\n'
    fenced = fenced.replace("\\n", "\n")
    assert normalize_message_text(fenced) == fenced
    inline = 'write `s = "\\u00e9"` here'
    assert normalize_message_text(inline) == inline
    # Prose outside the fence is still repaired.
    mixed = 'escribe "\\u00e9" as `\\u00e9`'
    assert normalize_message_text(mixed) == 'escribe "é" as `\\u00e9`'


def test_unterminated_fence_treats_the_rest_as_code():
    """An open fence means everything after it is a snippet the model is
    still writing — only the prose before it gets repaired."""
    raw = 'prose \\u00e9 then ```python\nx = "\\u00e9"\n'
    assert normalize_message_text(raw) == 'prose é then ```python\nx = "\\u00e9"\n'


def test_structured_content_blocks_are_decoded_too():
    """Some compatible endpoints hand back a content block instead of a
    string; the escape arrives in the same shape."""
    assert normalize_message_text({"type": "text", "text": REPORTED}) == REPORTED_TEXT
    assert normalize_message_text([{"text": REPORTED}]) == REPORTED_TEXT
    assert normalize_message_text({"text": {"value": REPORTED}}) == REPORTED_TEXT


def test_unknown_shapes_stay_machine_parseable():
    """Tool-call payloads go through the same funnel and must stay JSON."""
    content = {"name": "step_complete", "arguments": {"status": "done"}}
    assert json.loads(normalize_message_text(content)) == content


def test_decode_helper_is_idempotent_and_never_raises():
    once = decode_literal_unicode_escapes(REPORTED)
    assert decode_literal_unicode_escapes(once) == once
    for raw in ("\ud800", "\uffff" * 10, "\\u0000", "\\ud800\\ud800"):
        assert isinstance(decode_literal_unicode_escapes(raw), str)


def test_promote_embedded_think_stores_readable_content():
    """The global LiteLLM normalizer writes this back onto the message, and
    readers that touch ``message.content`` directly must not see escapes."""
    message = SimpleNamespace(
        content=f"{REPORTED}<thinking>hidden</thinking>",
        reasoning_content=None,
    )

    promote_embedded_think(message)

    assert message.content == REPORTED_TEXT
    assert message.reasoning_content == "hidden"


def test_escaped_content_is_written_back_even_without_think_blocks():
    """The transcript persists ``message.content`` verbatim, so an escaped
    reply left on the message comes back in the next turn's history."""
    message = SimpleNamespace(content=REPORTED, reasoning_content=None)

    promote_embedded_think(message)

    assert message.content == REPORTED_TEXT


def test_a_tool_call_blob_in_content_stays_parseable():
    """FC-fallback parses tool calls out of ``content``. A doubly-encoded
    payload carries JSON's own quotes and braces as ``\\u0022`` / ``\\u007b``;
    decoding those would break the parse, so they stay escaped."""
    blob = r'{\u0022name\u0022: \u0022respond\u0022, \u007b\u0022a\u0022: 1\u007d}'
    assert normalize_message_text(blob) == blob
    # A newline carried as \n stays escaped for the same reason.
    assert normalize_message_text(r'{"a": "line\nnext"}') == r'{"a": "line\nnext"}'
    # Prose around it is still repaired — only the unsafe codepoints are kept.
    assert normalize_message_text(r'\u00a1Hola \u0022quote\u0022') == '¡Hola \\u0022quote\\u0022'


# ── Streaming ────────────────────────────────────────────────────────────
# The stream is what the user actually watches: once the TUI has shown a
# chunk it is never re-rendered, so a split or escaped delta is permanent.


class _Hooks:
    def __init__(self) -> None:
        self.chunks: list[str] = []

    def notify_stream_chunk(
        self, speaker: str, chunk: str, kind: str = "agent",
    ) -> None:
        self.chunks.append(chunk)


def _chunks(*pieces: str) -> list[Any]:
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content=piece, tool_calls=[],
        ))])
        for piece in pieces
    ]


def _arg_chunks(*pieces: str) -> list[Any]:
    return [
        SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(
            content=None,
            tool_calls=[SimpleNamespace(
                index=0,
                id="tc-1" if i == 0 else None,
                function=SimpleNamespace(
                    name="respond" if i == 0 else None, arguments=piece,
                ),
            )],
        ))])
        for i, piece in enumerate(pieces)
    ]


def test_plain_streamed_text_with_escapes_is_readable():
    content, tool_calls, streamed = _consume_stream(
        _chunks(r"\u00a1Hola", " \\ud83d\\udc4b"), _Hooks(),
    )
    assert content == "¡Hola 👋"
    assert tool_calls == []
    assert streamed is True


def test_escapes_split_across_deltas_are_reassembled():
    """Token boundaries cut ``\\uXXXX`` in half. Publishing the half would
    show the escape forever and break the incremental diff."""
    hooks = _Hooks()
    content, _, _ = _consume_stream(
        _chunks(r"\u00", "a1Hola ", r"\ud83d", r"\udc4b"), hooks,
    )
    assert content == "¡Hola 👋"
    assert "".join(hooks.chunks) == "¡Hola 👋"


def test_a_stream_ending_mid_escape_keeps_the_text():
    """The held-back fragment must be released, not dropped with the stream
    — dropping it would silently truncate the reply."""
    content, _, _ = _consume_stream(_chunks(r"hola \u00"), _Hooks())
    assert content == "hola \\u00"


def test_streamed_code_fence_is_not_decoded():
    hooks = _Hooks()
    content, _, _ = _consume_stream(
        _chunks("escribe ", "`\\u00e9`", " en Python"), hooks,
    )
    assert content == 'escribe `\\u00e9` en Python'
    assert "".join(hooks.chunks) == 'escribe `\\u00e9` en Python'


def test_respond_tool_preview_streams_decoded_text():
    """``respond`` args stream as raw JSON; the preview the user watches has
    to be decoded text, and the diff must not lose characters."""
    hooks = _Hooks()
    content, tool_calls, streamed = _consume_stream(
        _arg_chunks('{"message":"', r"\u00a1Ho", "la \\ud83d", r'\udc4b"}'),
        hooks,
    )
    assert [tc.function.name for tc in tool_calls] == ["respond"]
    assert streamed is True
    assert "".join(hooks.chunks) == "¡Hola 👋"
    # The final, non-streamed pass still parses the real JSON payload.
    assert json.loads(tool_calls[0].function.arguments) == {"message": "¡Hola 👋"}


def test_respond_reply_is_decoded_even_when_never_streamed():
    """Non-streaming ``respond`` goes through ``json.loads`` + text_argument,
    which leaves an escaped payload escaped."""
    result = _build_respond(
        SimpleNamespace(function=SimpleNamespace(
            name="respond",
            arguments=json.dumps({"message": REPORTED}),
        )),
        "hola",
    )
    assert result.kind == "respond"
    assert result.reply == REPORTED_TEXT
