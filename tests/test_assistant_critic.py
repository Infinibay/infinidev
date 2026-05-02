"""Tests for the assistant pair-programming critic."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from infinidev.engine.loop.critic import (
    AssistantCritic,
    CriticVerdict,
    _format_proposed_calls,
    _format_tool_catalog,
    _parse_verdict,
    _strip_principal_system,
)


def _make_tc(name: str, args: dict | str) -> SimpleNamespace:
    """Build a fake tool call matching the litellm OpenAI shape."""
    arg_payload = args if isinstance(args, str) else __import__("json").dumps(args)
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arg_payload),
        id="tc-" + name,
    )


# ── Pure helpers (no LLM) ──────────────────────────────────────────────


class TestParseVerdict:
    def test_parses_clean_json(self):
        v = _parse_verdict('{"action": "recommendation", "message": "use edit_symbol"}')
        assert v == CriticVerdict(action="recommendation", message="use edit_symbol")

    def test_action_is_lowercased_and_trimmed(self):
        v = _parse_verdict('{"action": "  REJECT  ", "message": "no"}')
        assert v is not None
        assert v.action == "reject"

    def test_unknown_action_returns_none(self):
        assert _parse_verdict('{"action": "veto", "message": "no"}') is None

    def test_missing_action_returns_none(self):
        assert _parse_verdict('{"message": "hi"}') is None

    def test_non_string_message_is_coerced(self):
        v = _parse_verdict('{"action": "information", "message": {"a": 1}}')
        assert v is not None
        assert v.message == '{"a": 1}'

    def test_empty_content_returns_none(self):
        assert _parse_verdict("") is None
        assert _parse_verdict("   ") is None

    def test_garbage_returns_none(self):
        assert _parse_verdict("definitely not json") is None

    def test_continue_with_message_is_silent_via_property(self):
        v = _parse_verdict('{"action": "continue", "message": "ignored"}')
        assert v is not None
        assert v.is_silent

    def test_continue_with_empty_message_is_silent(self):
        v = CriticVerdict(action="continue", message="")
        assert v.is_silent

    def test_information_with_empty_message_is_also_silent(self):
        # An action that *would* speak but with no content adds nothing useful.
        v = CriticVerdict(action="information", message="   ")
        assert v.is_silent


class TestFormatProposedCalls:
    def test_renders_each_call_compactly(self):
        rendered = _format_proposed_calls([
            _make_tc("read_file", {"path": "foo.py"}),
            _make_tc("execute_command", {"command": "ls"}),
        ])
        assert "read_file(path=foo.py)" in rendered
        assert "execute_command(command=ls)" in rendered

    def test_truncates_long_string_args(self):
        long = "x" * 500
        rendered = _format_proposed_calls([_make_tc("write", {"content": long})])
        # Truncation suffix appears with the count of trimmed chars.
        assert "... (+260 chars)" in rendered
        assert len(rendered) < 600  # Way below 500-char arg

    def test_handles_empty_list(self):
        assert _format_proposed_calls([]) == "- <none>"

    def test_handles_unparseable_args(self):
        bad = SimpleNamespace(
            function=SimpleNamespace(name="oops", arguments="{not-json"),
            id="x",
        )
        # safe_json_loads returns the literal string for malformed input,
        # so .items() blows up — the helper must catch it gracefully.
        rendered = _format_proposed_calls([bad])
        assert "oops" in rendered or "unparseable" in rendered


class TestFormatToolCatalog:
    def test_renders_sorted_with_first_line_only(self):
        catalog = _format_tool_catalog({
            "zzz": "z tool",
            "aaa": "a tool\nsecond line should be hidden",
        })
        # Sorted alphabetically.
        assert catalog.index("aaa") < catalog.index("zzz")
        assert "second line" not in catalog

    def test_empty_catalog_message(self):
        assert "(sin catálogo" in _format_tool_catalog({})

    def test_missing_description_falls_back(self):
        out = _format_tool_catalog({"foo": ""})
        assert "foo" in out
        assert "sin descripción" in out


class TestStripPrincipalSystem:
    def test_drops_only_system_messages(self):
        msgs = [
            {"role": "system", "content": "PROTOCOL"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "system", "content": "MORE PROTOCOL"},
        ]
        out = _strip_principal_system(msgs)
        assert all(m["role"] != "system" for m in out)
        assert len(out) == 2


# ── AssistantCritic with mocked litellm ────────────────────────────────


@pytest.fixture
def fake_assistant_params(monkeypatch):
    """Stub get_litellm_params_for_assistant so tests don't need real config."""
    fake = {"model": "ollama_chat/qwen3:4b", "api_key": "x"}
    monkeypatch.setattr(
        "infinidev.engine.loop.critic.get_litellm_params_for_assistant",
        lambda: fake,
    )
    return fake


def _make_response(content: str) -> MagicMock:
    """Shape a fake OpenAI-style response: response.choices[0].message.content."""
    msg = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=msg)
    return SimpleNamespace(choices=[choice])


class TestAssistantCriticReview:
    def test_model_short_name_strips_provider_prefix(self, fake_assistant_params):
        c = AssistantCritic({"read_file": "Reads a file"})
        assert c.model_short_name == "qwen3:4b"

    def test_review_returns_verdict_on_clean_response(self, fake_assistant_params):
        c = AssistantCritic({"read_file": "Reads"})
        with patch(
            "infinidev.engine.llm_client.call_llm",
            return_value=_make_response('{"action": "reject", "message": "path missing"}'),
        ) as call:
            v = c.review(
                [{"role": "system", "content": "PROTOCOL"},
                 {"role": "user", "content": "do X"}],
                [_make_tc("read_file", {"path": "/nope"})],
            )
        assert v == CriticVerdict(action="reject", message="path missing")
        # System message from the principal must NOT be in the prompt the
        # critic actually sent — otherwise the critic would inherit the
        # principal's protocol/tool-schema rules and try to play driver.
        sent_msgs = call.call_args.args[1]
        assert sent_msgs[0]["role"] == "system"
        assert "emit_verdict" in sent_msgs[0]["content"].lower()
        # No principal system message should leak through.
        assert not any(
            m["role"] == "system" and "PROTOCOL" in m["content"]
            for m in sent_msgs
        )

    def test_review_returns_none_on_empty_tool_calls(self, fake_assistant_params):
        c = AssistantCritic({})
        assert c.review([], []) is None

    def test_review_returns_none_on_llm_exception(self, fake_assistant_params):
        c = AssistantCritic({"read_file": "Reads"})
        with patch(
            "infinidev.engine.llm_client.call_llm",
            side_effect=RuntimeError("network down"),
        ):
            v = c.review([{"role": "user", "content": "x"}],
                         [_make_tc("read_file", {"path": "a"})])
        assert v is None  # silent fallback, never blocks

    def test_review_text_without_tool_call_becomes_information(self, fake_assistant_params):
        """When the model responds with plain text (no emit_verdict tool
        call, no JSON, no prefix), the text reaches the principal as an
        ``information`` verdict. This is the deliberate text-fallback
        channel — the critic's input always reaches the principal,
        even when the model ignores the tool schema entirely.
        """
        c = AssistantCritic({"read_file": "Reads"})
        with patch(
            "infinidev.engine.llm_client.call_llm",
            return_value=_make_response("the model just rambled"),
        ):
            v = c.review([{"role": "user", "content": "x"}],
                         [_make_tc("read_file", {"path": "a"})])
        assert v is not None
        assert v.action == "information"
        assert v.message == "the model just rambled"

    def test_thinking_in_response_is_not_propagated(self, fake_assistant_params):
        """The critic's reasoning must be discarded — only JSON survives.

        We simulate a model that emitted a long thinking block followed
        by valid JSON. The verdict's message must be only the JSON's
        message field, NOT the thinking text.
        """
        thinking = "<think>I should be careful here...</think>\n"
        json_part = '{"action": "information", "message": "file is in src/"}'
        c = AssistantCritic({"read_file": "Reads"})
        with patch(
            "infinidev.engine.llm_client.call_llm",
            return_value=_make_response(thinking + json_part),
        ):
            v = c.review([{"role": "user", "content": "x"}],
                         [_make_tc("read_file", {"path": "a"})])
        assert v is not None
        assert "should be careful" not in v.message
        assert v.message == "file is in src/"


# ── End-to-end: feature disabled by default ────────────────────────────


class TestEngineDefaultDisabled:
    def test_engine_init_critic_is_none_by_default(self):
        """With the feature flag off, engine state must show no critic.

        This is the contract that lets the parallel-execution branch be
        a pure no-op for the 99% of users who haven't opted in. The
        test forces the flag off locally rather than asserting the
        live ``settings`` value, since a developer's own
        ``~/.infinidev/settings.json`` may legitimately enable it.
        """
        from infinidev.engine.loop.engine import LoopEngine
        from infinidev.config.settings import settings as _settings

        prior = _settings.ASSISTANT_LLM_ENABLED
        _settings.ASSISTANT_LLM_ENABLED = False
        try:
            eng = LoopEngine()
            assert eng._critic is None
            assert eng._pending_critic_messages == []
        finally:
            _settings.ASSISTANT_LLM_ENABLED = prior
