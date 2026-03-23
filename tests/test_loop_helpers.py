"""Tests for loop engine helper functions (not the full execute loop)."""

import json

import pytest

from infinidev.engine.loop_engine import (
    _extract_tool_detail,
    _extract_tool_error,
    _is_malformed_tool_call,
    _is_transient,
    _parse_text_tool_calls,
)


# ── _is_transient ────────────────────────────────────────────────────────────


class TestTransientErrorDetection:
    """Classify LLM errors as transient vs permanent."""

    def test_connection_error_is_transient(self):
        """Connection errors trigger retry."""
        assert _is_transient(Exception("APIConnectionError: connection error")) is True

    def test_rate_limit_is_transient(self):
        """Rate limit / 429 triggers retry."""
        assert _is_transient(Exception("Rate limit exceeded (429)")) is True

    def test_timeout_is_transient(self):
        """Timeout triggers retry."""
        assert _is_transient(Exception("Request timeout after 30s")) is True

    def test_overloaded_is_transient(self):
        """Server overloaded triggers retry."""
        assert _is_transient(Exception("503 server overloaded")) is True

    def test_permanent_not_transient(self):
        """'does not support tools' is permanent even if it contains transient substring."""
        assert _is_transient(Exception("does not support tools")) is False

    def test_not_found_is_permanent(self):
        """'not found' overrides transient matches."""
        assert _is_transient(Exception("tool 'X' not found")) is False

    def test_unknown_error_not_transient(self):
        """Random error message is not transient."""
        assert _is_transient(Exception("something completely different")) is False


# ── _is_malformed_tool_call ──────────────────────────────────────────────────


class TestMalformedToolCallDetection:
    """Detect malformed tool call errors from LLM providers."""

    def test_error_parsing_detected(self):
        """'error parsing tool call' matches."""
        assert _is_malformed_tool_call(Exception("error parsing tool call")) is True

    def test_invalid_character_detected(self):
        """'invalid character' matches."""
        assert _is_malformed_tool_call(Exception("invalid character in arguments")) is True

    def test_normal_error_not_malformed(self):
        """Random error not detected as malformed."""
        assert _is_malformed_tool_call(Exception("connection refused")) is False


# ── _parse_text_tool_calls ───────────────────────────────────────────────────


class TestParseTextToolCalls:
    """Parse tool calls from model text in manual TC mode."""

    def test_parse_manual_mode_json(self):
        """Our JSON format: {"tool_calls": [...]}."""
        text = json.dumps({
            "tool_calls": [
                {"name": "read_file", "arguments": {"path": "test.py"}}
            ]
        })
        calls = _parse_text_tool_calls(text)
        assert calls is not None
        assert len(calls) == 1
        assert calls[0]["name"] == "read_file"

    def test_parse_qwen_tool_call_tags(self):
        """Qwen <tool_call>{...}</tool_call> format."""
        text = '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
        calls = _parse_text_tool_calls(text)
        assert calls is not None
        assert calls[0]["name"] == "read_file"

    def test_parse_qwen_pipe_delimited(self):
        """Qwen <|tool_call|>{...}<|/tool_call|> format."""
        text = '<|tool_call|>{"name": "read_file", "arguments": {"path": "x.py"}}<|/tool_call|>'
        calls = _parse_text_tool_calls(text)
        assert calls is not None
        assert calls[0]["name"] == "read_file"

    def test_parse_mistral_format(self):
        """Mistral [TOOL_CALLS] [{...}] format."""
        text = '[TOOL_CALLS] [{"name": "read_file", "arguments": {"path": "x.py"}}]'
        calls = _parse_text_tool_calls(text)
        assert calls is not None
        assert calls[0]["name"] == "read_file"

    def test_parse_markdown_code_block(self):
        """Tool call inside ```json ... ```."""
        text = '```json\n{"tool_calls": [{"name": "read_file", "arguments": {"path": "x.py"}}]}\n```'
        calls = _parse_text_tool_calls(text)
        assert calls is not None
        assert calls[0]["name"] == "read_file"

    def test_parse_empty_content_returns_none(self):
        """Empty/whitespace returns None."""
        assert _parse_text_tool_calls("") is None
        assert _parse_text_tool_calls("   ") is None

    def test_parse_strips_thinking_sections(self):
        """Content with <think>...</think> around tool call still parses."""
        text = (
            "<think>I need to read the file...</think>"
            '<tool_call>{"name": "read_file", "arguments": {"path": "x.py"}}</tool_call>'
        )
        calls = _parse_text_tool_calls(text)
        assert calls is not None
        assert calls[0]["name"] == "read_file"

    def test_parse_plain_text_returns_none(self):
        """Plain text with no tool calls returns None."""
        assert _parse_text_tool_calls("Hello, how can I help you?") is None


# ── _extract_tool_detail ─────────────────────────────────────────────────────


class TestExtractToolDetail:
    """Extract human-readable detail from tool call arguments."""

    def test_extract_path_from_read_file(self):
        """Returns the path value for read_file."""
        detail = _extract_tool_detail("read_file", json.dumps({"path": "src/auth.py"}))
        assert detail == "src/auth.py"

    def test_extract_query_from_code_search(self):
        """Returns query for code_search."""
        detail = _extract_tool_detail("code_search", json.dumps({"query": "gradient optimizer"}))
        assert detail == "gradient optimizer"

    def test_truncates_long_values(self):
        """Values > 80 chars are truncated."""
        long_val = "x" * 100
        detail = _extract_tool_detail("read_file", json.dumps({"path": long_val}))
        assert len(detail) <= 80
        assert detail.endswith("...")

    def test_handles_invalid_json(self):
        """Malformed arguments returns empty string."""
        assert _extract_tool_detail("read_file", "not json at all") == ""

    def test_handles_empty_arguments(self):
        """Empty arguments returns empty string."""
        assert _extract_tool_detail("read_file", "") == ""


# ── _extract_tool_error ──────────────────────────────────────────────────────


class TestExtractToolError:
    """Extract error message from tool results."""

    def test_extracts_error_from_json(self):
        """{"error": "msg"} returns "msg"."""
        result = _extract_tool_error(json.dumps({"error": "File not found"}))
        assert "File not found" in result

    def test_no_error_returns_empty(self):
        """Normal content returns empty string."""
        assert _extract_tool_error("some normal text") == ""

    def test_non_json_returns_empty(self):
        """Plain text returns empty string."""
        assert _extract_tool_error("hello world") == ""

    def test_empty_returns_empty(self):
        """Empty string returns empty string."""
        assert _extract_tool_error("") == ""

    def test_json_without_error_key(self):
        """JSON without 'error' key returns empty."""
        assert _extract_tool_error(json.dumps({"result": "ok"})) == ""
