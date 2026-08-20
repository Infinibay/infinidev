"""Tests for loop engine helper functions (not the full execute loop)."""

import json
from types import SimpleNamespace

import pytest

from infinidev.engine.llm_client import (
    call_llm,
    is_transient as _is_transient,
    is_malformed_tool_call as _is_malformed_tool_call,
)
from infinidev.engine.engine_logging import (
    extract_tool_detail as _extract_tool_detail,
    extract_tool_error as _extract_tool_error,
)
from infinidev.engine.formats.tool_call_parser import (
    parse_text_tool_calls as _parse_text_tool_calls,
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


class TestLLMRetryBudget:
    """The engine owns one retry budget instead of nesting two loops."""

    @staticmethod
    def _stub_capabilities(monkeypatch):
        caps = SimpleNamespace(
            supports_json_mode=False,
            supports_tool_choice_required=True,
        )
        monkeypatch.setattr(
            "infinidev.config.model_capabilities.get_model_capabilities",
            lambda: caps,
        )

    def test_call_disables_litellm_transport_retries(self, monkeypatch):
        """call_llm retries itself, so its individual transport call retries zero times."""
        self._stub_capabilities(monkeypatch)
        seen = {}
        response = object()
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or response,
        )

        params = {
            "model": "minimax/MiniMax-M3",
            "num_retries": 3,
            "retry_strategy": "exponential_backoff_retry",
        }
        result = call_llm(
            params,
            [{"role": "user", "content": "x"}],
            retry_attempts=1,
        )

        assert result is response

        assert seen["num_retries"] == 0
        assert "retry_strategy" not in seen
        assert params["num_retries"] == 3

    def test_call_can_skip_json_mode_for_incompatible_provider(self, monkeypatch):
        caps = SimpleNamespace(
            supports_json_mode=True,
            supports_tool_choice_required=True,
        )
        monkeypatch.setattr(
            "infinidev.config.model_capabilities.get_model_capabilities",
            lambda: caps,
        )
        seen = {}
        response = object()
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or response,
        )

        result = call_llm(
            {"model": "zai/glm-5.2"},
            [{"role": "user", "content": "return json"}],
            retry_attempts=1,
            use_json_mode=False,
        )

        assert result is response
        assert "response_format" not in seen

    def test_call_can_disable_zai_thinking_per_request(self, monkeypatch):
        from infinidev.config.settings import settings

        self._stub_capabilities(monkeypatch)
        seen = {}
        response = object()
        monkeypatch.setattr(settings, "LLM_PROVIDER", "zai_coding")
        monkeypatch.setattr(settings, "THINKING_ENABLED", True)
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or response,
        )

        result = call_llm(
            {"model": "zai/glm-5.2", "max_tokens": 256},
            [{"role": "user", "content": "classify this"}],
            retry_attempts=1,
            thinking_enabled=False,
        )

        assert result is response
        assert seen["extra_body"]["thinking"] == {"type": "disabled"}
        assert "thinking" not in seen
        assert seen["max_tokens"] == 256
        assert seen["messages"][-1]["content"] == "classify this"

    def test_configured_retries_are_total_outer_budget(self, monkeypatch):
        """N configured retries mean one initial call plus exactly N retries."""
        from infinidev.config.settings import settings
        from infinidev.engine import llm_client

        self._stub_capabilities(monkeypatch)
        calls = 0

        def fail_with_timeout(**kwargs):
            nonlocal calls
            calls += 1
            assert kwargs["num_retries"] == 0
            raise Exception("request timeout")

        monkeypatch.setattr(settings, "LLM_NUM_RETRIES", 2)
        monkeypatch.setattr("litellm.completion", fail_with_timeout)
        monkeypatch.setattr(llm_client.time, "sleep", lambda delay: None)

        with pytest.raises(Exception, match="request timeout"):
            call_llm({"model": "minimax/MiniMax-M3"}, [{"role": "user", "content": "x"}])

        assert calls == 3

    def test_minimax_output_is_capped_to_provider_contract(self, monkeypatch):
        from infinidev.config.settings import settings

        self._stub_capabilities(monkeypatch)
        seen = {}
        response = object()
        monkeypatch.setattr(settings, "LLM_PROVIDER", "minimax")
        monkeypatch.setattr(settings, "THINKING_ENABLED", True)
        monkeypatch.setattr(settings, "THINKING_BUDGET", "high")
        monkeypatch.setattr(
            "litellm.completion",
            lambda **kwargs: seen.update(kwargs) or response,
        )

        result = call_llm(
            {"model": "minimax/MiniMax-M3"},
            [{"role": "user", "content": "x"}],
            retry_attempts=1,
        )

        assert result is response
        assert seen["max_tokens"] == 8_192


class TestRollingHorizonToolRouting:
    """Invalid planning actions are removed before the model samples them."""

    @staticmethod
    def _schema(name):
        return {"type": "function", "function": {"name": name}}

    def test_full_horizon_hides_add_step(self):
        from infinidev.engine.loop.llm_caller import LLMCaller

        plan = SimpleNamespace(
            rolling_horizon_limit=2,
            steps=[
                SimpleNamespace(status="active"),
                SimpleNamespace(status="pending"),
            ],
        )
        schemas = [self._schema("read_file"), self._schema("add_step")]
        ctx = SimpleNamespace(
            planning_schemas=schemas,
            tool_schemas=schemas,
            state=SimpleNamespace(plan=plan),
        )

        available = LLMCaller._available_schemas(ctx, is_planning=False)

        assert [schema["function"]["name"] for schema in available] == ["read_file"]

    def test_open_horizon_keeps_add_step(self):
        from infinidev.engine.loop.llm_caller import LLMCaller

        plan = SimpleNamespace(
            rolling_horizon_limit=2,
            steps=[SimpleNamespace(status="active")],
        )
        schemas = [self._schema("read_file"), self._schema("add_step")]
        ctx = SimpleNamespace(
            planning_schemas=schemas,
            tool_schemas=schemas,
            state=SimpleNamespace(plan=plan),
        )

        available = LLMCaller._available_schemas(ctx, is_planning=False)

        assert available is schemas

    def test_completion_turn_exposes_state_only_plan_tools(self):
        from infinidev.engine.loop.llm_caller import LLMCaller

        schemas = [
            self._schema("read_file"),
            self._schema("add_step"),
            self._schema("modify_step"),
            self._schema("remove_step"),
            self._schema("add_note"),
            self._schema("step_complete"),
        ]
        ctx = SimpleNamespace(
            planning_schemas=schemas,
            tool_schemas=schemas,
            state=SimpleNamespace(plan=None),
        )

        available = LLMCaller._available_schemas(
            ctx, is_planning=False, completion_only=True,
        )

        assert [schema["function"]["name"] for schema in available] == [
            "add_step", "modify_step", "remove_step", "add_note", "step_complete",
        ]

    def test_semantic_stagnation_hides_discovery_but_keeps_action_tools(self):
        from infinidev.engine.loop.llm_caller import LLMCaller

        schemas = [
            self._schema("read_file"),
            self._schema("recall_context"),
            self._schema("edit_file"),
            self._schema("execute_command"),
            self._schema("step_complete"),
        ]
        ctx = SimpleNamespace(
            planning_schemas=schemas,
            tool_schemas=schemas,
            state=SimpleNamespace(plan=None),
            suppress_discovery_this_step=True,
        )

        available = LLMCaller._available_schemas(ctx, is_planning=False)

        assert [schema["function"]["name"] for schema in available] == [
            "edit_file", "execute_command", "step_complete",
        ]

    def test_semantic_recovery_exposes_only_bounded_local_context(self):
        from infinidev.engine.loop.llm_caller import LLMCaller

        schemas = [
            self._schema("read_file"),
            self._schema("recall_context"),
            self._schema("web_search"),
            self._schema("edit_file"),
            self._schema("execute_command"),
            self._schema("step_complete"),
        ]
        ctx = SimpleNamespace(
            planning_schemas=schemas,
            tool_schemas=schemas,
            state=SimpleNamespace(plan=None),
            suppress_discovery_this_step=True,
            semantic_recovery_context_calls=2,
        )

        available = LLMCaller._available_schemas(ctx, is_planning=False)

        assert [schema["function"]["name"] for schema in available] == [
            "read_file", "edit_file", "execute_command", "step_complete",
        ]

    def test_recovery_keeps_direct_reads_visible_without_a_call_budget(self):
        from infinidev.engine.loop.llm_caller import LLMCaller

        schemas = [
            self._schema("read_file"),
            self._schema("recall_context"),
            self._schema("web_search"),
            self._schema("edit_file"),
            self._schema("execute_command"),
            self._schema("step_complete"),
        ]
        ctx = SimpleNamespace(
            planning_schemas=schemas,
            tool_schemas=schemas,
            state=SimpleNamespace(plan=None),
            suppress_discovery_this_step=True,
            semantic_recovery_context_calls=0,
            unlimited_recovery_reads=True,
        )

        available = LLMCaller._available_schemas(ctx, is_planning=False)

        assert [schema["function"]["name"] for schema in available] == [
            "read_file", "edit_file", "execute_command", "step_complete",
        ]

    def test_recovery_hides_read_when_target_source_is_live_until_pressure(self):
        from infinidev.engine.loop.llm_caller import LLMCaller

        schemas = [
            self._schema("read_file"),
            self._schema("edit_file"),
            self._schema("execute_command"),
            self._schema("step_complete"),
        ]
        state = SimpleNamespace(
            plan=None,
            last_prompt_tokens=100_000,
            read_delivery_revisions={
                json.dumps(["/workspace/module.py", None]): "1:10",
            },
        )
        ctx = SimpleNamespace(
            planning_schemas=schemas,
            tool_schemas=schemas,
            state=state,
            suppress_discovery_this_step=True,
            semantic_recovery_context_calls=0,
            unlimited_recovery_reads=True,
            max_context_tokens=1_000_000,
        )

        available = LLMCaller._available_schemas(ctx, is_planning=False)
        assert [schema["function"]["name"] for schema in available] == [
            "edit_file", "execute_command", "step_complete",
        ]

        state.last_prompt_tokens = 800_000
        available = LLMCaller._available_schemas(ctx, is_planning=False)
        assert [schema["function"]["name"] for schema in available] == [
            "read_file", "edit_file", "execute_command", "step_complete",
        ]


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

    def test_openai_compat_parse_error_detected(self):
        """OpenAI-compatible 500s that wrap a JSON parse error must be malformed,
        not transient — otherwise ``is_transient`` matches ``internal server error``
        and wastes all retries on a deterministic failure that then crashes the loop.
        """
        msg = (
            "litellm.InternalServerError: InternalServerError: Custom_openaiException "
            "- Failed to parse tool call arguments as JSON: "
            "[json.exception.parse_error.101] parse error at line 1, column 7993: "
            "syntax error while parsing value - invalid string: missing closing quote"
        )
        from infinidev.engine.llm_client import is_transient
        assert _is_malformed_tool_call(Exception(msg)) is True
        assert is_transient(Exception(msg)) is False


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


# ── Mis-nested sibling-param rescue (shared _normalize_single_call) ───────────


# (delimiters, raw single-call body) for each model family. The body is a tool
# call whose ``expected_output`` is emitted as a SIBLING of ``arguments``
# instead of inside it — small models do this regularly. After normalization
# the param MUST be folded into arguments so the tool receives it.
_SIBLING_BODY = (
    '{"name": "add_step", "arguments": {"title": "x"}, "expected_output": "y"}'
)

_FAMILY_CASES = [
    ("qwen", f"<tool_call>{_SIBLING_BODY}</tool_call>"),
    ("qwen_pipe", f"<|tool_call|>{_SIBLING_BODY}<|/tool_call|>"),
    ("mistral", f"[TOOL_CALLS] [{_SIBLING_BODY}]"),
    ("llama", f"<|python_tag|>{_SIBLING_BODY}<|eot_id|>"),
    ("function_call", f"<function_call>{_SIBLING_BODY}</function_call>"),
    ("tool_tag", f"<tool>{_SIBLING_BODY}</tool>"),
    ("manual_json", _SIBLING_BODY),
]


class TestMisNestedSiblingParamRescue:
    """Every family adapter must fold a mis-nested sibling param into arguments.

    Regression for the bug where each adapter carried its own un-rescued copy
    of ``_normalize_single_call`` and silently dropped sibling params.
    """

    @pytest.mark.parametrize("family,text", _FAMILY_CASES, ids=[c[0] for c in _FAMILY_CASES])
    def test_sibling_param_folded_into_arguments(self, family, text):
        calls = _parse_text_tool_calls(text)
        assert calls is not None, f"{family}: parsed nothing"
        assert calls[0]["name"] == "add_step"
        args = calls[0]["arguments"]
        assert args.get("title") == "x"
        # The mis-nested sibling must end up INSIDE arguments.
        assert args.get("expected_output") == "y", f"{family}: sibling param dropped"

    def test_mistral_array_value_not_truncated(self):
        """Mistral must capture the full balanced [...] even when an argument
        value is itself a JSON array (a non-greedy regex truncated at the
        first ``]``, corrupting the JSON and dropping the call)."""
        text = (
            '[TOOL_CALLS] '
            '[{"name": "add_step", "arguments": {"items": [1, 2, 3], "title": "x"}}]'
        )
        calls = _parse_text_tool_calls(text)
        assert calls is not None
        assert calls[0]["name"] == "add_step"
        assert calls[0]["arguments"]["items"] == [1, 2, 3]
        assert calls[0]["arguments"]["title"] == "x"

    def test_llama_splits_sequential_calls_and_drops_prose(self):
        """Llama must bound each <|python_tag|> payload and split multiple
        calls instead of greedily swallowing to EOF."""
        text = (
            '<|python_tag|>{"name": "a", "arguments": {"k": 1}}<|eom_id|>'
            "trailing prose that is not a tool call"
            '<|python_tag|>{"name": "b", "arguments": {}}<|eot_id|>'
        )
        calls = _parse_text_tool_calls(text)
        assert calls is not None
        assert [c["name"] for c in calls] == ["a", "b"]


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
