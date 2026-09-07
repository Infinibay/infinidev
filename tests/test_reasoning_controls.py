"""Reasoning contracts, command parity, and the locked SDK's outgoing payloads."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from infinidev.config.reasoning import (
    apply_reasoning,
    effort_listing,
    effort_profile,
    normalize_model_request,
)
from infinidev.config.settings import settings
from infinidev.config.thinking_budget import apply_thinking_budget


@pytest.fixture(autouse=True)
def reasoning_settings(monkeypatch):
    monkeypatch.setattr(settings, "THINKING_ENABLED", True)
    monkeypatch.setattr(settings, "THINKING_BUDGET", "medium")
    monkeypatch.setattr(settings, "THINKING_BUDGET_TOKENS", 4096)


@pytest.mark.parametrize("provider,model,choices", [
    ("openai", "gpt-6-astra", "low medium high xhigh max"),
    ("openai", "gpt-5.6-sol", "none low medium high xhigh max"),
    ("openai", "gpt-5.6-luna", "none low medium high xhigh max"),
    ("openai", "gpt-5.5-pro", "medium high xhigh"),
    ("openai", "gpt-5.1", "none low medium high"),
    ("openai", "gpt-5-mini", "minimal low medium high"),
    ("openai", "o3", "low medium high"),
    ("anthropic", "claude-fable-5-1", "low medium high xhigh max"),
    ("anthropic", "claude-opus-5", "off low medium high xhigh max"),
    ("anthropic", "claude-sonnet-5", "off low medium high xhigh max"),
    ("anthropic", "claude-sonnet-4-6", "off low medium high max"),
    ("anthropic", "claude-opus-4-5-20251101", "off low medium high"),
    ("anthropic", "claude-haiku-4-5", "off low medium high custom"),
    ("zai", "glm-5.3", "low high max"),
    ("zai_coding", "glm-5.3-flash", "low high max"),
    ("zai", "glm-5.2", "off high max"),
    ("zai", "glm-4.7", "off on"),
    ("qwen", "qwen3.8-max", "off low medium xhigh"),
    ("qwen_subscription", "qwen3.8-flash", "off low medium xhigh"),
    ("qwen", "qwen3.8-2.4t-a95b", "low medium xhigh"),
    ("qwen", "qwen3-32b", "off low medium high custom"),
    ("ollama", "gpt-oss:20b", "low medium high"),
    ("ollama", "qwen3:8b", "off on"),
    ("gemini", "gemini-3-pro-preview", "low high"),
    ("deepseek", "deepseek-v4-pro", "off low high max"),
    ("mistral", "mistral-small-latest", "none high"),
    ("openai", "gpt-4.1", ""),
    ("openai_compatible", "custom-model", ""),
    ("qwen", "qwen3-coder-plus", ""),
])
def test_exact_model_choices(provider, model, choices):
    assert effort_profile(provider, model).choices == tuple(choices.split())


@pytest.mark.parametrize("model", ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-terra"])
@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_openai_levels_survive_actual_responses_bridge(monkeypatch, model, level):
    from infinidev.config.llm import apply_provider_transport
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    monkeypatch.setattr(settings, "THINKING_BUDGET", level)
    params = {"model": f"openai/{model}", "temperature": 0.2, "top_p": 0.9}
    apply_provider_transport(params, "openai")
    normalize_model_request(params, bridge=True)
    assert params["model"] == f"openai/responses/{model}"
    assert "temperature" not in params and "top_p" not in params
    params.pop("model")
    payload = LiteLLMResponsesTransformationHandler().transform_request(
        model, [{"role": "user", "content": "Inspect the source"}],
        params, {}, {}, MagicMock(),
    )
    assert payload["reasoning"]["effort"] == level


@pytest.mark.parametrize("model,level", [
    (model, level)
    for model in ("gpt-6-astra", "gpt-5.6-sol")
    for level in ("low", "medium", "high", "xhigh", "max")
] + [("gpt-5.6-sol", "none")])
@pytest.mark.parametrize("stream", [False, True])
def test_openai_effort_survives_full_completion_to_http(monkeypatch, model, level, stream):
    import litellm

    from infinidev.config.llm import get_litellm_params
    from infinidev.config.usage import UsageSelection
    from infinidev.engine import usage
    from infinidev.engine.llm_client import call_llm
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "LLM_MODEL", f"openai/{model}")
    monkeypatch.setattr(settings, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(settings, "LLM_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setattr(settings, "LLM_TEMPERATURE", 0.2)
    monkeypatch.setattr(settings, "THINKING_BUDGET", level)
    monkeypatch.setattr(litellm, "drop_params", False)
    ledger = usage.UsageLedger()
    monkeypatch.setattr(usage, "usage_ledger", ledger)
    requests = []

    def post(self, url, **kwargs):
        payload = kwargs["json"] if "json" in kwargs else json.loads(kwargs["data"])
        requests.append((url, payload))
        response = {
            "id": "resp_test", "object": "response", "created_at": 1,
            "model": model, "status": "completed", "error": None,
            "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                        "status": "completed", "content": [{
                            "type": "output_text", "text": "Checked.", "annotations": [],
                        }]}],
            "usage": {"input_tokens": 8, "output_tokens": 2, "total_tokens": 10},
        }
        if kwargs.get("stream"):
            item = response["output"][0]
            part = item["content"][0]
            indices = {"item_id": item["id"], "output_index": 0, "content_index": 0}
            events = [
                {"type": "response.created", "response": {
                    **response, "output": [], "status": "in_progress",
                }},
                {"type": "response.output_item.added", "output_index": 0, "item": {
                    **item, "content": [], "status": "in_progress",
                }},
                {"type": "response.content_part.added", **indices,
                 "part": {**part, "text": ""}},
                {"type": "response.output_text.delta", **indices, "delta": "Checked."},
                {"type": "response.output_text.done", **indices, "text": "Checked."},
                {"type": "response.content_part.done", **indices, "part": part},
                {"type": "response.output_item.done", "output_index": 0, "item": item},
                {"type": "response.completed", "response": response},
            ]
            data = "".join(
                f"event: {event['type']}\ndata: "
                + json.dumps({**event, "sequence_number": i}) + "\n\n"
                for i, event in enumerate(events)
            )
            return httpx.Response(200, request=httpx.Request("POST", url),
                                  headers={"content-type": "text/event-stream"}, text=data)
        return httpx.Response(200, request=httpx.Request("POST", url), json=response)

    monkeypatch.setattr(HTTPHandler, "post", post)
    params = get_litellm_params()
    response = call_llm(params, [{"role": "user", "content": "Inspect the source"}],
                        tools=[{"type": "function", "function": {
                            "name": "read_file", "parameters": {
                                "type": "object", "properties": {},
                            },
                        }}], retry_attempts=1,
                        on_thinking_chunk=(lambda text: None) if stream else None)
    assert response.choices[0].message.content == "Checked."
    snapshot = ledger.snapshot(UsageSelection.current().request_params())
    assert snapshot["requests"] == 1
    assert snapshot["input_tokens"] == 8
    assert snapshot["output_tokens"] == 2
    assert len(requests) == 1
    url, payload = requests[0]
    assert url == "https://api.openai.com/v1/responses"
    assert payload["model"] == model
    assert payload["reasoning"]["effort"] == level
    assert "reasoning_effort" not in payload
    assert "allowed_openai_params" not in payload
    assert payload["tools"][0]["type"] == "function"
    assert payload["tool_choice"] == "auto"
    assert payload["store"] is False
    assert "reasoning.encrypted_content" in payload["include"]
    if level == "none":
        assert payload["temperature"] == 0.2
    else:
        assert "temperature" not in payload


@pytest.mark.parametrize("model", ["claude-fable-5-1", "claude-opus-5", "claude-sonnet-5"])
@pytest.mark.parametrize("level", ["low", "medium", "high", "xhigh", "max"])
def test_anthropic_effort_survives_actual_messages_transformation(monkeypatch, model, level):
    from infinidev.config.llm import apply_provider_transport
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    monkeypatch.setattr(settings, "THINKING_BUDGET", level)
    params = {"model": f"anthropic/{model}", "temperature": 0.2,
              "tool_choice": "required", "max_tokens": 64000,
              "extra_body": {"output_config": {"format": {"type": "json_schema"}}}}
    apply_provider_transport(params, "anthropic")
    assert params["tool_choice"] == "auto"
    assert "temperature" not in params
    assert params["extra_body"]["output_config"]["format"] == {"type": "json_schema"}
    params.pop("model")
    params.update(params.pop("extra_body"))
    params["output_config"].pop("format")
    payload = AnthropicConfig().transform_request(
        model, [{"role": "user", "content": "Inspect the source"}], params, {}, {},
    )
    assert payload["output_config"]["effort"] == level
    assert payload["thinking"] == {"type": "adaptive"}


def test_manual_anthropic_budget_has_output_headroom(monkeypatch):
    monkeypatch.setattr(settings, "THINKING_BUDGET", "high")
    params = {"max_tokens": 1024}
    apply_thinking_budget(params, "anthropic", "claude-haiku-4-5")
    assert params["thinking"]["budget_tokens"] == 16384
    assert params["max_tokens"] > 16384


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-6-astra"), ("anthropic", "claude-fable-5-1"),
    ("zai", "glm-5.3"), ("qwen", "qwen3.8-2.4t-a95b"),
])
def test_mandatory_thinking_with_disabled_preference(monkeypatch, provider, model):
    monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
    monkeypatch.setattr(settings, "LLM_MODEL", model)
    monkeypatch.setattr(settings, "THINKING_ENABLED", False)
    params = {}
    apply_thinking_budget(params, provider, model)
    assert "disabled" not in str(params)
    assert "cannot be disabled" in effort_listing()


def test_qwen_effort_and_budget_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(settings, "THINKING_BUDGET", "high")
    params = {"extra_body": {"thinking_budget": 2000, "preserve_thinking": True}}
    apply_reasoning(params, "qwen", "qwen3.8-max")
    assert params["extra_body"] == {
        "enable_thinking": True, "reasoning_effort": "xhigh", "preserve_thinking": True,
    }


def test_codex_uses_own_catalog_including_new_levels(monkeypatch):
    monkeypatch.setattr("infinidev.config.codex_catalog.reasoning_levels",
                        lambda model: ["low", "high", "max", "ultra"])
    assert effort_profile("openai_subscription", "gpt-6-astra").choices == (
        "low", "high", "max", "ultra",
    )
    assert "ultra" not in effort_profile("openai", "gpt-6-astra").choices


def test_openrouter_uses_gateway_metadata_not_native_provider(monkeypatch):
    monkeypatch.setattr("infinidev.config.openrouter_reasoning.model_reasoning", lambda model: {
        "supported_efforts": ["high", "low"], "mandatory": True,
    })
    monkeypatch.setattr(settings, "THINKING_BUDGET", "medium")
    assert effort_profile("openrouter", "openrouter/z-ai/glm-5.3").choices == ("low", "high")
    params = {}
    apply_reasoning(params, "openrouter", "openrouter/z-ai/glm-5.3")
    assert params == {"extra_body": {"reasoning": {"enabled": True, "effort": "high"}}}


@pytest.mark.parametrize("provider,model", [
    ("openai", "gpt-6-astra"), ("anthropic", "claude-sonnet-4-6"),
    ("zai", "glm-5.3"), ("qwen", "qwen3.8-max"), ("openai", "gpt-4.1"),
])
def test_cli_and_tui_show_same_model_specific_listing(monkeypatch, capsys, provider, model):
    from infinidev.cli.commands import handle_effort_command
    from infinidev.ui.handlers.commands import handle_effort

    monkeypatch.setattr(settings, "LLM_PROVIDER", provider)
    monkeypatch.setattr(settings, "LLM_MODEL", model)
    handle_effort_command(["/effort"])
    app = MagicMock()
    handle_effort(app, ["/effort"])
    assert capsys.readouterr().out.strip() == app.add_message.call_args.args[1]


@pytest.mark.parametrize("tui", [False, True])
def test_effort_command_reenables_thinking_and_rejects_invalid_level(monkeypatch, tui):
    from infinidev.cli.commands import handle_effort_command
    from infinidev.ui.handlers.commands import handle_effort

    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.setattr(settings, "THINKING_ENABLED", False)
    save = MagicMock()
    monkeypatch.setattr(type(settings), "save_user_settings", save)
    monkeypatch.setattr("infinidev.config.settings.reload_all", lambda: None)
    invoke = (lambda parts: handle_effort(MagicMock(), parts)) if tui else handle_effort_command
    invoke(["/effort", "high"])
    save.assert_called_once_with({"THINKING_BUDGET": "high", "THINKING_ENABLED": True})
    invoke(["/effort", "xhigh"])
    assert save.call_count == 1


@pytest.mark.parametrize("principal,model,expected", [
    ("anthropic", "openai/gpt-6-astra", {"reasoning_effort": "medium"}),
    ("openai", "anthropic/claude-fable-5-1", {"thinking": {"type": "adaptive"}}),
])
def test_helper_effort_uses_its_provider_not_principals(monkeypatch, principal, model, expected):
    from infinidev.engine.llm_client import call_llm

    monkeypatch.setattr(settings, "LLM_PROVIDER", principal)
    seen = {}
    monkeypatch.setattr("litellm.completion", lambda **kwargs: seen.update(kwargs) or object())
    call_llm({"model": model}, [{"role": "user", "content": "Check"}], retry_attempts=1)
    for key, value in expected.items():
        assert seen[key] == value
    if model.startswith("openai/"):
        assert "thinking" not in seen
    else:
        assert "reasoning_effort" not in seen


def test_new_anthropic_native_schema_preserves_effort(monkeypatch):
    monkeypatch.setattr(settings, "THINKING_BUDGET", "max")
    params = {"model": "anthropic/claude-fable-5-1", "response_format": {
        "type": "json_schema", "json_schema": {"name": "result", "schema": {
            "type": "object", "properties": {"ok": {"type": "boolean"}},
            "required": ["ok"], "additionalProperties": False,
        }},
    }}
    apply_reasoning(params, "anthropic", params["model"])
    normalize_model_request(params, bridge=True)
    assert "response_format" not in params
    assert "tools" not in params
    output = params["extra_body"]["output_config"]
    assert output["effort"] == "max"
    assert output["format"]["schema"]["properties"] == {"ok": {"type": "boolean"}}


def test_gemini_flash_can_disable_thinking(monkeypatch):
    monkeypatch.setattr(settings, "THINKING_ENABLED", False)
    params = {}
    apply_reasoning(params, "gemini", "gemini-2.5-flash")
    assert params == {"thinking": {"thinking_budget": 0}}
    assert "off" not in effort_profile("gemini", "gemini-2.5-pro").choices


def test_openai_opaque_reasoning_survives_tool_round_trip():
    from infinidev.engine.behavior.reasoning_content import (
        extract_reasoning, reasoning_history_fields,
    )
    from litellm.completion_extras.litellm_responses_transformation.transformation import (
        LiteLLMResponsesTransformationHandler,
    )

    item = {"id": "rs_example", "type": "reasoning", "summary": [],
            "encrypted_content": "opaque-test-value"}
    assistant = {"role": "assistant", "content": "", "tool_calls": [{
        "id": "call_1", "type": "function",
        "function": {"name": "read_file", "arguments": '{"path":"cache.py"}'},
    }], **reasoning_history_fields({"reasoning_items": [item]})}
    assert extract_reasoning(assistant).text == ""
    payload = LiteLLMResponsesTransformationHandler().transform_request(
        "gpt-6-astra", [assistant, {"role": "tool", "tool_call_id": "call_1",
                                   "content": "return x.detach()"}], {}, {}, {}, MagicMock(),
    )
    assert payload["input"][0]["encrypted_content"] == "opaque-test-value"
    assert payload["input"][1]["type"] == "function_call"
    assert payload["input"][2]["type"] == "function_call_output"
