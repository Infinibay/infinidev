"""Bounded semantic recovery keeps routing separate from permissions."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from infinidev.engine.orchestration import semantic_routing


def _response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


@pytest.mark.parametrize("raw", ["garbage", '{}', '{"kind":"write","understanding":"x"}',
    '{"kind":"escalate","understanding":""}',
    '{"kind":"escalate","understanding":"x","permission":true}'])
def test_invalid_decision_does_not_infer_work(monkeypatch, raw):
    monkeypatch.setattr(semantic_routing, "call_llm", lambda *a, **kw: _response(raw))
    monkeypatch.setattr(semantic_routing, "get_litellm_params_for_behavior", lambda: {"model":"test"})
    assert semantic_routing.classify_chat_response("go", "reply", recent_context="") is None


def test_classifier_is_bounded_and_treats_transcript_as_data(monkeypatch):
    seen = {}
    monkeypatch.setattr(semantic_routing, "get_litellm_params_for_behavior", lambda: {"model":"small", "timeout":999})
    def call(params, **kwargs):
        seen.update(params=params, **kwargs)
        return _response('{"kind":"escalate","understanding":"Apply agreed fix"}')
    monkeypatch.setattr(semantic_routing, "call_llm", call)
    decision = semantic_routing.classify_chat_response("dale", "Voy a corregirlo.", recent_context="x" * 50000 + "agreed fix")
    assert decision.kind == "escalate"
    assert seen["params"]["model"] == "small"
    assert seen["params"]["max_tokens"] == 256
    assert seen["params"]["timeout"] <= 30
    assert seen["params"]["num_retries"] == 0
    assert seen["retry_attempts"] == 1
    assert seen["thinking_enabled"] is False
    assert not seen.get("tools")
    data = json.loads(seen["messages"][1]["content"])
    assert data["current_user_request"] == "dale"
    assert data["assistant_reply"] == "Voy a corregirlo."
    assert len(data["recent_context"]) <= 8000
    assert data["recent_context"].endswith("agreed fix")
    prompt = seen["messages"][0]["content"]
    assert "permission" in prompt.lower()
    assert "quoted" in prompt.lower()
    assert "approval" in prompt.lower()


def test_classifier_failure_keeps_original_decision(monkeypatch):
    monkeypatch.setattr(semantic_routing, "get_litellm_params_for_behavior", lambda: {"model":"test"})
    def fail(*args, **kwargs):
        raise TimeoutError("bounded timeout")
    monkeypatch.setattr(semantic_routing, "call_llm", fail)
    assert semantic_routing.classify_chat_response("go", "reply", recent_context="") is None


def test_local_alias_disables_template_thinking_at_transport(monkeypatch):
    import litellm

    from infinidev.config.settings import settings

    monkeypatch.setattr(settings, "BEHAVIOR_LLM_PROVIDER", "llama_cpp")
    monkeypatch.setattr(settings, "LLM_PROVIDER", "llama_cpp")
    monkeypatch.setattr(semantic_routing, "get_litellm_params_for_behavior", lambda: {
        "model": "custom_openai/Strata",
        "api_base": "http://localhost:9999/v1",
        "extra_body": {"chat_template_kwargs": {"enable_thinking": True, "keep": "value"}},
    })
    from infinidev.config.model_capabilities import ModelCapabilities

    monkeypatch.setattr(
        "infinidev.config.model_capabilities.get_model_capabilities", ModelCapabilities,
    )
    seen = []

    def completion(**kwargs):
        seen.append(kwargs)
        return _response('{"kind":"respond","understanding":""}')

    monkeypatch.setattr(litellm, "completion", completion)
    result = semantic_routing.classify_chat_response(
        "Explain this file", "Here is the explanation.", recent_context="",
    )
    assert result.kind == "respond"
    assert len(seen) == 1
    assert seen[0]["max_tokens"] == 256
    assert seen[0]["num_retries"] == 0
    assert seen[0]["extra_body"]["chat_template_kwargs"] == {
        "enable_thinking": False, "keep": "value",
    }
