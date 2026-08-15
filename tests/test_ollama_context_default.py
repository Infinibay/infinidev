"""Regression tests for Ollama-owned context allocation by default."""

from __future__ import annotations

from infinidev.config.llm import get_litellm_params
from infinidev.config.settings import Settings, settings


def test_ollama_num_ctx_defaults_to_server_control(monkeypatch) -> None:
    monkeypatch.delenv("INFINIDEV_OLLAMA_NUM_CTX", raising=False)

    assert Settings().OLLAMA_NUM_CTX == 0


def test_default_ollama_request_omits_num_ctx(monkeypatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "LLM_MODEL", "ollama_chat/qwen3.8:latest")
    monkeypatch.setattr(settings, "OLLAMA_NUM_CTX", 0)

    assert "num_ctx" not in get_litellm_params()


def test_explicit_ollama_num_ctx_is_preserved(monkeypatch) -> None:
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "LLM_MODEL", "ollama_chat/qwen3.8:latest")
    monkeypatch.setattr(settings, "OLLAMA_NUM_CTX", 131_072)

    assert get_litellm_params()["num_ctx"] == 131_072
