from __future__ import annotations

import json
from pathlib import Path

import pytest

from bench.agent_task_run import (
    AgentTaskRunConfig,
    _canonical_profile_sha,
    capture_tool_trace,
    changed_paths,
    configured_evaluation_runtime,
)


@pytest.mark.parametrize(
    ("provider", "model"),
    [
        ("anthropic", "claude-sonnet"),
        ("kimi", "kimi-k2"),
        ("minimax", "minimax-m2"),
        ("openai_subscription", "gpt-5.6-sol"),
    ],
)
def test_run_config_is_provider_neutral(tmp_path: Path, provider: str, model: str) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "provider": provider,
                "model": model,
                "model_identity": f"{provider}:{model}@revision",
                "min_request_interval_seconds": 2.0,
            }
        ),
        encoding="utf-8",
    )
    config = AgentTaskRunConfig.from_path(path)
    assert config.provider == provider
    assert config.model == model


def test_run_config_rejects_unsafe_request_pacing(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps(
            {
                "provider": "anthropic",
                "model": "claude",
                "model_identity": "anthropic:claude@revision",
                "min_request_interval_seconds": 1.9,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least 2 seconds"):
        AgentTaskRunConfig.from_path(path)


def test_changed_paths_ignores_runtime_caches(tmp_path: Path) -> None:
    source = tmp_path / "source"
    result = tmp_path / "result"
    source.mkdir()
    result.mkdir()
    (source / "a.py").write_text("old", encoding="utf-8")
    (result / "a.py").write_text("new", encoding="utf-8")
    (result / ".pytest_cache").mkdir()
    (result / ".pytest_cache" / "state").write_text("x", encoding="utf-8")
    assert changed_paths(source, result) == ("a.py",)


def test_provider_runtime_resolves_route_disables_retries_and_restores_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from infinidev.config.settings import settings

    profile = {
        "schema_version": 1,
        "provenance": "explicit_user",
        "name": "quality-and-control",
        "description": "Prefer quality with visible control.",
        "weights": {"quality": 1.0},
    }
    manifest = {
        "dataset_sha256": "dataset",
        "utility_profile": profile,
        "utility_profile_sha256": _canonical_profile_sha(profile),
        "conditions": {"baseline": None, "candidate": {"system_prompt": "Inspect evidence."}},
    }
    config = AgentTaskRunConfig(
        provider="kimi",
        model="kimi-k3",
        model_identity="kimi:kimi-k3@revision",
        api_key_env="KIMI_TEST_KEY",
    )
    monkeypatch.setenv("KIMI_TEST_KEY", "test-secret")
    previous = (
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
        settings.LLM_BASE_URL,
        settings.LLM_API_KEY,
        settings.LLM_NUM_RETRIES,
    )
    with configured_evaluation_runtime(config, "baseline", manifest, tmp_path):
        assert settings.LLM_PROVIDER == "kimi"
        assert settings.LLM_MODEL == "moonshot/kimi-k3"
        assert settings.LLM_BASE_URL == "https://api.moonshot.ai/v1"
        assert settings.LLM_API_KEY == "test-secret"
        assert settings.LLM_NUM_RETRIES == 0
    assert (
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
        settings.LLM_BASE_URL,
        settings.LLM_API_KEY,
        settings.LLM_NUM_RETRIES,
    ) == previous


def test_provider_runtime_fails_before_mutation_when_key_is_missing(tmp_path: Path) -> None:
    from infinidev.config.settings import settings

    profile = {
        "schema_version": 1,
        "provenance": "explicit_user",
        "name": "quality-and-control",
        "description": "Prefer quality with visible control.",
        "weights": {"quality": 1.0},
    }
    manifest = {
        "utility_profile": profile,
        "utility_profile_sha256": _canonical_profile_sha(profile),
        "conditions": {"baseline": None, "candidate": {"system_prompt": "Inspect evidence."}},
    }
    config = AgentTaskRunConfig(
        provider="minimax",
        model="MiniMax-M3",
        model_identity="minimax:MiniMax-M3@revision",
        api_key_env="DEFINITELY_MISSING_AGENT_TASK_KEY",
    )
    previous = (settings.LLM_PROVIDER, settings.LLM_MODEL, settings.LLM_BASE_URL)
    with pytest.raises(ValueError, match="environment variable is empty"):
        with configured_evaluation_runtime(config, "baseline", manifest, tmp_path):
            pass
    assert (settings.LLM_PROVIDER, settings.LLM_MODEL, settings.LLM_BASE_URL) == previous


def test_tool_trace_captures_exact_post_tool_evidence() -> None:
    from infinidev.engine.hooks.hooks import HookContext, HookEvent, hook_manager

    records: list[dict[str, object]] = []
    with capture_tool_trace(records):
        hook_manager.dispatch(
            HookContext(
                event=HookEvent.POST_TOOL,
                tool_name="execute_command",
                arguments={"command": "python tools/semantic_search.py available"},
                result='{"error":"semantic index unavailable"}',
                metadata={"tool_run_id": "trace-1"},
            )
        )
    assert records == [
        {
            "tool_run_id": "trace-1",
            "tool_name": "execute_command",
            "arguments": {"command": "python tools/semantic_search.py available"},
            "result": '{"error":"semantic index unavailable"}',
            "result_truncated": False,
            "failed": True,
        }
    ]
