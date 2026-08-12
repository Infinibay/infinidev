from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from bench.agent_task_run import (
    ADAPTIVE_BEHAVIOR_TREATMENT_MARKER,
    AgentTaskRunConfig,
    TASK_POLICY_TREATMENT_MARKER,
    _verify,
    _canonical_profile_sha,
    capture_tool_trace,
    changed_paths,
    configured_evaluation_runtime,
    copy_workspace,
    select_agent_tasks,
)
from bench.agent_task_eval import load_tasks


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
    (result / "node_modules" / "package").mkdir(parents=True)
    (result / "node_modules" / "package" / "index.js").write_text("x", encoding="utf-8")
    (result / "target" / "debug").mkdir(parents=True)
    (result / "target" / "debug" / "artifact").write_text("x", encoding="utf-8")
    (result / "build" / "generated").mkdir(parents=True)
    (result / "build" / "generated" / "artifact").write_text("x", encoding="utf-8")
    (result / "test").mkdir()
    (result / "test" / "test_default").write_bytes(b"binary")
    (result / "src" / "example.egg-info").mkdir(parents=True)
    (result / "src" / "example.egg-info" / "PKG-INFO").write_text("x", encoding="utf-8")
    assert changed_paths(source, result) == ("a.py",)


@pytest.mark.skipif(os.name == "nt", reason="directory symlinks require elevated privileges")
def test_workspace_copy_preserves_directory_symlinks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    result = tmp_path / "result"
    target = source / "target"
    target.mkdir(parents=True)
    (target / "data.txt").write_text("same", encoding="utf-8")
    (source / "alias").symlink_to("target", target_is_directory=True)

    copy_workspace(source, result)

    assert (result / "alias").is_symlink()
    assert changed_paths(source, result) == ()


def test_verifier_imports_src_layout_package_before_site_packages(tmp_path: Path) -> None:
    package = tmp_path / "src" / "local_eval_package"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("VALUE = 'workspace'\n", encoding="utf-8")

    verified = _verify(
        "{python} -c \"import local_eval_package; assert local_eval_package.VALUE == 'workspace'\"",
        tmp_path,
        5.0,
    )

    assert verified.returncode == 0, verified.stderr


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
        settings.EXECUTE_COMMANDS_PERMISSION,
        settings.FILE_OPERATIONS_PERMISSION,
        settings.TOOL_EFFECTS_PERMISSION,
    )
    with configured_evaluation_runtime(config, "baseline", manifest, tmp_path):
        assert settings.LLM_PROVIDER == "kimi"
        assert settings.LLM_MODEL == "moonshot/kimi-k3"
        assert settings.LLM_BASE_URL == "https://api.moonshot.ai/v1"
        assert settings.LLM_API_KEY == "test-secret"
        assert settings.LLM_NUM_RETRIES == 0
        assert settings.EXECUTE_COMMANDS_PERMISSION == "auto_approve"
        assert settings.FILE_OPERATIONS_PERMISSION == "auto_approve"
        assert settings.TOOL_EFFECTS_PERMISSION == "auto_approve"
    assert (
        settings.LLM_PROVIDER,
        settings.LLM_MODEL,
        settings.LLM_BASE_URL,
        settings.LLM_API_KEY,
        settings.LLM_NUM_RETRIES,
        settings.EXECUTE_COMMANDS_PERMISSION,
        settings.FILE_OPERATIONS_PERMISSION,
        settings.TOOL_EFFECTS_PERMISSION,
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


def test_task_policy_treatment_isolates_runtime_flags_and_skips_calibration(
    tmp_path: Path,
) -> None:
    from infinidev.config.settings import settings

    profile = {
        "schema_version": 1,
        "provenance": "explicit_user",
        "name": "task-policy-e2e",
        "description": "Compare conditional task policies only.",
        "weights": {"quality": 1.0},
    }
    manifest = {
        "utility_profile": profile,
        "utility_profile_sha256": _canonical_profile_sha(profile),
        "conditions": {
            "baseline": None,
            "candidate": {"system_prompt": TASK_POLICY_TREATMENT_MARKER},
        },
    }
    config = AgentTaskRunConfig(
        provider="minimax",
        model="MiniMax-M3",
        model_identity="minimax:MiniMax-M3@revision",
        treatment="task_policy",
    )
    previous = (
        settings.TASK_POLICIES_ENABLED,
        settings.TASK_POLICIES_SHADOW_MODE,
        settings.TASK_POLICIES_EMBEDDINGS_ENABLED,
        settings.TASK_POLICIES_LLM_FALLBACK_ENABLED,
        settings.TASK_POLICIES_EVIDENCE_GATED,
        settings.PROMPT_CALIBRATION_PROFILE,
    )
    with configured_evaluation_runtime(config, "candidate", manifest, tmp_path):
        assert settings.TASK_POLICIES_ENABLED
        assert not settings.TASK_POLICIES_SHADOW_MODE
        assert settings.TASK_POLICIES_EMBEDDINGS_ENABLED
        assert not settings.TASK_POLICIES_LLM_FALLBACK_ENABLED
        assert not settings.TASK_POLICIES_EVIDENCE_GATED
        assert settings.PROMPT_CALIBRATION_PROFILE == ""
    assert (
        settings.TASK_POLICIES_ENABLED,
        settings.TASK_POLICIES_SHADOW_MODE,
        settings.TASK_POLICIES_EMBEDDINGS_ENABLED,
        settings.TASK_POLICIES_LLM_FALLBACK_ENABLED,
        settings.TASK_POLICIES_EVIDENCE_GATED,
        settings.PROMPT_CALIBRATION_PROFILE,
    ) == previous


def test_adaptive_behavior_treatment_varies_only_runtime_interventions(
    tmp_path: Path,
) -> None:
    from infinidev.config.settings import settings

    profile = {
        "schema_version": 1,
        "provenance": "explicit_user",
        "name": "efficient-runtime",
        "description": "Keep quality while reducing model and tool cost.",
        "weights": {"quality": 1.0},
    }
    manifest = {
        "utility_profile": profile,
        "utility_profile_sha256": _canonical_profile_sha(profile),
        "conditions": {
            "baseline": None,
            "candidate": {"system_prompt": ADAPTIVE_BEHAVIOR_TREATMENT_MARKER},
        },
    }
    config = AgentTaskRunConfig(
        provider="minimax",
        model="MiniMax-M3",
        model_identity="minimax:MiniMax-M3@revision",
        treatment="adaptive_behavior",
    )
    previous = (
        settings.TASK_POLICIES_ENABLED,
        settings.ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED,
        settings.ADAPTIVE_RUNTIME_BEHAVIOR_SHADOW_MODE,
    )
    with configured_evaluation_runtime(config, "candidate", manifest, tmp_path):
        assert not settings.TASK_POLICIES_ENABLED
        assert settings.ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED
        assert not settings.ADAPTIVE_RUNTIME_BEHAVIOR_SHADOW_MODE
    with configured_evaluation_runtime(config, "baseline", manifest, tmp_path):
        assert not settings.TASK_POLICIES_ENABLED
        assert settings.ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED
        assert settings.ADAPTIVE_RUNTIME_BEHAVIOR_SHADOW_MODE
    assert (
        settings.TASK_POLICIES_ENABLED,
        settings.ADAPTIVE_RUNTIME_BEHAVIOR_ENABLED,
        settings.ADAPTIVE_RUNTIME_BEHAVIOR_SHADOW_MODE,
    ) == previous


def test_structured_evaluation_task_exposes_declared_verifier() -> None:
    from types import SimpleNamespace

    from bench.agent_task_run import _structured_evaluation_task

    task = SimpleNamespace(
        request="Corrige el comportamiento inválido conservando el contrato público existente.",
        category="bugfix",
        verify_command="npm test",
    )

    structured = _structured_evaluation_task(task, None)

    assert structured.kind == "bugfix"
    assert structured.derived_verification_criteria == [
        "Run `npm test` and require exit code 0 before completion."
    ]


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


def test_task_selection_supports_small_live_subsets() -> None:
    tasks = load_tasks(Path("bench/agent_task_pilot.tasks.jsonl"))

    selected = select_agent_tasks(
        tasks,
        split="validation",
        include_drafts=True,
        task_ids=("reversible-ambiguity", "tool-failure-recovery"),
    )

    assert [task.id for task in selected] == [
        "reversible-ambiguity",
        "tool-failure-recovery",
    ]


def test_task_selection_rejects_unknown_ids() -> None:
    tasks = load_tasks(Path("bench/agent_task_pilot.tasks.jsonl"))

    with pytest.raises(ValueError, match="unknown agent task ids"):
        select_agent_tasks(
            tasks,
            split="validation",
            include_drafts=True,
            task_ids=("typo-task",),
        )
