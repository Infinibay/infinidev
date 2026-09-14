from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from bench.agent_task_run import (
    ADAPTIVE_BEHAVIOR_TREATMENT_MARKER,
    AgentTaskRunConfig,
    TASK_POLICY_TREATMENT_MARKER,
    _frozen_task_policy_profile,
    _verify,
    _canonical_profile_sha,
    capture_tool_trace,
    changed_paths,
    configured_evaluation_runtime,
    copy_workspace,
    select_agent_tasks,
)
from bench.agent_task_eval import file_sha256, load_tasks


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


def test_run_config_requires_complete_prediction_report_identity(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({
            "provider": "minimax",
            "model": "MiniMax-M3",
            "model_identity": "minimax:MiniMax-M3@revision",
            "treatment": "task_policy",
            "task_policy_prediction_report": "/tmp/report.json",
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="path and SHA-256"):
        AgentTaskRunConfig.from_path(path)


def test_frozen_task_policy_profile_uses_scores_without_granting_authority(
    tmp_path: Path,
) -> None:
    report = tmp_path / "nemotron.json"
    report.write_text(
        json.dumps({
            "model": "nvidia/Nemotron-3-Embed-1B-BF16",
            "version": "manual-task-policy-e5-finetune-cv-v4",
            "folds": [{
                "prediction_only": {
                    "ids": ["bug-task"],
                    "predictions": [["bugfix.root_cause"]],
                    "method_scores": [[0.91, 0.02, 0.03, 0.04, 0.05, 0.06]],
                }
            }],
        }),
        encoding="utf-8",
    )

    profile = _frozen_task_policy_profile(
        "Fix the broken parser.",
        "bug-task",
        report_path=str(report),
        expected_sha256=file_sha256(report),
    )

    assert profile.operations == ("bugfix",)
    assert profile.authority == ("answer", "diagnose", "modify")
    assert [item.id for item in profile.selected_policies] == ["bugfix.root_cause"]
    assert profile.selected_policies[0].source == "embedding"
    assert profile.selected_policies[0].score == pytest.approx(0.91)
    assert not profile.semantic_abstained


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


# ── withheld verification contracts ───────────────────────────────────
#
# A task that ships its own verifier in the workspace measures reading
# comprehension: the model opens the file and satisfies the assertions it
# finds. Withheld paths are removed from the agent's copy and restored only to
# be run, so the verdict measures the implementation.


def test_withheld_paths_are_absent_while_the_agent_works() -> None:
    from bench.agent_task_run import _restore_withheld, _withhold_paths

    tasks = load_tasks(Path("bench/engine_eval_v2.tasks.jsonl"))
    task = next(t for t in tasks if t.id == "pricing-rounding")
    assert task.withheld_paths == ("verify_contract.py",)

    with tempfile.TemporaryDirectory() as temp:
        workspace = Path(temp) / "repo"
        copy_workspace(
            Path("bench/agent_task_fixtures") / task.repository_fixture, workspace,
        )

        saved = _withhold_paths(workspace, task.withheld_paths)

        assert not (workspace / "verify_contract.py").exists()
        assert (workspace / "src" / "pricing.py").exists()
        assert _restore_withheld(workspace, saved) == ()
        assert (workspace / "verify_contract.py").exists()


def test_a_file_the_agent_wrote_where_the_contract_belongs_is_reported() -> None:
    from bench.agent_task_run import _restore_withheld, _withhold_paths

    with tempfile.TemporaryDirectory() as temp:
        workspace = Path(temp) / "repo"
        workspace.mkdir(parents=True)
        (workspace / "verify_contract.py").write_text("The real contract\n")
        saved = _withhold_paths(workspace, ("verify_contract.py",))

        (workspace / "verify_contract.py").write_text("Agent-authored decoy\n")

        assert _restore_withheld(workspace, saved) == ("verify_contract.py",)
        assert (workspace / "verify_contract.py").read_text() == "The real contract\n"


def test_a_withheld_path_outside_the_workspace_is_refused() -> None:
    from bench.agent_task_run import _withhold_paths

    with tempfile.TemporaryDirectory() as temp:
        workspace = Path(temp) / "repo"
        workspace.mkdir(parents=True)

        with pytest.raises(ValueError, match="escapes the workspace"):
            _withhold_paths(workspace, ("../../etc/passwd",))


@pytest.mark.parametrize(
    "task_id,fixture,source_file",
    (
        ("pricing-rounding", "pricing_rounding", "pricing.py"),
        ("cart-immutability", "cart_immutability", "cart.py"),
        ("options-override", "options_override", "config.py"),
        ("wide-sum", "wide_tree", "mod_27.py"),
    ),
)
def test_every_hidden_contract_fails_pristine_and_passes_its_reference(
    task_id: str, fixture: str, source_file: str,
) -> None:
    """A hidden contract has to be a real regression, not a task that always passes.

    Both directions matter. A contract that already passes measures nothing; one
    its own reference cannot satisfy is an unwinnable task, and an unwinnable
    task looks exactly like a hard one in the results.
    """
    import shutil
    import subprocess
    import sys

    fixture_root = Path("bench/agent_task_fixtures") / fixture
    reference_root = Path("bench/agent_task_reference_solutions") / task_id

    with tempfile.TemporaryDirectory() as temp:
        workspace = Path(temp) / "repo"
        copy_workspace(fixture_root, workspace)

        pristine = subprocess.run(
            [sys.executable, "verify_contract.py"],
            cwd=workspace, capture_output=True, text=True, timeout=60,
        )
        assert pristine.returncode != 0, (
            f"{task_id}: the hidden contract passes before the work is done"
        )

        shutil.copy(
            reference_root / "src" / source_file,
            workspace / "src" / source_file,
        )
        solved = subprocess.run(
            [sys.executable, "verify_contract.py"],
            cwd=workspace, capture_output=True, text=True, timeout=60,
        )
        assert solved.returncode == 0, f"{task_id}: {solved.stdout}{solved.stderr}"


def test_the_audit_tasks_verifier_stays_visible() -> None:
    """The audit's verifier judges the report, so hiding it would be a lottery.

    The rule the corpus settled on: withhold a verifier that judges behaviour,
    never one that judges wording. `research_audit`'s checks name the modules and
    count line references, which is wording — so it ships in the workspace and
    the task is honest about its gate being weak.
    """
    tasks = {
        task.id: task
        for task in load_tasks(Path("bench/engine_eval_v8.tasks.jsonl"))
    }

    assert tasks["research-audit"].withheld_paths == ()
    assert tasks["research-audit"].expected_changed_paths == ("AUDIT.md",)
    assert "src/*" in tasks["research-audit"].forbidden_changed_paths


@pytest.mark.parametrize(
    "task_id,fixture",
    (
        ("pricing-rounding", "pricing_rounding"),
        ("cart-immutability", "cart_immutability"),
        ("options-override", "options_override"),
        ("wide-sum", "wide_tree"),
    ),
)
def test_the_hidden_contract_is_not_in_the_agents_workspace(
    task_id: str, fixture: str,
) -> None:
    """The point of the withhold is that the model cannot read its grader."""
    from bench.agent_task_run import _withhold_paths

    tasks = {
        task.id: task for task in load_tasks(Path("bench/engine_eval_v8.tasks.jsonl"))
    }
    task = tasks[task_id]
    assert task.withheld_paths == ("verify_contract.py",)

    with tempfile.TemporaryDirectory() as temp:
        workspace = Path(temp) / "repo"
        copy_workspace(Path("bench/agent_task_fixtures") / fixture, workspace)

        saved = _withhold_paths(workspace, task.withheld_paths)

        assert saved == {"verify_contract.py": saved["verify_contract.py"]}
        assert not (workspace / "verify_contract.py").exists()
        # The visible tests stay: the agent needs something to run and the task
        # tells it to run them.
        assert (workspace / "tests").is_dir()


def test_the_runners_interpreter_is_reachable_as_python() -> None:
    """A task that says "run the tests" needs an interpreter to run them with."""
    import shutil
    import subprocess
    import sys

    from bench.agent_task_run import interpreter_on_path

    with interpreter_on_path():
        resolved = shutil.which("python")
        assert resolved is not None, (
            "the agent types `python -m pytest`; without the runner's "
            "interpreter on PATH it gets 'command not found' and builds a "
            "virtualenv inside the task repository instead"
        )
        assert Path(resolved).resolve() == Path(sys.executable).resolve()
        probe = subprocess.run(
            ["python", "-c", "print('reachable')"],
            capture_output=True, text=True, timeout=60,
        )
        assert probe.returncode == 0, probe.stderr
        assert "reachable" in probe.stdout


def test_a_resumed_campaign_reuses_the_executions_it_already_paid_for() -> None:
    """A transient provider timeout stops the runner by design.

    What it must not do is throw away the completed half of a long campaign, so
    the driver reads the units already recorded and asks for the rest.
    """
    from bench.agent_task_ab import _completed_units

    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "observations.jsonl"
        assert _completed_units(path) == frozenset(), "a missing file has none"

        path.write_text(
            "\n".join(
                json.dumps({
                    "task_id": task_id,
                    "repetition": repetition,
                    "condition": "baseline",
                    "error": error,
                })
                for task_id, repetition, error in (
                    ("a-task", 0, ""),
                    ("a-task", 1, ""),
                    # The runner writes the provider error and then stops. The
                    # row is a diagnostic: counting it as done fails a task the
                    # model never finished and hides it from the rerun.
                    ("b-task", 0, "Timeout: litellm.Timeout: APITimeoutError"),
                    ("b-task", 1, ""),
                )
            )
            + "\n",
            encoding="utf-8",
        )

        assert _completed_units(path) == frozenset({
            ("a-task", 0, "baseline"),
            ("a-task", 1, "baseline"),
            ("b-task", 1, "baseline"),
        })


def test_a_resumed_pass_clears_only_the_artifacts_it_will_rewrite() -> None:
    """The runner refuses an existing artifact directory.

    A unit that is being re-run has a stale directory left by the attempt that
    stopped on a provider error. One that is being reused must keep its
    artifacts, which are the evidence behind its row.
    """
    from bench.agent_task_ab import _clear_stale_artifacts

    with tempfile.TemporaryDirectory() as temp:
        artifacts = Path(temp) / "artifacts"
        for name in (
            "test-selection.r0.baseline",
            "options-override.r0.baseline",
            "cart-immutability.r1.baseline",
        ):
            (artifacts / name).mkdir(parents=True)

        _clear_stale_artifacts(
            artifacts,
            frozenset({("cart-immutability", 1, "baseline")}),
            ("baseline",),
        )

        assert sorted(p.name for p in artifacts.iterdir()) == [
            "cart-immutability.r1.baseline"
        ]


def test_the_task_workspace_is_a_git_repository() -> None:
    """The agent is told to review with git_diff; the workspace must support it.

    62 of the 189 failed tool calls across the recorded campaigns, 33% of all
    failures, were `fatal: not a git repository`.
    """
    import shutil as _shutil
    import subprocess
    import sys

    from bench.agent_task_run import init_git_workspace

    if not _shutil.which("git"):
        pytest.skip("git is not installed on this host")

    with tempfile.TemporaryDirectory() as temp:
        workspace = Path(temp) / "repo"
        copy_workspace(Path("bench/agent_task_fixtures/pricing_rounding"), workspace)

        assert init_git_workspace(workspace) is True

        status = subprocess.run(
            ["git", "status", "--short"], cwd=workspace,
            capture_output=True, text=True, timeout=60,
        )
        assert status.returncode == 0, status.stderr
        assert status.stdout.strip() == "", (
            "the fixture baseline must be committed, so a clean status means "
            "the agent has not changed anything yet"
        )

        (workspace / "src" / "pricing.py").write_text("x = 1\n", encoding="utf-8")
        diff = subprocess.run(
            ["git", "diff", "--name-only"], cwd=workspace,
            capture_output=True, text=True, timeout=60,
        )
        assert "src/pricing.py" in diff.stdout


def test_the_evaluation_registers_the_project_its_tools_write_under(temp_db) -> None:
    """A foreign key is not something a task should discover at runtime.

    ``findings``, ``artifacts`` and the knowledge tables all reference
    ``projects(id)``. The evaluation invents the id, so without this row every
    write under it failed the foreign key and the DB-backed surface went
    untested behind that error.
    """
    import sqlite3

    from bench.agent_task_run import ensure_project

    project_id = 0x5EED1234
    ensure_project(project_id, "agent-task::probe")

    with sqlite3.connect(temp_db) as conn:
        row = conn.execute(
            "SELECT name FROM projects WHERE id = ?", (project_id,)
        ).fetchone()
        assert row is not None, "the project row the tools write under is missing"
        assert row[0] == "agent-task::probe"

    # Idempotent: a resumed run calls it again.
    ensure_project(project_id, "agent-task::probe")

    # And the foreign key the review found actually holds now.
    with sqlite3.connect(temp_db) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute(
            "INSERT INTO findings (project_id, topic, content) VALUES (?, ?, ?)",
            (project_id, "probe", "content"),
        )


def test_the_runner_can_measure_the_default_engine_mode() -> None:
    """The direct loop cannot see the pipeline the product defaults to.

    ``TASK_ENGINE_MODE`` selects an engine adapter; the default one runs the
    chat agent, the spec elaboration and the review phase around the loop. A
    benchmark that only calls ``LoopEngine.execute`` can never measure any of
    that, so the config can ask for the real pipeline instead.
    """
    config = AgentTaskRunConfig(
        provider="minimax",
        model="MiniMax-M3",
        model_identity="minimax:MiniMax-M3",
    )
    assert config.pipeline_mode is False, "the direct loop stays the default"

    opted_in = AgentTaskRunConfig(
        provider="minimax",
        model="MiniMax-M3",
        model_identity="minimax:MiniMax-M3",
        pipeline_mode=True,
    )
    assert opted_in.pipeline_mode is True


def test_pipeline_mode_is_read_from_the_config_file(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        json.dumps({
            "provider": "minimax",
            "model": "MiniMax-M3",
            "model_identity": "minimax:MiniMax-M3",
            "pipeline_mode": True,
        }),
        encoding="utf-8",
    )

    assert AgentTaskRunConfig.from_path(path).pipeline_mode is True


def test_the_agents_index_is_not_counted_as_the_agents_work() -> None:
    """Ken writes its index beside the workspace it indexes.

    Left in, .ken/ken.db, .ken/vectors/* and the daemon files appear in every
    changed-path diff, inflating the change-size metric and the artifact copy.
    """
    from bench.agent_task_run import _IGNORED_PARTS

    assert ".ken" in _IGNORED_PARTS
    assert {".venv", "venv", ".git", "__pycache__"} <= _IGNORED_PARTS


def test_a_provider_failure_is_not_recorded_as_a_failed_task() -> None:
    """The pipeline returns engine exceptions as the turn's text.

    An exhausted token plan therefore produced rows with no error, no status,
    zero tokens and a failed verification — indistinguishable from a task the
    model could not do. Six such rows were recorded before this existed.
    """
    from bench.agent_task_run import provider_failure_in_reply

    quota = (
        "The orchestrator engine failed: APIConnectionError: litellm."
        "APIConnectionError: MinimaxException - rate_limit_error: Token Plan "
        "usage limit reached"
    )
    assert provider_failure_in_reply(quota, 0), "zero tokens plus that text is a failure"

    # A run that reached the model cannot have spent nothing.
    assert provider_failure_in_reply(quota, 1200) == ""
    # A real answer can mention a timeout without being one.
    assert provider_failure_in_reply(
        "I raised the client timeout to 30 s in src/client.py:12.", 1200,
    ) == ""
    assert provider_failure_in_reply("", 0) == ""


def test_the_runner_counts_every_provider_call_not_just_the_loop() -> None:
    """A turn bills more than the LoopEngine's own counter.

    The chat agent, planner, council, spec elaborator and task-policy
    classifier call the provider directly; the engine's ``total_prompt_tokens``
    never saw them. Counting at the provider boundary is what makes the
    comparison between engine modes honest, because the phases are not
    distributed evenly across modes.
    """
    from bench.agent_task_run import _AuxUsage

    class _Response:
        usage = type("U", (), {"prompt_tokens": 183, "completion_tokens": 8})()

    usage = _AuxUsage()
    usage.add(2_100, "chat")
    usage.add_provider_call(_Response())

    assert usage.provider_calls == 1
    assert usage.provider_prompt == 183
    assert usage.provider_completion == 8
    assert usage.total == 2_100

    usage.reset()
    assert usage.provider_calls == 0
    assert usage.by_lane == {}
    assert usage.total == 0


def test_a_response_without_usage_counts_the_call_but_adds_no_tokens() -> None:
    """A call that happened is a call, even when the provider sends no usage.

    Counting it keeps ``provider_calls`` a truthful count of round trips;
    adding zero tokens keeps the totals from inventing a number.
    """
    from bench.agent_task_run import _AuxUsage

    usage = _AuxUsage()
    usage.add_provider_call(type("R", (), {"usage": None})())

    assert usage.provider_calls == 1
    assert usage.provider_prompt == 0
    assert usage.provider_completion == 0


def test_the_fixture_baseline_cannot_track_a_compiled_cache(tmp_path) -> None:
    """A stray `__pycache__` used to become a tracked baseline file.

    The fixtures carry no `.gitignore`, so a cache left by someone running
    pytest inside the fixture directory was committed by `init_git_workspace`.
    Every later run then recompiled the `.pyc`, the workspace baseline saw it
    change, and the reviewer's diff filled with binary cache noise — 7 370 of
    7 980 characters in one `wide-sum` run.
    """
    import shutil
    import subprocess

    from bench.agent_task_run import init_git_workspace

    if not shutil.which("git"):
        pytest.skip("git is not installed")

    repo = tmp_path / "repo"
    (repo / "src" / "__pycache__").mkdir(parents=True)
    (repo / "src" / "mod.py").write_text("value = 1\n", encoding="utf-8")
    (repo / "src" / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"\x00binary")
    (repo / ".pytest_cache").mkdir()
    (repo / ".pytest_cache" / "CACHEDIR.TAG").write_text("x", encoding="utf-8")

    assert init_git_workspace(repo) is True

    tracked = subprocess.run(
        ["git", "ls-files"], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout
    assert "src/mod.py" in tracked
    assert ".pyc" not in tracked, tracked
    assert "__pycache__" not in tracked, tracked
    assert ".pytest_cache" not in tracked, tracked
