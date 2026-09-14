"""The artifact-derived metrics the engine comparison reports.

``max_workless_rounds`` and ``extra_changed_files`` are the two measurements
that describe the *shape* of a run rather than a counter the engine keeps, so
they are read out of ``run.json``. Both exist because the goals they serve had
no number: a run that loops without working, and a run that changes what it was
not asked to change.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.agent_task_repeated_compare import (
    _metric,
    extra_changed_files,
    max_workless_rounds,
    render_markdown,
)


def _artifact(tmp_path: Path, **fields: object) -> str:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(fields), encoding="utf-8")
    return str(path)


def test_workless_rounds_measures_the_longest_streak(tmp_path: Path) -> None:
    artifact = _artifact(tmp_path, action_records=[
        {"tool_calls_count": 3},
        {"tool_calls_count": 0},
        {"tool_calls_count": 0},
        {"tool_calls_count": 1},
        {"tool_calls_count": 0},
        {"tool_calls_count": 0},
        {"tool_calls_count": 0},
        {"tool_calls_count": 2},
    ])

    assert max_workless_rounds(artifact) == 3


def test_workless_rounds_counts_a_trailing_streak(tmp_path: Path) -> None:
    """The livelock that motivated the metric ran to the iteration limit."""
    artifact = _artifact(tmp_path, action_records=[
        {"tool_calls_count": 1},
        *({"tool_calls_count": 0} for _ in range(11)),
    ])

    assert max_workless_rounds(artifact) == 11


def test_a_missing_artifact_reports_zero_not_a_crash() -> None:
    assert max_workless_rounds("/nonexistent/run.json") == 0
    assert extra_changed_files("/nonexistent/run.json") == 0
    assert max_workless_rounds("") == 0


def test_extra_changed_files_counts_only_undeclared_paths(tmp_path: Path) -> None:
    artifact = _artifact(
        tmp_path,
        task={"expected_changed_paths": ["src/pricing.py"]},
        changed_paths=["src/pricing.py", "README.md", "src/unrelated.py"],
    )

    assert extra_changed_files(artifact) == 2


def test_extra_changed_files_is_silent_when_the_task_declares_nothing(
    tmp_path: Path,
) -> None:
    """A task that lists no expected paths cannot say which new file was invited."""
    artifact = _artifact(
        tmp_path,
        task={"expected_changed_paths": []},
        changed_paths=["src/anything.py"],
    )

    assert extra_changed_files(artifact) == 0


def test_the_report_names_the_metrics_and_the_paired_verdict() -> None:
    report = {
        "pairs": 2,
        "label_a": "a",
        "label_b": "b",
        "unpaired_a": [],
        "unpaired_b": [],
        "success": {"a": 2, "b": 2, "total": 2},
        "metrics": {
            "extra_changed_files": {
                "median_a": 0.0, "median_b": 1.0, "delta": 1.0, "delta_pct": 0.0,
                "range_a": [0.0, 0.0], "range_b": [0.0, 2.0],
                "pairs": 2, "pairs_better_b": 0, "pairs_worse_b": 1,
                "pairs_non_tied": 1, "sign_test_p": 1.0, "resolvable": False,
            },
        },
        "per_task": {},
    }

    markdown = render_markdown(report)

    assert "extra_changed_files" in markdown
    assert "pairs better / worse" in markdown
    assert "sign test" in markdown


def test_answer_wording_is_reported_but_never_gates_success() -> None:
    """A regex over free text must not decide whether the work was done."""
    import dataclasses

    from bench.agent_task_eval import AgentTaskObservation

    fields = {f.name for f in dataclasses.fields(AgentTaskObservation)}
    assert "final_answer_patterns_ok" in fields
    assert "final_pattern_checks" in fields

    payload = {
        "task_id": "t", "condition": "baseline", "repetition": 0,
        "provider": "p", "model": "m", "model_identity": "i",
        "dataset_sha256": "d", "condition_manifest_sha256": "c",
        "condition_sha256": "s", "success": True, "verify_exit_code": 0,
        "engine_status": "done", "changed_paths": ["src/thing.py"],
        "forbidden_changes": [], "missing_expected_changes": [],
        "final_pattern_checks": {"stage": False},
        "action_pattern_checks": {"pytest": True},
        "prompt_tokens": 1, "completion_tokens": 1, "latency_seconds": 1.0,
        "tool_calls": 1,
    }

    rebuilt = AgentTaskObservation.from_dict(payload)

    assert rebuilt.success is True, (
        "the deliverable passed its verifier; a missing word in the summary is a "
        "communication signal, not a failed task"
    )
    assert rebuilt.final_pattern_checks == {"stage": False}


def test_changed_lines_counts_the_recorded_diff(tmp_path: Path) -> None:
    """The protocol says a one-line fix stays a one-line fix."""
    from bench.agent_task_repeated_compare import changed_lines

    artifact = _artifact(tmp_path, changed_files_summary=(
        "### src/app.py (modified)\n"
        "```diff\n"
        "--- a/src/app.py\n"
        "+++ b/src/app.py\n"
        "@@ -1,3 +1,4 @@\n"
        " context\n"
        "-old line\n"
        "+new line\n"
        "+another line\n"
        "```\n"
    ))

    assert changed_lines(artifact) == 3, "two added and one removed"


def test_introduced_placeholders_flags_only_added_lines(tmp_path: Path) -> None:
    """Removing a TODO is not shipping one."""
    from bench.agent_task_repeated_compare import introduced_placeholders

    artifact = _artifact(tmp_path, changed_files_summary=(
        "```diff\n"
        "-    # TODO: handle the empty case\n"
        "+    if not items:\n"
        "+        return []\n"
        "```\n"
    ))
    assert introduced_placeholders(artifact) == 0

    artifact = _artifact(tmp_path, changed_files_summary=(
        "```diff\n"
        "+    raise NotImplementedError\n"
        "```\n"
    ))
    assert introduced_placeholders(artifact) == 1


def test_change_metrics_are_silent_without_an_artifact() -> None:
    from bench.agent_task_repeated_compare import (
        changed_lines,
        introduced_placeholders,
    )

    assert changed_lines("/nonexistent/run.json") == 0
    assert introduced_placeholders("/nonexistent/run.json") == 0


def test_a_run_that_never_reached_the_model_has_no_cost(tmp_path: Path) -> None:
    """A pipeline that halts before the loop scores zero tokens and no error.

    Counting that zero as a measurement credits the arm with the cheapest
    possible result for work it declined to start. It still counts as a failed
    task, because refusing to start is a real outcome.
    """
    from bench.agent_task_repeated_compare import compare, load_arm

    def _arm(label: str, prompt_tokens: int, success: bool) -> Path:
        arm = tmp_path / label
        arm.mkdir()
        path = arm / "observations.jsonl"
        path.write_text(
            json.dumps({
                "task_id": "t", "repetition": 0, "condition": "baseline",
                "prompt_tokens": prompt_tokens, "completion_tokens": prompt_tokens,
                "tool_calls": 1 if prompt_tokens else 0,
                "latency_seconds": 1.0, "success": success,
            }) + "\n",
            encoding="utf-8",
        )
        return path

    report = compare(
        load_arm(_arm("a", 1000, True), "a"),
        load_arm(_arm("b", 0, False), "b"),
    )

    assert report["pairs"] == 1
    assert report["measured_pairs"] == 0
    assert report["excluded_pairs"] == ["t#r0"]
    assert report["success"] == {"a": 1, "b": 0, "total": 1}
    assert report["metrics"]["prompt_tokens"]["pairs"] == 0


# ── the turn costs more than the loop ─────────────────────────────────


def test_pipeline_prompt_tokens_prefers_the_provider_total() -> None:
    """The loop counter is a subset of what the provider billed.

    Comparing engine modes on ``prompt_tokens`` alone credits whichever mode
    runs fewer of the phases that call the provider directly — the chat
    agent, the planner, the council, the spec elaborator and the task-policy
    classifier. ``pipeline_prompt_tokens`` is the number that cannot be gamed
    that way.
    """
    row = {
        "prompt_tokens": 80_000,
        "aux_prompt_tokens": 0,
        "provider_prompt_tokens": 96_500,
    }

    assert _metric(row, "pipeline_prompt_tokens") == 96_500


def test_pipeline_prompt_tokens_falls_back_for_rows_without_a_provider_total() -> None:
    """Observation files written before the callback existed still compare."""
    row = {"prompt_tokens": 80_000, "aux_prompt_tokens": 4_200}

    assert _metric(row, "pipeline_prompt_tokens") == 84_200


def test_pipeline_completion_tokens_uses_the_provider_total_too() -> None:
    """The loop's output counter is a subset of the provider's, same as input."""
    row = {
        "completion_tokens": 2_800,
        "provider_completion_tokens": 4_150,
    }

    assert _metric(row, "pipeline_completion_tokens") == 4_150


def test_pipeline_completion_tokens_falls_back_to_the_loop_counter() -> None:
    assert _metric({"completion_tokens": 2_800}, "pipeline_completion_tokens") == 2_800
    assert _metric({}, "provider_calls") == 0


def test_the_diff_of_one_file_counts_once_even_when_paths_are_two_spellings() -> None:
    """Runs recorded before the tracker keyed by realpath carry the file twice.

    That happened in 22 of the 32 stored runs, so a metric counting the raw
    summary doubled in some runs and not others — a coin flip unrelated to
    what the model did.
    """
    from bench.agent_task_repeated_compare import deduplicated_diff

    summary = (
        "### /var/folders/x/repo/src/a.py (modified)\n"
        "```diff\n--- a/a.py\n+++ b/a.py\n@@\n-    old = 1\n+    new = 2\n```\n\n"
        "### /private/var/folders/x/repo/src/a.py (modified)\n"
        "```diff\n--- a/a.py\n+++ b/a.py\n@@\n-    old = 1\n+    new = 2\n```"
    )

    deduped = deduplicated_diff(summary)
    assert deduped.count("-    old = 1") == 1
    assert deduped.count("### ") == 1


def test_two_genuinely_different_files_are_both_counted() -> None:
    from bench.agent_task_repeated_compare import deduplicated_diff

    summary = (
        "### /repo/src/a.py (modified)\n```diff\n+a\n```\n\n"
        "### /repo/src/b.py (created)\n```diff\n+b\n```"
    )

    assert deduplicated_diff(summary).count("### ") == 2
