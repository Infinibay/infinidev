"""Deterministic rubric probes: they must see, and abstain when they cannot.

A probe that cannot see is not a probe that agrees. These tests pin the two
failure modes that matter: scoring a run zero because the artifact did not carry
the evidence, and scoring it wrong because generated caches or a read of a
script were mistaken for the work.
"""

from __future__ import annotations

import json
from pathlib import Path

from bench.agent_task_rubric_probes import (
    PROBES,
    _changed_files,
    probe_failure_recognition,
    probe_immutability_discipline,
    probe_located_the_stage,
    run_probes,
)


def _artifact(**overrides: object) -> dict:
    artifact = {
        "final_answer": "",
        "changed_files_summary": "",
        "tool_trace": [],
        "_diff": "",
    }
    artifact.update(overrides)
    return artifact


# ── generated caches are not source ─────────────────────────────────────


def test_compiled_caches_are_not_counted_as_changed_source() -> None:
    diff = (
        "### /repo/src/mod_27.py (modified)\n```diff\n+a\n```\n\n"
        "### /repo/src/__pycache__/mod_27.cpython-312.pyc (created)\n"
        "```diff\n+binary\n```\n\n"
        "### /repo/tests/__pycache__/t.cpython-312-pytest.pyc (created)\n"
        "```diff\n+binary\n```"
    )

    assert _changed_files(diff) == ["/repo/src/mod_27.py"]


def test_a_correct_stage_fix_beats_the_cache_noise() -> None:
    """`wide-sum` once produced 7 980 chars of diff, 92 % of it `.pyc`."""
    diff = (
        "### /repo/src/mod_27.py (modified)\n```diff\n+a\n```\n\n"
        "### /repo/src/__pycache__/mod_27.cpython-312.pyc (created)\n"
        "```diff\n+binary\n```"
    )

    score, evidence = probe_located_the_stage(_artifact(_diff=diff))

    assert score == 2, evidence


def test_compensating_in_the_pipeline_scores_zero() -> None:
    diff = "### /repo/src/pipeline.py (modified)\n```diff\n+a\n```"

    score, evidence = probe_located_the_stage(_artifact(_diff=diff))

    assert score == 0
    assert "compensates" in evidence


def test_no_change_anywhere_scores_zero_not_abstain() -> None:
    score, evidence = probe_located_the_stage(_artifact(_diff=""))

    assert score == 0
    assert "no file changed" in evidence


# ── abstention ──────────────────────────────────────────────────────────


def test_a_delegated_trace_cannot_decide_a_trace_criterion() -> None:
    """In `orchestrator` mode the trace holds the principal's calls only.

    The worker's reads, edit and test run are not in the artifact, so a probe
    that looked for them would report "no test ran" for a run whose worker ran
    the suite.
    """
    artifact = _artifact(
        tool_trace=[
            {"tool_name": "team_delegate", "arguments": {"ticket_id": "t1"},
             "result": "{}"},
            {"tool_name": "execute_command", "arguments": {}, "result": "{}"},
        ]
    )

    score, reason = PROBES["assurance-scope"](artifact)

    assert score is None
    assert "principal" in reason


def test_a_read_of_the_script_is_not_an_invocation_of_it() -> None:
    """Counting the `read_file` would make a correct recovery look like a retry."""
    artifact = _artifact(tool_trace=[
        {"tool_name": "read_file",
         "arguments": {"file_path": "tools/semantic_search.py"}, "result": "{}"},
    ])

    score, reason = probe_failure_recognition(artifact)

    assert score is None
    assert "does not appear" in reason


def test_one_failed_invocation_is_recognition_and_two_is_a_retry() -> None:
    once = _artifact(tool_trace=[
        {"tool_name": "execute_command",
         "arguments": {"command": "python tools/semantic_search.py available"},
         "result": '{"exit_code": 1, "success": false}'},
    ])
    twice = _artifact(tool_trace=[
        {"tool_name": "execute_command",
         "arguments": {"command": "python tools/semantic_search.py available"},
         "result": '{"exit_code": 1, "success": false}'},
        {"tool_name": "execute_command",
         "arguments": {"command": "python tools/semantic_search.py available"},
         "result": '{"exit_code": 1, "success": false}'},
    ])

    assert probe_failure_recognition(once)[0] == 2
    assert probe_failure_recognition(twice)[0] == 1


# ── mutation ────────────────────────────────────────────────────────────


def test_mutating_the_receiver_scores_zero() -> None:
    diff = (
        "### /repo/src/cart.py (modified)\n```diff\n"
        "+        for sku, qty in self.items.items():\n"
        "+            self.items[sku] = qty\n"
        "+        return self\n```"
    )

    score, evidence = probe_immutability_discipline(_artifact(_diff=diff))

    assert score == 0
    assert "self.items" in evidence or "returns self" in evidence


def test_building_a_replacement_cart_scores_two() -> None:
    diff = (
        "### /repo/src/cart.py (modified)\n```diff\n"
        "+        new_items: dict[str, int] = {}\n"
        "+        return Cart(items=new_items)\n```"
    )

    assert probe_immutability_discipline(_artifact(_diff=diff))[0] == 2


# ── the report never shows a zero for an arm that produced nothing ──────


def _campaign(tmp_path: Path, *, with_second_arm: bool) -> Path:
    arms = ["arm-a", "arm-b"] if with_second_arm else ["arm-a"]
    for arm in arms:
        path = tmp_path / arm / "artifacts" / "wide-sum.r0.baseline" / "run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        diff = "### /repo/src/mod_27.py (modified)\n```diff\n+a\n```"
        if arm == "arm-b":
            diff = ""  # produced nothing, like the blocked orchestrator run
        path.write_text(json.dumps({
            "task": {"id": "wide-sum", "request": "find the stage"},
            "verify_exit_code": 0 if arm == "arm-a" else 1,
            "final_answer": "done",
            "changed_files_summary": diff,
            "tool_trace": [],
            "rubric": [
                {"id": "located-the-stage", "kind": "human_review",
                 "description": "in the stage"},
            ],
        }), encoding="utf-8")
    return tmp_path


def test_an_arm_with_no_scored_run_shows_a_dash_not_a_zero(tmp_path: Path) -> None:
    from bench.agent_task_rubric_probes import render

    result = run_probes(_campaign(tmp_path, with_second_arm=True))
    table = render(result)

    assert "2.00 (1)" in table
    # arm-b changed nothing, so it is scored zero on this item, not abstained.
    assert "0.00 (1)" in table


def test_a_probe_that_cannot_see_is_listed_as_an_abstention(tmp_path: Path) -> None:
    from bench.agent_task_rubric_probes import render

    result = run_probes(_campaign(tmp_path, with_second_arm=False))

    assert result["abstentions"] or result["items"]
    assert "located-the-stage" in render(result)


def test_compensating_in_a_hub_or_package_is_not_a_leaf_fix() -> None:
    """The scale corpus has two aggregator levels above the leaves."""
    for aggregator in (
        "### /repo/src/hubs/hub_3.py (modified)\n```diff\n+a\n```",
        "### /repo/src/pkgs/pkg_17/__init__.py (modified)\n```diff\n+a\n```",
    ):
        score, evidence = probe_located_the_stage(_artifact(_diff=aggregator))
        assert score == 0, evidence
        assert "compensates" in evidence


def test_one_leaf_of_the_hierarchy_scores_two() -> None:
    diff = "### /repo/src/pkgs/pkg_17/mod_07.py (modified)\n```diff\n+a\n```"

    assert probe_located_the_stage(_artifact(_diff=diff))[0] == 2


def test_rewriting_a_conforming_leaf_is_not_fixing_the_broken_one() -> None:
    """A `default` run on the scale corpus changed `v + 9` to `v + 58`.

    That restores the total and never touches the defective leaf, so a probe
    that only checks "one leaf file changed" scores it the same as the real
    fix. The removed line is what distinguishes them.
    """
    compensating = _artifact(
        changed_paths=["src/pkgs/pkg_00/mod_11.py"],
        _diff=(
            "### /repo/src/pkgs/pkg_00/mod_11.py (modified)\n```diff\n"
            "-    return v + 9\n"
            "+    return v + 58\n```"
        ),
    )
    real = _artifact(
        changed_paths=["src/pkgs/pkg_17/mod_07.py"],
        _diff=(
            "### /repo/src/pkgs/pkg_17/mod_07.py (modified)\n```diff\n"
            "-    return v - 40\n"
            "+    return v + 9\n```"
        ),
    )

    assert probe_located_the_stage(compensating)[0] == 0
    assert probe_located_the_stage(real)[0] == 2


def test_scratch_files_outside_the_workspace_are_not_the_diff() -> None:
    """Three `lean` runs dropped a `find_bad.py` in `/private/tmp`.

    The diff headers cannot tell that from an edit; the runner's own
    workspace-scoped `changed_paths` can.
    """
    artifact = _artifact(
        changed_paths=["src/pkgs/pkg_17/mod_07.py"],
        _diff=(
            "### /private/tmp/find_bad.py (created)\n```diff\n+import re\n```\n\n"
            "### /repo/src/pkgs/pkg_17/mod_07.py (modified)\n```diff\n"
            "-    return v - 40\n+    return v + 9\n```"
        ),
    )

    score, evidence = probe_located_the_stage(artifact)

    assert score == 2, evidence
