"""Compare two observation files that differ by one engine setting.

``agent_task_eval.py`` builds a paired report across ``baseline`` and
``candidate`` *conditions* inside one run, which is the right tool for
prompt-guidance experiments. It is the wrong tool for engine changes, where
both arms are code or settings rather than condition text, and it is the wrong
tool for deciding whether an effect is real: with ``repetitions: 1`` on six
small tasks, the spread inside one arm is larger than most effects the
experiment is meant to detect.

This script pairs observations by ``(task_id, repetition)`` across two files
and prints, for each metric, the median per arm, the paired delta, and the
within-arm spread. The spread is printed because a delta smaller than it is
not evidence, and a report that hides that invites a decision the data cannot
support.

Usage::

    python -m bench.agent_task_repeated_compare \
        bench/runs/RUN/rep-legacy/observations.jsonl \
        bench/runs/RUN/rep-closure/observations.jsonl \
        --label-a legacy --label-b closure
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

#: Metrics reported as lower-is-better counts. ``success`` is a rate and is
#: handled separately. ``max_workless_rounds`` is read out of the run artifact
#: rather than the observation row, because it describes the shape of the run
#: and not a counter the engine reports.
_METRICS = (
    "prompt_tokens",
    "aux_prompt_tokens",
    "pipeline_prompt_tokens",
    "completion_tokens",
    "pipeline_completion_tokens",
    "provider_calls",
    "tool_calls",
    "malformed_tool_calls",
    "max_workless_rounds",
    "extra_changed_files",
    "changed_lines",
    "introduced_placeholders",
    "final_answer_wording",
    "answers_without_a_command",
    "latency_seconds",
)

#: Fewest non-tied pairs a sign test needs before its verdict is reported.
_MIN_SIGN_TEST_PAIRS = 6

#: A round is workless when the model produced no tool call at all. One is a
#: model thinking; a streak of them is a loop, and it is the shape a rejected
#: closure took when the engine refused it without saying why.
_WORKLESS_STREAK_ALARM = 3


def _load_artifact(run_artifact: str) -> dict:
    """Read one ``run.json``, or return nothing when it is missing."""
    if not run_artifact:
        return {}
    path = Path(run_artifact)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def extra_changed_files(run_artifact: str) -> int:
    """Files the run changed that the task did not declare.

    Scope discipline is a product bar — do not change what you were not asked to
    change — and the deterministic half of it is countable: everything in
    ``changed_paths`` that the task did not list as an expected change. Only
    tasks that declare expected paths can be scored, because a task that
    declares none has no way to say which new file was invited.
    """
    artifact = _load_artifact(run_artifact)
    task = artifact.get("task") or {}
    expected = {str(item) for item in task.get("expected_changed_paths") or ()}
    if not expected:
        return 0
    changed = {str(item) for item in artifact.get("changed_paths") or ()}
    return len(changed - expected)


#: What a placeholder looks like in the lines a run added. The loop protocol
#: forbids shipping one ("no `TODO`, no stub function, no placeholder for
#: later"), and it is the cheapest craft signal a diff carries.
_PLACEHOLDER = re.compile(
    r"\b(?:TODO|FIXME|XXX|HACK)\b|NotImplementedError|raise NotImplemented",
    re.IGNORECASE,
)

#: One ``### <path> (action)`` header per changed file in the recorded diff.
_DIFF_HEADER = re.compile(r"^### (?P<path>.+?) \((?P<action>[^)]*)\)$")


def deduplicated_diff(summary: str) -> str:
    """The recorded diff with one section per *file*, not per path spelling.

    Runs recorded before ``FileChangeTracker`` keyed by ``realpath`` carry the
    same file twice when a tool call named ``/var/...`` and the workspace scan
    named ``/private/var/...``. That happened in 22 of 32 stored runs and
    doubled ``changed_lines`` in exactly those runs — a per-run coin flip
    unrelated to what the model did, which is worse than a uniform bias.
    """
    if not summary:
        return ""
    parts = summary.split("\n\n")
    kept: list[str] = []
    seen: set[str] = set()
    for part in parts:
        header = part.splitlines()[0] if part.splitlines() else ""
        match = _DIFF_HEADER.match(header)
        if match is None:
            kept.append(part)
            continue
        identity = os.path.realpath(match.group("path"))
        if identity in seen:
            continue
        seen.add(identity)
        kept.append(part)
    return "\n\n".join(kept)


def changed_lines(run_artifact: str) -> int:
    """Source lines the run added or removed, from the recorded diff.

    The protocol asks for the smallest reversible change and says a one-line
    fix stays a one-line fix. This is the count that would move if a run
    rewrote a file instead.
    """
    artifact = _load_artifact(run_artifact)
    summary = deduplicated_diff(str(artifact.get("changed_files_summary") or ""))
    total = 0
    for line in summary.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            continue
        if line.startswith("+") or line.startswith("-"):
            total += 1
    return total


def introduced_placeholders(run_artifact: str) -> int:
    """Placeholders the run added, which the protocol forbids shipping."""
    artifact = _load_artifact(run_artifact)
    summary = deduplicated_diff(str(artifact.get("changed_files_summary") or ""))
    return sum(
        1
        for line in summary.splitlines()
        if line.startswith("+")
        and not line.startswith("+++")
        and _PLACEHOLDER.search(line)
    )


def max_workless_rounds(run_artifact: str) -> int:
    """Longest streak of model rounds in one run that called no tool.

    Read from ``run.json``: ``action_records`` is one entry per outer-loop
    round. A run that keeps re-attempting a rejected closure shows up here as a
    long streak, which no token total can distinguish from an expensive but
    healthy run.
    """
    artifact = _load_artifact(run_artifact)
    longest = current = 0
    for record in artifact.get("action_records", ()) or ():
        if isinstance(record, str):
            continue
        try:
            calls = int(record.get("tool_calls_count", 0) or 0)
        except (TypeError, ValueError):
            calls = 0
        if calls:
            current = 0
        else:
            current += 1
            longest = max(longest, current)
    return longest


@dataclass(frozen=True)
class Arm:
    """One configuration's observations, keyed by (task_id, repetition)."""

    label: str
    rows: dict[tuple[str, int], dict]


def load_arm(path: Path, label: str) -> Arm:
    rows: dict[tuple[str, int], dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (str(row["task_id"]), int(row.get("repetition", 0)))
        if key in rows:
            raise ValueError(f"duplicate observation for {key} in {path}")
        rows[key] = row
    if not rows:
        raise ValueError(f"no observations in {path}")
    return Arm(label=label, rows=rows)


def _was_measured(row: dict) -> bool:
    """Whether the run reached the model at all.

    ``prompt_tokens`` is zero only when no request was made, which happens when
    a phase before the loop ends the turn. Such a row has no cost to compare.
    """
    return int(row.get("prompt_tokens") or 0) > 0


def _metric(row: dict, name: str) -> float:
    """Read one metric, deriving the artifact-only ones on demand."""
    if name == "max_workless_rounds":
        return float(max_workless_rounds(str(row.get("run_artifact", ""))))
    if name == "extra_changed_files":
        return float(extra_changed_files(str(row.get("run_artifact", ""))))
    if name == "changed_lines":
        return float(changed_lines(str(row.get("run_artifact", ""))))
    if name == "introduced_placeholders":
        return float(introduced_placeholders(str(row.get("run_artifact", ""))))
    if name == "final_answer_wording":
        # 0 is good: it counts answers that missed the wording the task asked for.
        return 0.0 if row.get("final_answer_patterns_ok", True) else 1.0
    if name == "answers_without_a_command":
        # 0 is good: it counts answers the user cannot check because they name
        # no command whose result could be verified.
        from bench.agent_task_answer_quality import answer_quality

        answer = str(
            _load_artifact(str(row.get("run_artifact", ""))).get("final_answer", "")
            or ""
        )
        if not answer:
            return 0.0
        return 0.0 if answer_quality(answer)["names_a_command"] else 1.0
    if name == "pipeline_prompt_tokens":
        # What the turn really cost. ``prompt_tokens`` is the LoopEngine's own
        # counter and excludes the chat agent, the planner, the council, the
        # spec elaborator and the task-policy classifier, which call the
        # provider directly. Comparing engine modes on the loop alone credits
        # whichever mode runs fewer of those phases.
        #
        # Prefer the provider-boundary total when the row carries one; fall
        # back to the loop counter plus the voluntarily reported lanes, so
        # observation files written before the callback existed still compare.
        provider = int(row.get("provider_prompt_tokens", 0) or 0)
        if provider > 0:
            return float(provider)
        return float(row.get("prompt_tokens", 0) or 0) + float(
            row.get("aux_prompt_tokens", 0) or 0
        )
    if name == "pipeline_completion_tokens":
        # Same asymmetry as the prompt side: the phases outside the loop
        # generate output too, and only the loop's output is in
        # ``completion_tokens``.
        provider = int(row.get("provider_completion_tokens", 0) or 0)
        if provider > 0:
            return float(provider)
        return float(row.get("completion_tokens", 0) or 0)
    return float(row.get(name, 0) or 0)


def _sign_test_p_value(ties: int, directional: int) -> float:
    """Two-sided sign-test p-value over the pairs that moved at all.

    With no ties this is the exact binomial test against a fair coin, which is
    the strongest statement a paired design with this sample size can make.
    Ties carry no direction and are dropped, as the test requires.
    """
    trials = ties + directional
    if trials == 0:
        return 1.0
    observed = max(ties, directional)
    tail = sum(_comb(trials, k) for k in range(observed, trials + 1))
    return min(1.0, 2 * tail / (2 ** trials))


def _comb(n: int, k: int) -> int:
    from math import comb

    return comb(n, k)


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _spread(values: list[float]) -> tuple[float, float]:
    return (min(values), max(values)) if values else (0.0, 0.0)


def compare(arm_a: Arm, arm_b: Arm) -> dict:
    """Return the paired comparison, or raise when the arms do not line up."""
    shared = sorted(set(arm_a.rows) & set(arm_b.rows))
    only_a = sorted(set(arm_a.rows) - set(arm_b.rows))
    only_b = sorted(set(arm_b.rows) - set(arm_a.rows))
    if not shared:
        raise ValueError("the two arms share no (task, repetition) pairs")

    per_task: dict[str, dict] = defaultdict(lambda: {"a": [], "b": []})
    metrics: dict[str, dict[str, list[float]]] = {
        name: {"a": [], "b": []} for name in _METRICS
    }
    successes = {"a": 0, "b": 0}
    excluded: list[str] = []
    for key in shared:
        row_a, row_b = arm_a.rows[key], arm_b.rows[key]
        task_id = key[0]
        per_task[task_id]["a"].append(row_a)
        per_task[task_id]["b"].append(row_b)
        successes["a"] += int(bool(row_a.get("success")))
        successes["b"] += int(bool(row_b.get("success")))
        # A run that never reached the model has no cost. Reporting its zero as
        # a measurement would credit the arm with the cheapest possible result
        # for work it declined to start: a pipeline that halts on a product
        # question before entering the loop scores zero tokens and an error-free
        # row. It still counts in the success tally, because refusing to start
        # is a real outcome.
        if not _was_measured(row_a) or not _was_measured(row_b):
            excluded.append(f"{task_id}#r{key[1]}")
            continue
        for name in _METRICS:
            metrics[name]["a"].append(float(_metric(row_a, name)))
            metrics[name]["b"].append(float(_metric(row_b, name)))

    report: dict = {
        "pairs": len(shared),
        "measured_pairs": len(shared) - len(excluded),
        "excluded_pairs": excluded,
        "unpaired_a": [f"{t}#r{r}" for t, r in only_a],
        "unpaired_b": [f"{t}#r{r}" for t, r in only_b],
        "label_a": arm_a.label,
        "label_b": arm_b.label,
        "metrics": {},
        "per_task": {},
    }
    for name, values in metrics.items():
        med_a, med_b = _median(values["a"]), _median(values["b"])
        delta = med_b - med_a
        spread_a = _spread(values["a"])
        spread_b = _spread(values["b"])
        # Compare the PAIRED deltas, not the marginal ranges. The arms share a
        # task and a model response distribution, so pairing removes most of
        # the variance; a rule that asked the median delta to exceed the whole
        # within-arm range threw away a change that moved eight of nine pairs
        # in the same direction. The sign test is what the design supports.
        pairs = [b - a for a, b in zip(values["a"], values["b"])]
        improved = sum(1 for value in pairs if value < 0)
        worsened = sum(1 for value in pairs if value > 0)
        non_tied = improved + worsened
        p_value = _sign_test_p_value(len(pairs) - non_tied, improved)
        # A sign test on two pairs is not evidence, however small its p-value.
        # ``max_workless_rounds`` is zero for almost every run, so its pairs are
        # mostly ties and one disagreement can look decisive.
        enough = non_tied >= _MIN_SIGN_TEST_PAIRS
        report["metrics"][name] = {
            "median_a": round(med_a, 2),
            "median_b": round(med_b, 2),
            "delta": round(delta, 2),
            "delta_pct": round(delta / med_a * 100, 1) if med_a else 0.0,
            "range_a": [round(v, 2) for v in spread_a],
            "range_b": [round(v, 2) for v in spread_b],
            "pairs": len(pairs),
            "pairs_better_b": improved,
            "pairs_worse_b": worsened,
            "sign_test_p": round(p_value, 4),
            "pairs_non_tied": non_tied,
            "resolvable": p_value < 0.05 and enough,
        }

    for task_id, rows in sorted(per_task.items()):
        entry: dict = {
            "repetitions": len(rows["a"]),
            "success_a": sum(1 for r in rows["a"] if r.get("success")),
            "success_b": sum(1 for r in rows["b"] if r.get("success")),
        }
        for name in _METRICS:
            a = [float(_metric(r, name)) for r in rows["a"]]
            b = [float(_metric(r, name)) for r in rows["b"]]
            entry[name] = {
                "median_a": round(_median(a), 2),
                "median_b": round(_median(b), 2),
                "delta": round(_median(b) - _median(a), 2),
            }
        report["per_task"][task_id] = entry

    report["success"] = {
        "a": successes["a"],
        "b": successes["b"],
        "total": len(shared),
    }
    return report


def render_markdown(report: dict) -> str:
    a, b = report["label_a"], report["label_b"]
    lines = [
        f"# {a} vs {b}",
        "",
        f"Paired executions: {report['pairs']} "
        f"(success {report['success']['a']} vs {report['success']['b']})",
        f"Pairs with a cost measurement: {report.get('measured_pairs', report['pairs'])}",
    ]
    if report.get("excluded_pairs"):
        lines.append(
            "Excluded from the cost metrics, never reached the model: "
            + ", ".join(report["excluded_pairs"])
        )
    if report["unpaired_a"] or report["unpaired_b"]:
        lines.append("")
        lines.append(
            f"Unpaired: {len(report['unpaired_a'])} in {a}, "
            f"{len(report['unpaired_b'])} in {b}"
        )
    lines += [
        "",
        "| metric | median A | median B | delta | delta % | pairs better / worse | sign test p | resolved |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for name, m in report["metrics"].items():
        lines.append(
            f"| {name} | {m['median_a']} | {m['median_b']} | {m['delta']} "
            f"| {m['delta_pct']}% | {m['pairs_better_b']} / {m['pairs_worse_b']} "
            f"| {m['sign_test_p']} "
            f"| {'yes' if m['resolvable'] else 'no'} |"
        )
    lines += [
        "",
        "`resolved` is a two-sided sign test over the pairs that moved: the arm "
        "comparison is paired by (task, repetition), so the paired deltas carry "
        "the signal and the marginal ranges do not. `pairs better / worse` "
        "counts how many pairs moved in each direction (lower is better).",
        "",
        "| task | reps | success | prompt tokens (A/B) | tool calls (A/B) | latency s (A/B) |",
        "| --- | ---: | --- | --- | --- | --- |",
    ]
    for task_id, entry in report["per_task"].items():
        lines.append(
            f"| {task_id} | {entry['repetitions']} "
            f"| {entry['success_a']}/{entry['repetitions']} vs "
            f"{entry['success_b']}/{entry['repetitions']} "
            f"| {entry['prompt_tokens']['median_a']} / "
            f"{entry['prompt_tokens']['median_b']} "
            f"| {entry['tool_calls']['median_a']} / {entry['tool_calls']['median_b']} "
            f"| {entry['latency_seconds']['median_a']} / "
            f"{entry['latency_seconds']['median_b']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("arm_a", type=Path)
    parser.add_argument("arm_b", type=Path)
    parser.add_argument("--label-a", default="A")
    parser.add_argument("--label-b", default="B")
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--json", dest="json_out", type=Path, default=None)
    args = parser.parse_args()

    report = compare(
        load_arm(args.arm_a, args.label_a),
        load_arm(args.arm_b, args.label_b),
    )
    markdown = render_markdown(report)
    print(markdown)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
