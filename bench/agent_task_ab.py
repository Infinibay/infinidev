"""Run two engine configurations over the same corpus and compare them.

An engine change is not a prompt-guidance candidate, so it does not belong in
the ``baseline``/``candidate`` condition pair that ``agent_task_eval.py`` scores.
It is two runs of the *same* condition with one setting or code path different,
and the only honest way to read the result is paired by ``(task, repetition)``
with the within-arm spread in view.

This script does the whole loop: run arm A, run arm B, compare, write the
markdown and the JSON. Both arms must declare the same ``repetitions``; the
comparison refuses to run when they do not.

Example::

    python -m bench.agent_task_ab \\
        bench/engine_eval_v2.tasks.jsonl \\
        bench/engine_eval_v2.minimax.conditions.json \\
        bench/agent_task_run.minimax.legacy.json \\
        bench/agent_task_run.minimax.closure.json \\
        --label-a closure-off --label-b closure-on \\
        --output-root bench/runs/20260913-engine-v2/ab-closure

Each arm writes ``<output-root>/<label>/{observations.jsonl,artifacts/}``; the
comparison writes ``comparison.md`` and ``comparison.json`` beside them.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from bench.agent_task_repeated_compare import compare, load_arm, render_markdown
from bench.agent_task_run import AgentTaskRunConfig, run_campaign


def _clear_stale_artifacts(
    artifacts: Path,
    done: frozenset[tuple[str, int, str]],
    conditions: tuple[str, ...],
) -> None:
    """Drop the artifact directories of units this pass is about to re-run.

    A unit is re-run when nothing valid is recorded for it, and the previous
    attempt left its directory behind — the row the runner writes before it
    stops on a provider error, for instance. The runner refuses to reuse a
    directory, so without this a resume cannot retry anything.
    """
    if not artifacts.is_dir():
        return
    for entry in artifacts.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        for condition in conditions:
            suffix = f".{condition}"
            if not name.endswith(suffix):
                continue
            stem = name[: -len(suffix)]
            task_id, _, repetition = stem.rpartition(".r")
            if not repetition.isdigit():
                continue
            if (task_id, int(repetition), condition) in done:
                continue
            shutil.rmtree(entry, ignore_errors=True)


def _completed_units(path: Path) -> frozenset[tuple[str, int, str]]:
    """The ``(task, repetition, condition)`` units already recorded in *path*."""
    if not path.is_file() or not path.stat().st_size:
        return frozenset()
    units: set[tuple[str, int, str]] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if str(row.get("error", "")).strip():
            # The runner records the runtime/provider error and then stops. The
            # row is a diagnostic, not a measurement: counting it as done both
            # fails a task the model never finished and hides it from the rerun.
            continue
        units.add(
            (str(row["task_id"]), int(row.get("repetition", 0)), str(row["condition"]))
        )
    return frozenset(units)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("conditions", type=Path)
    parser.add_argument("config_a", type=Path)
    parser.add_argument("config_b", type=Path)
    parser.add_argument("--label-a", default="a")
    parser.add_argument("--label-b", default="b")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--fixture-root", type=Path, default=Path("bench/agent_task_fixtures")
    )
    parser.add_argument(
        "--split", choices=("calibration", "validation"), default="validation"
    )
    parser.add_argument("--include-drafts", action="store_true")
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument(
        "--condition", action="append", choices=("baseline", "candidate"),
        default=[],
    )
    args = parser.parse_args()

    config_a = AgentTaskRunConfig.from_path(args.config_a)
    config_b = AgentTaskRunConfig.from_path(args.config_b)
    if config_a.repetitions != config_b.repetitions:
        raise SystemExit(
            "the two arms must declare the same repetitions, otherwise the "
            "paired comparison compares different sample sizes"
        )
    if config_a.model_identity != config_b.model_identity:
        raise SystemExit("the two arms must run the same model identity")

    conditions = tuple(args.condition) or ("baseline",)
    task_ids = tuple(args.task_id)
    observations: dict[str, Path] = {}
    for label, config_path in ((args.label_a, args.config_a), (args.label_b, args.config_b)):
        arm_root = args.output_root / label
        arm_root.mkdir(parents=True, exist_ok=True)
        path = arm_root / "observations.jsonl"
        # Reuse whatever this arm already paid for. A transient provider timeout
        # stops the runner by design, and a campaign is long enough that losing
        # the completed half to one timeout makes the tool unusable.
        done = _completed_units(path)
        if done:
            print(
                f"=== arm {label}: {config_path} "
                f"(reusing {len(done)} completed executions) ===",
                flush=True,
            )
        else:
            print(f"=== arm {label}: {config_path} ===", flush=True)
        _clear_stale_artifacts(arm_root / "artifacts", done, conditions)
        run_campaign(
            args.tasks,
            args.conditions,
            config_path,
            path,
            arm_root / "artifacts",
            fixture_root=args.fixture_root,
            split=args.split,
            include_drafts=args.include_drafts,
            task_ids=task_ids,
            conditions=conditions,
            skip_units=done,
        )
        observations[label] = path

    report = compare(
        load_arm(observations[args.label_a], args.label_a),
        load_arm(observations[args.label_b], args.label_b),
    )
    markdown = render_markdown(report)
    (args.output_root / "comparison.md").write_text(markdown, encoding="utf-8")
    (args.output_root / "comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(markdown)
    print(f"wrote {args.output_root / 'comparison.md'}")


if __name__ == "__main__":
    main()
