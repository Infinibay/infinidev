"""What the task-policy router costs per turn, and what its classifier changes.

Two campaigns left this open. The policy-catalog A/B found that rendering only
the selected guidance saves 11,5 % of the prompt but costs 13,6 % of the
latency, in 7 of 8 pairs — a direction the direct-loop measurement had reported
the other way round. The classifier A/B compared `preferred` against `off` and
resolved nothing.

Both questions are cheaper than a campaign because the router runs *before* the
loop, once per turn, on the raw request. It can be timed in isolation, and what
the classifier does to the profile can be diffed against what local routing
produces — which matters more than the latency, because `preferred` replaces the
literal operation set rather than adding to it.

Usage::

    python -m bench.task_policy_router_cost bench/engine_eval_v8.tasks.jsonl \\
        --repeats 3 --markdown bench/runs/<date>-<name>/router-cost.md
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

MODES = ("off", "fallback", "preferred")


class _Usage:
    """Provider-side tally for the classifier's own calls."""

    def __init__(self) -> None:
        self.prompt = 0
        self.completion = 0
        self.calls = 0


_USAGE = _Usage()


def _register_callback() -> None:
    import litellm
    from litellm.integrations.custom_logger import CustomLogger

    class _Watcher(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            usage = getattr(response_obj, "usage", None)
            _USAGE.calls += 1
            _USAGE.prompt += int(getattr(usage, "prompt_tokens", 0) or 0)
            _USAGE.completion += int(getattr(usage, "completion_tokens", 0) or 0)

    litellm.callbacks.append(_Watcher())


def _profile(request: str, mode: str):
    from infinidev.engine.task_policies.router import resolve_task_profile

    return resolve_task_profile(
        request,
        enable_embeddings=True,
        enable_llm_fallback=False,
        embedding_threshold=0.18,
        embedding_margin=0.04,
        max_policies=3,
        llm_classifier_mode=mode,
    )


def _shape(profile) -> dict:
    return {
        "operations": sorted(profile.operations),
        "authority": sorted(getattr(profile, "authority", ()) or ()),
        "policies": sorted(item.id for item in profile.selected_policies),
    }


def measure(requests: list[tuple[str, str]], repeats: int) -> dict:
    _register_callback()
    rows = []
    for task_id, request in requests:
        shapes = {mode: _shape(_profile(request, mode)) for mode in MODES}
        timings: dict[str, list[float]] = {mode: [] for mode in MODES}
        for _ in range(repeats):
            for mode in MODES:
                started = time.perf_counter()
                _profile(request, mode)
                timings[mode].append(time.perf_counter() - started)
        rows.append({
            "task_id": task_id,
            "shapes": shapes,
            "seconds": {
                mode: statistics.median(values) for mode, values in timings.items()
            },
            "operations_changed_by_preferred": (
                shapes["preferred"]["operations"] != shapes["off"]["operations"]
            ),
            "operations_dropped_by_preferred": sorted(
                set(shapes["off"]["operations"]) - set(shapes["preferred"]["operations"])
            ),
        })
    return {
        "repeats": repeats,
        "rows": rows,
        "median_seconds": {
            mode: statistics.median([row["seconds"][mode] for row in rows])
            for mode in MODES
        },
        "classifier_tokens": {
            "calls": _USAGE.calls,
            "prompt": _USAGE.prompt,
            "completion": _USAGE.completion,
            "per_request": _USAGE.prompt / max(_USAGE.calls, 1),
        },
    }


def render(result: dict) -> str:
    lines = [
        "# Task-policy router cost",
        "",
        f"{len(result['rows'])} requests × {result['repeats']} repeats per mode.",
        "Timed in isolation: the router runs once per turn, before the loop.",
        "",
        "| mode | median seconds per turn |",
        "| --- | ---: |",
    ]
    for mode in MODES:
        lines.append(f"| `{mode}` | {result['median_seconds'][mode]:.3f} |")
    tokens = result["classifier_tokens"]
    lines += [
        "",
        f"Classifier calls: {tokens['calls']}, "
        f"{tokens['prompt']} prompt tokens ({tokens['per_request']:.0f} per request), "
        f"{tokens['completion']} completion tokens. **These never reach the "
        "LoopEngine's counters.**",
        "",
        "| task | off | fallback | preferred | ops changed | ops dropped by `preferred` |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in result["rows"]:
        lines.append(
            f"| `{row['task_id']}` | {row['seconds']['off']:.3f} | "
            f"{row['seconds']['fallback']:.3f} | {row['seconds']['preferred']:.3f} | "
            f"{'yes' if row['operations_changed_by_preferred'] else 'no'} | "
            f"{', '.join(row['operations_dropped_by_preferred']) or '—'} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks", type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--json", dest="json_out", type=Path, default=None)
    args = parser.parse_args()

    from bench.agent_task_eval import load_tasks

    tasks = load_tasks(args.tasks)
    result = measure([(task.id, task.request) for task in tasks], args.repeats)
    markdown = render(result)
    print(markdown)
    if args.markdown:
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.json_out:
        args.json_out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
