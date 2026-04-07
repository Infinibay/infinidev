#!/usr/bin/env python3
"""Evaluate Infinidev predictions against SWE-bench.

Usage:
    # Evaluate predictions
    python -m bench.evaluate

    # Custom predictions file
    python -m bench.evaluate --predictions bench/predictions.jsonl

    # Summary only (no per-instance details)
    python -m bench.evaluate --summary-only
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def load_predictions(path: Path) -> list[dict]:
    """Load predictions from JSONL file."""
    predictions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))
    return predictions


def print_summary(predictions: list[dict]) -> None:
    """Print a quick summary of predictions."""
    total = len(predictions)
    with_patch = sum(1 for p in predictions if p.get("model_patch", "").strip())
    empty = total - with_patch

    print(f"\n{'='*60}")
    print(f"  Predictions Summary")
    print(f"{'='*60}")
    print(f"  Total instances:    {total}")
    print(f"  With patches:       {with_patch} ({100*with_patch/total:.1f}%)" if total else "")
    print(f"  Empty (no changes): {empty}")
    print(f"{'='*60}\n")


def run_swebench_eval(
    predictions_path: Path,
    dataset: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    run_id: str = "infinidev_eval",
    max_workers: int = 1,
    timeout: int = 1800,
) -> None:
    """Run official SWE-bench evaluation via the swebench CLI.

    Spawns ``python -m swebench.harness.run_evaluation`` as a
    subprocess instead of importing main(): the swebench 4.x ``main()``
    function takes positional arguments and isn't argv-parseable
    in-process. The subprocess approach is also better isolated —
    swebench mutates global state heavily.

    Requires: ``pip install swebench`` and either Docker or Podman
    with the docker-socket compatibility layer enabled. When using
    Podman, set ``DOCKER_HOST=unix:///run/user/$UID/podman/podman.sock``
    in your environment before invoking this script.
    """
    import subprocess

    try:
        import swebench  # noqa: F401
    except ImportError:
        log.error(
            "swebench not installed. Install with:\n"
            "  pip install swebench\n"
            "Then re-run this script."
        )
        sys.exit(1)

    log.info("Running SWE-bench evaluation on %s...", predictions_path)
    log.info("Dataset=%s split=%s run_id=%s", dataset, split, run_id)

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "--dataset_name", dataset,
        "--split", split,
        "--predictions_path", str(predictions_path),
        "--max_workers", str(max_workers),
        "--run_id", run_id,
        "--timeout", str(timeout),
    ]
    log.info("Command: %s", " ".join(cmd))

    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        log.error("swebench evaluator exited with code %d", proc.returncode)
        sys.exit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Infinidev SWE-bench predictions")
    parser.add_argument("--predictions", default="bench/predictions.jsonl", help="Predictions JSONL file")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite", help="SWE-bench dataset")
    parser.add_argument("--split", default="test", help="Dataset split")
    parser.add_argument("--run-id", default="infinidev_eval", help="swebench run identifier")
    parser.add_argument("--max-workers", type=int, default=1, help="Parallel evaluator workers")
    parser.add_argument("--timeout", type=int, default=1800, help="Per-instance test timeout (seconds)")
    parser.add_argument("--summary-only", action="store_true", help="Only show prediction summary, don't run eval")
    args = parser.parse_args()

    predictions_path = Path(args.predictions)
    if not predictions_path.exists():
        log.error("Predictions file not found: %s", predictions_path)
        log.error("Run `python -m bench.run_swebench` first to generate predictions.")
        sys.exit(1)

    predictions = load_predictions(predictions_path)
    print_summary(predictions)

    if args.summary_only:
        # Print per-instance status
        for p in predictions:
            has_patch = bool(p.get("model_patch", "").strip())
            status = "PATCH" if has_patch else "EMPTY"
            print(f"  [{status}] {p['instance_id']}")
        return

    run_swebench_eval(
        predictions_path,
        dataset=args.dataset,
        split=args.split,
        run_id=args.run_id,
        max_workers=args.max_workers,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
