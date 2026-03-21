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


def run_swebench_eval(predictions_path: Path, dataset: str = "princeton-nlp/SWE-bench_Lite") -> None:
    """Run official SWE-bench evaluation.

    Requires: pip install swebench
    """
    try:
        from swebench.harness.run_evaluation import main as swebench_main
    except ImportError:
        log.error(
            "swebench not installed. Install with:\n"
            "  pip install swebench\n"
            "Then re-run this script."
        )
        sys.exit(1)

    # Convert JSONL predictions to the format swebench expects
    predictions = load_predictions(predictions_path)

    # swebench expects a JSON file with list of predictions
    eval_input = predictions_path.parent / "predictions_eval.json"
    with open(eval_input, "w") as f:
        json.dump(predictions, f, indent=2)

    log.info("Running SWE-bench evaluation on %d predictions...", len(predictions))
    log.info("This may take a while as it applies and tests each patch.")

    sys.argv = [
        "run_evaluation",
        "--predictions_path", str(eval_input),
        "--swe_bench_tasks", dataset,
        "--log_level", "INFO",
        "--timeout", "900",
    ]

    try:
        swebench_main()
    except SystemExit:
        pass


def main():
    parser = argparse.ArgumentParser(description="Evaluate Infinidev SWE-bench predictions")
    parser.add_argument("--predictions", default="bench/predictions.jsonl", help="Predictions JSONL file")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Lite", help="SWE-bench dataset")
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

    run_swebench_eval(predictions_path, args.dataset)


if __name__ == "__main__":
    main()
