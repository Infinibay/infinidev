#!/usr/bin/env python3
"""SWE-bench evaluation harness for Infinidev.

Usage:
    # Run all SWE-bench Lite instances with default model
    python -m bench.run_swebench

    # Run specific instances
    python -m bench.run_swebench --instance-id django__django-16379

    # Use a different model
    python -m bench.run_swebench --model ollama_chat/qwen3:32b

    # Limit to N instances (useful for testing)
    python -m bench.run_swebench --max-instances 5

    # Custom timeout per instance
    python -m bench.run_swebench --timeout 900
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from bench.config import BenchConfig
from bench.repo_setup import setup_instance, get_patch, cleanup_instance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("bench/run.log"),
    ],
)
log = logging.getLogger(__name__)


def load_dataset(config: BenchConfig) -> list[dict]:
    """Load SWE-bench dataset from HuggingFace."""
    from datasets import load_dataset
    log.info("Loading dataset %s (split=%s)...", config.dataset, config.split)
    ds = load_dataset(config.dataset, split=config.split)
    instances = list(ds)

    if config.instance_ids:
        instances = [i for i in instances if i["instance_id"] in config.instance_ids]
        log.info("Filtered to %d instance(s)", len(instances))

    if config.max_instances > 0:
        instances = instances[: config.max_instances]
        log.info("Limited to %d instance(s)", len(instances))

    return instances


def load_completed(config: BenchConfig) -> set[str]:
    """Load instance IDs already in predictions file."""
    completed = set()
    if config.resume and config.output.exists():
        with open(config.output) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                    completed.add(pred["instance_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return completed


def build_prompt(instance: dict) -> str:
    """Build the prompt to send to infinidev from a SWE-bench instance."""
    problem = instance["problem_statement"]
    hints = instance.get("hints_text", "")

    prompt = f"""Fix the following issue in this repository.

## Issue

{problem}
"""
    if hints:
        prompt += f"""
## Hints

{hints}
"""
    prompt += """
## Instructions

1. Find the relevant files and understand the bug
2. Implement the fix
3. Make sure your changes are minimal and focused on the issue
"""
    return prompt


def run_instance(instance: dict, config: BenchConfig) -> dict:
    """Run infinidev on a single SWE-bench instance.

    Returns a prediction dict with instance_id and model_patch.
    """
    instance_id = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]

    log.info("=== Running instance: %s ===", instance_id)
    log.info("Repo: %s | Base commit: %s", repo, base_commit[:12])

    # Setup repo
    instance_dir = setup_instance(
        repo=repo,
        base_commit=base_commit,
        workdir=config.workdir,
        cache_dir=config.cache_dir,
        instance_id=instance_id,
    )

    prompt = build_prompt(instance)
    result = {
        "instance_id": instance_id,
        "model_name_or_path": config.model,
        "model_patch": "",
    }

    try:
        start = time.time()

        # Run infinidev non-interactively
        env = os.environ.copy()
        env["INFINIDEV_WORKSPACE"] = str(instance_dir)

        # Always use the project's source via python -m to ensure latest code
        cmd = [sys.executable, "-m", "infinidev.cli.main"]

        proc = subprocess.run(
            cmd + [
                "--prompt", prompt,
                "--model", config.model,
                "--no-tui",
            ],
            cwd=str(instance_dir),
            stdout=subprocess.PIPE,
            stderr=None,  # Let stderr flow through to terminal
            text=True,
            timeout=config.timeout or None,
            env=env,
        )

        elapsed = time.time() - start
        log.info("Finished %s in %.1fs (exit=%d)", instance_id, elapsed, proc.returncode)

        # Collect the patch
        patch = get_patch(instance_dir)
        result["model_patch"] = patch

        if not patch.strip():
            log.warning("No changes produced for %s", instance_id)
        else:
            lines = patch.count("\n")
            log.info("Patch for %s: %d lines", instance_id, lines)

    except subprocess.TimeoutExpired:
        log.error("TIMEOUT on %s after %ds", instance_id, config.timeout)
        result["model_patch"] = ""
    except Exception:
        log.exception("Error running %s", instance_id)
        result["model_patch"] = ""
    finally:
        cleanup_instance(instance_dir)

    return result


def save_prediction(prediction: dict, output_path: Path) -> None:
    """Append a prediction to the JSONL output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "a") as f:
        f.write(json.dumps(prediction) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run SWE-bench evaluation with Infinidev")
    parser.add_argument("--model", default=None, help="LLM model override (LiteLLM format)")
    parser.add_argument("--dataset", default=None, help="HuggingFace dataset name")
    parser.add_argument("--split", default=None, help="Dataset split")
    parser.add_argument("--max-instances", type=int, default=None, help="Max instances to run (0=all)")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout per instance in seconds")
    parser.add_argument("--instance-id", action="append", dest="instance_ids", help="Run specific instance(s)")
    parser.add_argument("--output", default=None, help="Output predictions JSONL file")
    parser.add_argument("--workdir", default=None, help="Working directory for repo checkouts")
    parser.add_argument("--no-resume", action="store_true", help="Don't skip already-completed instances")
    parser.add_argument("--no-cleanup", action="store_true", help="Keep instance dirs after run")
    args = parser.parse_args()

    config = BenchConfig()
    if args.model:
        config.model = args.model
    if args.dataset:
        config.dataset = args.dataset
    if args.split:
        config.split = args.split
    if args.max_instances is not None:
        config.max_instances = args.max_instances
    if args.timeout:
        config.timeout = args.timeout
    if args.instance_ids:
        config.instance_ids = args.instance_ids
    if args.output:
        config.output = Path(args.output)
    if args.workdir:
        config.workdir = Path(args.workdir)
    if args.no_resume:
        config.resume = False

    config.workdir.mkdir(parents=True, exist_ok=True)

    log.info("SWE-bench Infinidev Harness")
    log.info("Model: %s", config.model)
    log.info("Dataset: %s [%s]", config.dataset, config.split)
    log.info("Output: %s", config.output)

    instances = load_dataset(config)
    completed = load_completed(config)

    if completed:
        log.info("Resuming: %d instances already completed", len(completed))

    total = len(instances)
    done = 0
    skipped = 0

    for i, instance in enumerate(instances, 1):
        iid = instance["instance_id"]

        if iid in completed:
            skipped += 1
            log.info("[%d/%d] Skipping %s (already done)", i, total, iid)
            continue

        log.info("[%d/%d] Processing %s", i, total, iid)
        prediction = run_instance(instance, config)
        save_prediction(prediction, config.output)
        done += 1

        has_patch = bool(prediction["model_patch"].strip())
        log.info("[%d/%d] %s: %s", i, total, iid, "PATCH" if has_patch else "NO PATCH")

    log.info("=== Complete: %d done, %d skipped, %d total ===", done, skipped, total)
    log.info("Predictions saved to: %s", config.output)


if __name__ == "__main__":
    main()
