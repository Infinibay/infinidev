#!/usr/bin/env python3
"""Humanity's Last Exam (HLE) evaluation harness for Infinidev.

Runs infinidev as an agent on HLE questions — the agent can use tools
(web_search, execute_command, etc.) to research and answer.

Usage:
    python -m bench.run_hle --max-instances 10
    python -m bench.run_hle --category "Math" --max-instances 5
    python -m bench.run_hle --model ollama_chat/qwen3.5:27b
"""

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("bench/hle_run.log"),
    ],
)
log = logging.getLogger(__name__)


def load_dataset_hle(max_instances=0, category=None, answer_type=None):
    """Load HLE dataset from HuggingFace."""
    from datasets import load_dataset
    log.info("Loading HLE dataset...")
    ds = load_dataset("cais/hle", split="test")
    instances = list(ds)

    if category:
        instances = [i for i in instances if i["category"] == category]
        log.info("Filtered to category '%s': %d instances", category, len(instances))

    if answer_type:
        instances = [i for i in instances if i["answer_type"] == answer_type]
        log.info("Filtered to answer_type '%s': %d instances", answer_type, len(instances))

    if max_instances > 0:
        instances = instances[:max_instances]

    log.info("Will process %d instances", len(instances))
    return instances


def build_prompt(instance):
    """Build the prompt for infinidev from an HLE instance."""
    question = instance["question"]
    answer_type = instance["answer_type"]
    category = instance["category"]

    # Use /explore prefix so infinidev routes to TreeEngine
    prompt = f"""/explore Answer the following expert-level question. Research thoroughly using all available tools — search the web, run calculations, write and execute code, look up references. Accuracy matters more than speed.

## Category: {category}

## Question

{question}

"""
    if answer_type == "multipleChoice":
        prompt += """This is a multiple choice question. Your final answer MUST be exactly one letter (A, B, C, D, or E).
End your response with: FINAL ANSWER: <letter>
"""
    else:
        prompt += """This is a free-form question. Your final answer should be a short, precise response.
End your response with: FINAL ANSWER: <your answer>
"""
    return prompt


def extract_answer(output, answer_type):
    """Extract the final answer from infinidev's output."""
    if not output:
        return ""

    # Look for FINAL ANSWER: pattern
    match = re.search(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", output, re.IGNORECASE)
    if match:
        answer = match.group(1).strip()
        if answer_type == "multipleChoice":
            # Extract just the letter
            letter_match = re.match(r"([A-E])", answer.upper())
            if letter_match:
                return letter_match.group(1)
        return answer

    # Fallback: for MC, look for standalone letter at end
    if answer_type == "multipleChoice":
        lines = output.strip().split("\n")
        for line in reversed(lines[-5:]):
            line = line.strip()
            if re.match(r"^[A-E]\.?\s*$", line):
                return line[0].upper()

    # Last line as fallback
    lines = output.strip().split("\n")
    return lines[-1].strip() if lines else ""


def normalize(s):
    """Normalize answer for comparison."""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(".")
    return s


def score_answer(predicted, gold, answer_type):
    """Score a predicted answer against gold."""
    if answer_type == "multipleChoice":
        return predicted.strip().upper() == gold.strip().upper()
    else:
        return normalize(predicted) == normalize(gold)


def run_instance(instance, model, timeout=600):
    """Run infinidev on a single HLE instance."""
    iid = instance["id"]
    prompt = build_prompt(instance)

    # Create a temp working directory
    workdir = Path(f"/tmp/infinidev-hle/{iid}")
    workdir.mkdir(parents=True, exist_ok=True)

    infinidev_cmd = shutil.which("infinidev")
    if infinidev_cmd:
        cmd = [infinidev_cmd]
    else:
        cmd = [sys.executable, "-m", "infinidev.cli.main"]

    try:
        start = time.time()
        proc = subprocess.run(
            cmd + [
                "--prompt", prompt,
                "--model", model,
                "--no-tui",
            ],
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=timeout or None,
        )
        elapsed = time.time() - start
        output = proc.stdout
        log.info("Finished %s in %.1fs", iid, elapsed)
        return output
    except subprocess.TimeoutExpired:
        log.error("TIMEOUT on %s", iid)
        return ""
    except Exception:
        log.exception("Error on %s", iid)
        return ""
    finally:
        # Cleanup
        if workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Run HLE evaluation with Infinidev")
    parser.add_argument("--model", default="ollama_chat/qwen3.5:27b")
    parser.add_argument("--max-instances", type=int, default=10)
    parser.add_argument("--category", default=None, help="Filter by category (Math, Physics, etc.)")
    parser.add_argument("--answer-type", default=None, help="Filter: exactMatch or multipleChoice")
    parser.add_argument("--output", default="bench/hle_predictions.jsonl")
    parser.add_argument("--timeout", type=int, default=0, help="Timeout per instance (0=none)")
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    if args.no_resume:
        args.resume = False

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load completed
    completed = set()
    if args.resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    pred = json.loads(line)
                    completed.add(pred["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        if completed:
            log.info("Resuming: %d instances already completed", len(completed))

    instances = load_dataset_hle(args.max_instances, args.category, args.answer_type)

    log.info("HLE Infinidev Harness")
    log.info("Model: %s", args.model)
    log.info("Instances: %d", len(instances))

    correct = 0
    total = 0
    skipped = 0

    for i, instance in enumerate(instances, 1):
        iid = instance["id"]

        if iid in completed:
            skipped += 1
            continue

        log.info("[%d/%d] %s (%s, %s)", i, len(instances), iid, instance["category"], instance["answer_type"])

        output = run_instance(instance, args.model, args.timeout)
        predicted = extract_answer(output, instance["answer_type"])
        is_correct = score_answer(predicted, instance["answer"], instance["answer_type"])

        total += 1
        if is_correct:
            correct += 1

        status = "CORRECT" if is_correct else "WRONG"
        log.info("  %s | predicted='%s' | gold='%s'", status, predicted[:50], instance["answer"][:50])

        prediction = {
            "id": iid,
            "category": instance["category"],
            "answer_type": instance["answer_type"],
            "gold_answer": instance["answer"],
            "predicted_answer": predicted,
            "correct": is_correct,
            "model": args.model,
            "raw_output": output[-500:] if output else "",
        }

        with open(output_path, "a") as f:
            f.write(json.dumps(prediction) + "\n")

    # Summary
    pct = (100 * correct / total) if total > 0 else 0
    log.info("=" * 60)
    log.info("  HLE Results: %d/%d correct (%.1f%%)", correct, total, pct)
    log.info("  Skipped (resume): %d", skipped)
    log.info("=" * 60)

    print(f"\nHLE Results: {correct}/{total} correct ({pct:.1f}%)")
    print(f"Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
