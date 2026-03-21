#!/usr/bin/env python3
"""Test predictions by applying patches and running FAIL_TO_PASS tests.

This is a lightweight local tester — not the official swebench evaluator,
but gives a quick signal on whether patches actually fix the failing tests.
"""

import json
import logging
import subprocess
import sys
from pathlib import Path

from bench.repo_setup import clone_or_cache, cleanup_instance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def run(cmd, cwd=None, timeout=300):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)


def test_instance(instance, model_patch, workdir, cache_dir):
    """Apply model_patch + test_patch, then run FAIL_TO_PASS tests."""
    iid = instance["instance_id"]
    repo = instance["repo"]
    base_commit = instance["base_commit"]
    test_patch = instance.get("test_patch", "")
    fail_to_pass = json.loads(instance["FAIL_TO_PASS"])

    log.info("Testing %s (%d tests to check)", iid, len(fail_to_pass))

    # Setup repo
    cached = clone_or_cache(repo, cache_dir)
    instance_dir = workdir / f"test__{iid.replace('/', '__')}"
    if instance_dir.exists():
        import shutil
        shutil.rmtree(instance_dir)

    instance_dir.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--no-checkout", str(cached), str(instance_dir)])
    run(["git", "checkout", base_commit], cwd=instance_dir)

    # Apply test patch first (adds the test cases that should fail before fix)
    if test_patch:
        proc = run(["git", "apply", "--allow-empty", "-"], cwd=instance_dir)
        # Use stdin for patch
        proc = subprocess.run(
            ["git", "apply", "--allow-empty"],
            input=test_patch, cwd=instance_dir,
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            log.warning("  Test patch failed to apply: %s", proc.stderr[:200])
            # Try with --reject
            subprocess.run(
                ["git", "apply", "--allow-empty", "--reject"],
                input=test_patch, cwd=instance_dir,
                capture_output=True, text=True,
            )

    # Apply model patch
    proc = subprocess.run(
        ["git", "apply", "--allow-empty"],
        input=model_patch, cwd=instance_dir,
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        log.error("  Model patch failed to apply: %s", proc.stderr[:200])
        cleanup_instance(instance_dir)
        return {"instance_id": iid, "status": "PATCH_FAIL", "details": proc.stderr[:200]}

    # Run FAIL_TO_PASS tests
    # Determine test runner based on repo
    if "django" in repo:
        # Django uses its own test runner
        test_results = []
        for test_id in fail_to_pass:
            # Convert test_id format: "test_name (module.Class)" -> module path
            proc = run(
                ["python", "-m", "pytest", "-xvs", test_id],
                cwd=instance_dir, timeout=120,
            )
            test_results.append(proc.returncode == 0)
    else:
        # Use pytest
        proc = run(
            ["python", "-m", "pytest", "-xvs"] + fail_to_pass,
            cwd=instance_dir, timeout=120,
        )
        test_results = [proc.returncode == 0]

    cleanup_instance(instance_dir)

    passed = all(test_results)
    status = "PASS" if passed else "FAIL"
    log.info("  %s: %s", iid, status)

    return {
        "instance_id": iid,
        "status": status,
        "tests_run": len(fail_to_pass),
        "all_passed": passed,
    }


def main():
    predictions_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("~/forcloude/swebench-run/predictions.jsonl").expanduser()
    workdir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("~/forcloude/swebench-run").expanduser()
    cache_dir = Path("/tmp/infinidev-bench/.cache")

    from datasets import load_dataset
    ds = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    instances = {item["instance_id"]: item for item in ds}

    with open(predictions_path) as f:
        preds = [json.loads(line) for line in f if line.strip()]

    results = []
    for pred in preds:
        iid = pred["instance_id"]
        patch = pred.get("model_patch", "").strip()
        if not patch:
            log.info("Skipping %s (no patch)", iid)
            results.append({"instance_id": iid, "status": "NO_PATCH"})
            continue

        if iid not in instances:
            log.warning("Instance %s not found in dataset", iid)
            continue

        try:
            result = test_instance(instances[iid], patch, workdir, cache_dir)
            results.append(result)
        except Exception as e:
            log.exception("Error testing %s", iid)
            results.append({"instance_id": iid, "status": "ERROR", "details": str(e)})

    # Summary
    print(f"\n{'='*60}")
    print(f"  Test Results Summary")
    print(f"{'='*60}")
    for r in results:
        icon = {"PASS": "+", "FAIL": "-", "NO_PATCH": " ", "PATCH_FAIL": "!", "ERROR": "!"}
        print(f"  [{icon.get(r['status'], '?')}] {r['instance_id']:40s} {r['status']}")

    passed = sum(1 for r in results if r["status"] == "PASS")
    total_with_patch = sum(1 for r in results if r["status"] not in ("NO_PATCH",))
    print(f"\n  Resolved: {passed}/{total_with_patch} (of those with patches)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
