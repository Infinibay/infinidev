#!/usr/bin/env python3
"""Run the complete fine-tuning dataset generation pipeline.

Usage:
    python -m finetune.run_pipeline              # Run all steps
    python -m finetune.run_pipeline download      # Only download repos
    python -m finetune.run_pipeline index          # Only index repos
    python -m finetune.run_pipeline scenarios      # Only generate scenarios
    python -m finetune.run_pipeline examples       # Only generate examples
    python -m finetune.run_pipeline format          # Only format dataset
"""

import sys


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"

    if step in ("all", "download"):
        print("=" * 60)
        print("STEP 1: Download repos")
        print("=" * 60)
        from finetune.download_repos import download_all
        download_all()

    if step in ("all", "index"):
        print("\n" + "=" * 60)
        print("STEP 2: Index repos")
        print("=" * 60)
        from finetune.index_repos import index_all
        index_all()

    if step in ("all", "scenarios"):
        print("\n" + "=" * 60)
        print("STEP 3: Generate scenarios")
        print("=" * 60)
        from finetune.generate_scenarios import generate_all
        generate_all()

    if step in ("all", "examples"):
        print("\n" + "=" * 60)
        print("STEP 4: Generate training examples")
        print("=" * 60)
        from finetune.generate_examples import generate_all
        generate_all()

    if step in ("all", "format"):
        print("\n" + "=" * 60)
        print("STEP 5: Format dataset")
        print("=" * 60)
        from finetune.format_dataset import format_all
        format_all()

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
