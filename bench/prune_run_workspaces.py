"""Free the disk a campaign leaves behind, without losing its measurements.

Every agent-task run copies its whole fixture tree into
``<arm>/artifacts/<task>.r<N>.<condition>/workspace`` and writes the run's
measurements next to it in ``run.json``.  The fixture trees are large — the
deep-repo corpus is 1 256 files — so 300 runs left **105 GB** of copies in
``bench/runs``, and the volume filled up mid-campaign.  The failure is not
loud: the run that hit it died with ``ENOSPC``, and two unrelated test runs
concurrently on the same volume reported a hundred errors that had nothing to
do with the code under test.

The copies are not worthless — ``agent_task_outcome_review`` reads the changed
files back out of them — so this never deletes by default.  It reports what is
there, and ``--apply`` removes only the ``workspace`` directories, which leaves
every ``run.json``, every ``observations.jsonl``, every ``probes.json`` and both
``comparison.*`` untouched.  Anything that needs a file's *content* after that
has to come from the run's own record instead, which is the honest version of
this: evidence belongs in the artifact, not in a scratch copy of the repository.

Usage::

    python -m bench.prune_run_workspaces --runs bench/runs
    python -m bench.prune_run_workspaces --runs bench/runs --apply
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


def _directory_size(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        try:
            if item.is_file() and not item.is_symlink():
                total += item.stat().st_size
        except OSError:
            continue
    return total


@dataclass(frozen=True)
class WorkspaceCopy:
    """One ``artifacts/<run>/workspace`` copy and what it is worth keeping."""

    workspace: Path
    bytes_used: int
    has_artifact: bool

    @property
    def is_orphan(self) -> bool:
        """True when there is no ``run.json`` beside it to justify the copy.

        An orphan is a copy from a run that died before writing its artifact —
        the ENOSPC case — and carries no evidence at all.
        """
        return not self.has_artifact


def find_copies(run_root: Path) -> list[WorkspaceCopy]:
    copies: list[WorkspaceCopy] = []
    for workspace in sorted(run_root.rglob("workspace")):
        if not workspace.is_dir() or workspace.parent.name == "":
            continue
        # Only the copies the runner makes: a fixture's own checkout can be
        # named ``workspace`` too, and that one is an input, not an output.
        if workspace.parent.parent.name != "artifacts":
            continue
        copies.append(
            WorkspaceCopy(
                workspace=workspace,
                bytes_used=_directory_size(workspace),
                has_artifact=(workspace.parent / "run.json").is_file(),
            )
        )
    return copies


def human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:,.1f} {unit}"
        value /= 1024
    return f"{value:,.1f} TB"


def prune(run_root: Path, *, apply: bool, keep_artifacts: bool) -> dict[str, object]:
    copies = find_copies(run_root)
    orphans = [c for c in copies if c.is_orphan]
    targeted = orphans if keep_artifacts else copies
    freed = 0
    removed = 0
    for copy in targeted:
        freed += copy.bytes_used
        if apply:
            import shutil

            shutil.rmtree(copy.workspace, ignore_errors=True)
            removed += 1
    return {
        "copies": len(copies),
        "orphans": len(orphans),
        "targeted": len(targeted),
        "bytes_targeted": freed,
        "bytes_total": sum(c.bytes_used for c in copies),
        "removed": removed,
        "applied": apply,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("bench/runs"))
    parser.add_argument(
        "--apply",
        action="store_true",
        help="actually delete; without it this only reports",
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="remove only copies whose run never wrote a run.json",
    )
    args = parser.parse_args()
    result = prune(args.runs, apply=args.apply, keep_artifacts=args.keep_artifacts)
    print(f"artifact workspaces under {args.runs}: {result['copies']}")
    print(f"  holding                        {human(int(result['bytes_total']))}")
    print(f"  with no run.json beside them   {result['orphans']}")
    if not result["targeted"]:
        print("\nnothing to remove")
        return
    if result["applied"]:
        print(
            f"\nremoved {result['removed']} copies, freeing "
            f"{human(int(result['bytes_targeted']))}; every run.json is untouched"
        )
    else:
        print(
            f"\nwould remove {result['targeted']} copies, freeing "
            f"{human(int(result['bytes_targeted']))}; pass --apply to do it"
        )


if __name__ == "__main__":
    main()
