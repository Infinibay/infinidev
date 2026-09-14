"""The `deep_repo` fixture: big enough to punish reading it, and solvable.

The scale probe only means something if three things hold: the tree really is
too big to read, the visible tests really do pass on the broken code (so the
stated total is the only signal), and exactly one leaf really is wrong (so the
task is a localization and not a lottery).
"""

from __future__ import annotations

import re
from pathlib import Path

from bench.build_deeprepo_fixture import (
    BROKEN_LEAF,
    HUBS,
    LEAVES_PER_PACKAGE,
    PACKAGES_PER_HUB,
    build,
    leaf_index,
    offset,
    total_offset,
)


def _build(tmp_path: Path) -> Path:
    root = tmp_path / "deep_repo"
    build(root)
    return root


def test_the_tree_is_too_big_to_read_one_file_at_a_time(tmp_path: Path) -> None:
    root = _build(tmp_path)

    leaves = list((root / "src" / "pkgs").glob("pkg_*/mod_*.py"))

    assert len(leaves) == HUBS * PACKAGES_PER_HUB * LEAVES_PER_PACKAGE == 1_200
    assert len(list((root / "src" / "pkgs").glob("pkg_*"))) == 40
    assert len(list((root / "src" / "hubs").glob("hub_*.py"))) == 8


def test_exactly_one_leaf_lies_about_its_offset(tmp_path: Path) -> None:
    root = _build(tmp_path)

    wrong = []
    for package in range(HUBS * PACKAGES_PER_HUB):
        for position in range(LEAVES_PER_PACKAGE):
            index = leaf_index(package, position)
            source = (root / "src" / "pkgs"
                      / f"pkg_{package:02d}" / f"mod_{position:02d}.py").read_text()
            if f"return v + {offset(index)}" not in source:
                wrong.append(index)

    assert wrong == [BROKEN_LEAF]


def test_the_shortfall_is_shared_by_about_fifty_leaves(tmp_path: Path) -> None:
    """The delta names a magnitude, not a location.

    If only one leaf carried the broken offset, the task would collapse into
    grepping for that constant.
    """
    shares = [i for i in range(HUBS * PACKAGES_PER_HUB * LEAVES_PER_PACKAGE)
              if offset(i) == offset(BROKEN_LEAF)]

    assert len(shares) > 20


def test_the_visible_suite_passes_on_the_broken_tree(tmp_path: Path) -> None:
    """The agent's only signal is the total the request states."""
    import subprocess
    import sys

    root = _build(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=root, capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_the_verifier_rejects_pristine_and_accepts_the_fix(tmp_path: Path) -> None:
    import subprocess
    import sys

    root = _build(tmp_path)
    pristine = subprocess.run(
        [sys.executable, "verify_contract.py"], cwd=root,
        capture_output=True, text=True,
    )
    assert pristine.returncode == 1
    assert f"expected {total_offset()}" in pristine.stdout

    package, position = divmod(BROKEN_LEAF, LEAVES_PER_PACKAGE)
    target = root / "src" / "pkgs" / f"pkg_{package:02d}" / f"mod_{position:02d}.py"
    target.write_text(
        re.sub(r"return v - 40", f"return v + {offset(BROKEN_LEAF)}",
               target.read_text()),
        encoding="utf-8",
    )
    fixed = subprocess.run(
        [sys.executable, "verify_contract.py"], cwd=root,
        capture_output=True, text=True,
    )
    assert fixed.returncode == 0, fixed.stdout + fixed.stderr
    # Leaf 517 is position 7 of package 17, which is what the fix must touch.
    assert divmod(BROKEN_LEAF, LEAVES_PER_PACKAGE) == (17, 7)


def test_regenerating_writes_the_same_bytes(tmp_path: Path) -> None:
    """The fixture is reproducible, not a trusted blob."""
    root = _build(tmp_path)
    before = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }
    build(root)
    after = {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*") if path.is_file()
    }

    assert before == after
