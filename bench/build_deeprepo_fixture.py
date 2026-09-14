"""Build the `deep_repo` fixture: a repository big enough to punish reading it.

`wide-sum` probes navigation over forty modules in one directory. That is not
scale: an agent can read forty files and still finish. This fixture is the same
question with a corpus that cannot be read — ~1 250 leaves behind a three-level
aggregation hierarchy (8 hubs → 40 packages → 1 200 leaves) — so the only
affordable route to the defect is to bisect the hierarchy, and a prompt that
drops exploration has somewhere to fail.

The defect is not greppable on purpose. Every leaf adds a small fixed offset and
several leaves share the same offset, so the total's shortfall names a magnitude
and not a location. The request states the expected total; the visible test
passes on the broken code.

Deterministic and idempotent: rerunning rewrites the same bytes, so the fixture
can be regenerated rather than trusted as a committed blob.

Usage::

    python -m bench.build_deeprepo_fixture [--root bench/agent_task_fixtures/deep_repo]
"""

from __future__ import annotations

import argparse
from pathlib import Path

#: 8 hubs × 5 packages × 30 leaves.
HUBS = 8
PACKAGES_PER_HUB = 5
LEAVES_PER_PACKAGE = 30

#: The leaf that lies about its offset. 517 is mid-corpus, and its offset (9) is
#: shared by ~50 other leaves, so the shortfall cannot be turned into a grep.
BROKEN_LEAF = 517
BROKEN_RETURNS = "v - 40"


def offset(index: int) -> int:
    """Every leaf's declared offset. Small, positive, and often repeated."""

    return 1 + (index * 7) % 23


def total_offset() -> int:
    return sum(offset(i) for i in range(HUBS * PACKAGES_PER_HUB * LEAVES_PER_PACKAGE))


def package_index(hub: int, slot: int) -> int:
    return hub * PACKAGES_PER_HUB + slot


def leaf_index(package: int, position: int) -> int:
    return package * LEAVES_PER_PACKAGE + position


def _leaf_source(index: int) -> str:
    body = BROKEN_RETURNS if index == BROKEN_LEAF else f"v + {offset(index)}"
    return (
        '"""One leaf of the aggregation hierarchy."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "\n"
        f"def shift(v: int) -> int:\n"
        f'    """Apply this leaf\'s offset."""\n'
        f"    return {body}\n"
    )


def _package_source(package: int) -> str:
    imports = "\n".join(
        f"from . import mod_{position:02d}"
        for position in range(LEAVES_PER_PACKAGE)
    )
    sequence = ", ".join(
        f"mod_{position:02d}" for position in range(LEAVES_PER_PACKAGE)
    )
    return (
        '"""One package: thirty leaves applied in order."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        f"{imports}\n"
        "\n"
        f"_LEAVES = ({sequence},)\n"
        "\n"
        "\n"
        "def apply(v: int) -> int:\n"
        f'    """Push ``v`` through package {package}\'s leaves."""\n'
        "    for leaf in _LEAVES:\n"
        "        v = leaf.shift(v)\n"
        "    return v\n"
    )


def _hub_source(hub: int) -> str:
    slots = list(range(PACKAGES_PER_HUB))
    imports = "\n".join(
        f"from ..pkgs.pkg_{package_index(hub, slot):02d} import apply as pkg_{slot}"
        for slot in slots
    )
    sequence = ", ".join(f"pkg_{slot}" for slot in slots)
    return (
        '"""One hub: five packages applied in order."""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        f"{imports}\n"
        "\n"
        f"_PACKAGES = ({sequence},)\n"
        "\n"
        "\n"
        "def apply(v: int) -> int:\n"
        f'    """Push ``v`` through hub {hub}\'s packages."""\n'
        "    for package in _PACKAGES:\n"
        "        v = package(v)\n"
        "    return v\n"
    )


_PIPELINE = '''"""The running total, hub by hub."""

from __future__ import annotations

{hubs}

_HUBS = ({sequence},)


def pipeline(start: int) -> int:
    """Push ``start`` through every hub in order."""
    value = start
    for hub in _HUBS:
        value = hub.apply(value)
    return value


def lowest_possible() -> int:
    """The smallest value the pipeline is allowed to return."""
    return 0
'''

_README = """# Aggregation pipeline

`pipeline(start)` pushes `start` through the whole hierarchy in order:

    8 hubs  ->  5 packages per hub  ->  30 leaves per package

Every leaf applies one fixed positive offset of its own; the packages and the
hubs only chain them. Nothing else in the repository modifies the value, so

    pipeline(0) == the sum of every leaf's offset

which this checkout computes to **{total}**.

Leaves live in `src/pkgs/pkg_NN/mod_MM.py`; each package's `apply` chains its
thirty leaves, each hub's `apply` chains its five packages, and
`src/pipeline.py` chains the eight hubs. The hierarchy is the intended way to
navigate: the leaves are deliberately uniform, so searching the tree by hand is
not the cheap path.
"""

_VISIBLE_TEST = '''from src.pipeline import pipeline


def test_the_pipeline_returns_an_integer() -> None:
    assert isinstance(pipeline(0), int)


def test_the_pipeline_is_monotonic() -> None:
    assert pipeline(10) > pipeline(0)


def test_lowest_possible_is_the_documented_floor() -> None:
    from src.pipeline import lowest_possible

    assert lowest_possible() == 0
'''

_VERIFY = '''"""Hidden behavioural contract for the aggregation pipeline.

Withheld from the agent's workspace and restored only to be run. It checks the
total the request states, so any fix that restores it passes regardless of which
leaf it lands in.
"""

import sys

from src.pipeline import pipeline

#: The sum of every leaf's declared offset, and the shortfall the broken leaf
#: introduces.
EXPECTED_TOTAL = {total}

CASES = (0, 5, -10, 100, -500)

failures: list[str] = []
for start in CASES:
    expected = start + EXPECTED_TOTAL
    try:
        observed = pipeline(start)
    except Exception as exc:  # noqa: BLE001 - report and keep going
        failures.append(f"pipeline({{start}}) raised {{exc!r}}")
        continue
    if observed != expected:
        failures.append(f"pipeline({{start}}) == {{observed}}, expected {{expected}}")

if failures:
    for failure in failures:
        print(failure)
    sys.exit(1)

print(f"{{len(CASES)}} pipeline cases passed")
'''

_PYPROJECT = """[project]
name = "aggregation"
version = "0.1.0"
requires-python = ">=3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
"""


def build(root: Path) -> dict[str, int]:
    """Write the fixture and return a small summary of what was written."""

    if root.exists():
        import shutil

        shutil.rmtree(root)
    (root / "src" / "hubs").mkdir(parents=True)
    (root / "src" / "pkgs").mkdir(parents=True)
    (root / "tests").mkdir(parents=True)

    leaves = 0
    packages = 0
    for hub in range(HUBS):
        (root / "src" / "hubs" / f"hub_{hub}.py").write_text(
            _hub_source(hub), encoding="utf-8"
        )
        for slot in range(PACKAGES_PER_HUB):
            package = package_index(hub, slot)
            package_dir = root / "src" / "pkgs" / f"pkg_{package:02d}"
            package_dir.mkdir()
            (package_dir / "__init__.py").write_text(
                _package_source(package), encoding="utf-8"
            )
            packages += 1
            for position in range(LEAVES_PER_PACKAGE):
                index = leaf_index(package, position)
                (package_dir / f"mod_{position:02d}.py").write_text(
                    _leaf_source(index), encoding="utf-8"
                )
                leaves += 1

    (root / "src" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "hubs" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "pkgs" / "__init__.py").write_text("", encoding="utf-8")
    (root / "src" / "pipeline.py").write_text(
        _PIPELINE.format(
            hubs="\n".join(f"from .hubs import hub_{hub}" for hub in range(HUBS)),
            sequence=", ".join(f"hub_{hub}" for hub in range(HUBS)),
        ),
        encoding="utf-8",
    )
    total = total_offset()
    (root / "README.md").write_text(_README.format(total=total), encoding="utf-8")
    (root / "tests" / "test_total_visible.py").write_text(
        _VISIBLE_TEST, encoding="utf-8"
    )
    (root / "verify_contract.py").write_text(
        _VERIFY.format(total=total), encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    return {"leaves": leaves, "packages": packages, "hubs": HUBS, "total": total}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path,
        default=Path("bench/agent_task_fixtures/deep_repo"),
    )
    args = parser.parse_args()
    summary = build(args.root)
    print(args.root)
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
