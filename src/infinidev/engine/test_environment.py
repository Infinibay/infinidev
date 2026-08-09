"""Ephemeral environment adjustments for tests run against a checkout."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
import re
import shlex

_PYTEST_COMMAND = re.compile(
    r"(?:^|[;&|]\s*)"
    r"(?:[A-Za-z_][A-Za-z0-9_]*=[^\s]+\s+)*"
    r"(?:(?:\S*/)?python(?:\d+(?:\.\d+)*)?\s+-m\s+pytest|"
    r"(?:\S*/)?(?:pytest|py\.test))(?:\s|$)"
)


def prepare_test_environment(
    command: str | Sequence[str],
    cwd: str,
    *,
    explicit_env: Mapping[str, str] | None = None,
    base_env: Mapping[str, str] | None = None,
) -> tuple[dict[str, str], str | None]:
    """Return a subprocess environment that tests the checked-out package.

    The agent's Python environment belongs to Infinidev, not to the repository
    under test.  In an observable src-layout checkout, pytest should import the
    checkout instead of a same-named distribution from site-packages.  The
    adjustment is process-local and explicit caller configuration always wins.
    """
    run_env = {
        str(key): str(value)
        for key, value in (base_env or os.environ).items()
    }
    if explicit_env:
        run_env.update({str(key): str(value) for key, value in explicit_env.items()})

    display_command = command if isinstance(command, str) else shlex.join(command)
    if not _PYTEST_COMMAND.search(display_command):
        return run_env, None
    if "PYTHONPATH" in (explicit_env or {}) or re.search(
        r"(?:^|\s)PYTHONPATH=", display_command
    ):
        return run_env, None

    src_dir = os.path.realpath(os.path.join(cwd, "src"))
    if not os.path.isdir(src_dir) or not _contains_python_package(src_dir):
        return run_env, None

    inherited = run_env.get("PYTHONPATH", "")
    run_env["PYTHONPATH"] = (
        src_dir if not inherited else f"{src_dir}{os.pathsep}{inherited}"
    )
    adjustment = f"Prepended {src_dir} to PYTHONPATH for src-layout pytest execution"
    return run_env, adjustment


def _contains_python_package(src_dir: str) -> bool:
    try:
        for entry in os.scandir(src_dir):
            if not entry.is_dir():
                continue
            if os.path.isfile(os.path.join(entry.path, "__init__.py")):
                return True
            if any(
                child.is_file() and child.name.endswith(".py")
                for child in os.scandir(entry.path)
            ):
                return True
    except OSError:
        return False
    return False


__all__ = ["prepare_test_environment"]
