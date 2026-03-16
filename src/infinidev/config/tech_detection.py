"""Technology detection for project workspaces."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def detect_tech_hints(workspace_path: str) -> list[str]:
    """Scan a workspace directory for technology indicator files.

    Returns a deduplicated list of technology names (e.g. ["python", "docker"]).
    Never raises -- returns [] on any error.
    """
    try:
        root = Path(workspace_path)
        if not root.is_dir():
            return []

        hints: list[str] = []
        _detect_from_dir(root, hints)
        return list(dict.fromkeys(hints))

    except Exception:
        logger.warning(
            "detect_tech_hints: failed for %s, returning empty list",
            workspace_path,
            exc_info=True,
        )
        return []


def _detect_from_dir(root: Path, hints: list[str]) -> None:
    """Populate *hints* by scanning *root* for technology indicators."""

    _SEARCH_DIRS = [
        root,
        root / "src",
        root / "apps",
        root / "packages",
        root / "lib",
        root / "cmd",
    ]

    def _has(name: str) -> bool:
        return any((d / name).exists() for d in _SEARCH_DIRS)

    _SKIP_DIRS = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".tox",
                   "target", "build", "dist", ".mypy_cache", ".pytest_cache"}

    def _any_ext(*exts: str) -> bool:
        """Check for files with given extensions, skipping large directories."""
        ext_set = set(exts)
        try:
            for item in root.iterdir():
                if item.is_file() and any(item.name.endswith(e) for e in ext_set):
                    return True
                if item.is_dir() and item.name not in _SKIP_DIRS and not item.name.startswith("."):
                    try:
                        for sub in item.iterdir():
                            if sub.is_file() and any(sub.name.endswith(e) for e in ext_set):
                                return True
                    except PermissionError:
                        continue
        except PermissionError:
            pass
        return False

    def _file_contains(name: str, *needles: str) -> bool:
        for d in _SEARCH_DIRS:
            p = d / name
            if not p.is_file():
                continue
            try:
                content = p.read_text(errors="ignore")
                if any(n in content for n in needles):
                    return True
            except OSError:
                continue
        return False

    # Languages
    if _has("pyproject.toml") or _has("setup.py") or _has("requirements.txt") or _any_ext(".py"):
        hints.append("python")
    if _has("tsconfig.json"):
        hints.append("typescript")
    elif _has("package.json"):
        hints.append("javascript")
    if _has("Cargo.toml"):
        hints.append("rust")

    has_cpp = _has("CMakeLists.txt") or _any_ext(".cpp", ".cc", ".cxx")
    if has_cpp:
        hints.append("cpp")
    if not has_cpp and _any_ext(".c"):
        hints.append("c")

    if _has("Gemfile"):
        hints.append("ruby")

    # Containers
    if _has("docker-compose.yml") or _has("docker-compose.yaml") or _has("Dockerfile"):
        hints.append("docker")

    # Shell
    if _any_ext(".sh"):
        hints.append("bash")

    # Databases / stores (check dependency files)
    deps_files = ("requirements.txt", "package.json", "Cargo.toml")
    if any(_file_contains(f, "redis") for f in deps_files):
        hints.append("redis")
    if any(_file_contains(f, "psycopg", "asyncpg", "postgres") for f in deps_files):
        hints.append("postgres")
    if any(_file_contains(f, "mysql", "pymysql", "mysqlclient") for f in deps_files):
        hints.append("mysql")
