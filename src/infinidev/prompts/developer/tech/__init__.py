"""Technology-specific prompt modules for the Developer agent.

Each technology has its own module exposing a ``get_prompt() -> str`` function.
Use ``get_tech_prompt(tech)`` to look up a prompt by technology name or alias.
"""

from __future__ import annotations

import importlib
from typing import Optional

__all__ = ["get_tech_prompt"]

# Maps every accepted alias (lowercased) to its module name within this package.
_REGISTRY: dict[str, str] = {
    # TypeScript
    "typescript": "typescript",
    "ts": "typescript",
    # JavaScript
    "javascript": "javascript",
    "js": "javascript",
    # Python
    "python": "python",
    "py": "python",
    # Rust
    "rust": "rust",
    "rs": "rust",
    # C
    "c": "c",
    # C++
    "c++": "cpp",
    "cpp": "cpp",
    # Ruby
    "ruby": "ruby",
    "rb": "ruby",
    # SQL (general)
    "sql": "sql_general",
    "sql_general": "sql_general",
    # PostgreSQL
    "postgres": "sql_postgres",
    "postgresql": "sql_postgres",
    "sql_postgres": "sql_postgres",
    # MySQL
    "mysql": "sql_mysql",
    "sql_mysql": "sql_mysql",
    # Redis
    "redis": "redis",
    # Bash / Shell
    "bash": "bash",
    "sh": "bash",
    "shell": "bash",
    # Docker
    "docker": "docker",
    # Podman
    "podman": "podman",
}


def get_tech_prompt(tech: str) -> Optional[str]:
    """Return the technology-specific prompt for *tech*, or ``None`` if unknown.

    The lookup is case-insensitive and strips surrounding whitespace.
    """
    module_name = _REGISTRY.get(tech.lower().strip())
    if module_name is None:
        return None
    module = importlib.import_module(f".{module_name}", package=__name__)
    return module.get_prompt()
