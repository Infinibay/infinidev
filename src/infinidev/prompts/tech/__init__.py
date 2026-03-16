"""Technology-specific prompt modules for Infinidev.

Each technology has its own module exposing a ``get_prompt() -> str`` function.
Use ``get_tech_prompt(tech)`` to look up a prompt by technology name or alias.
"""

from __future__ import annotations

import importlib
from typing import Optional

__all__ = ["get_tech_prompt"]

# Maps accepted aliases (lowercased) to module names within this package.
_REGISTRY: dict[str, str] = {
    "typescript": "typescript",
    "ts": "typescript",
    "javascript": "javascript",
    "js": "javascript",
    "python": "python",
    "py": "python",
    "rust": "rust",
    "rs": "rust",
}


def get_tech_prompt(tech: str) -> Optional[str]:
    """Return the technology-specific prompt for *tech*, or ``None`` if unknown."""
    module_name = _REGISTRY.get(tech.lower().strip())
    if module_name is None:
        return None
    module = importlib.import_module(f".{module_name}", package=__name__)
    return module.get_prompt()
