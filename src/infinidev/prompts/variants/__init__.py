"""Prompt style variant system.

Provides three prompt styles:
- ``full``  — current detailed prompts (DO/DON'T lists, examples, anti-patterns)
- ``generalized`` — condensed prose paragraphs capturing the same essence
- ``coding`` — behavioral rules expressed as pseudocode

The ``auto`` setting (default) selects ``generalized`` for small models
and ``full`` for large models.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ── Registry ────────────────────────────────────────────────────────────
# Keyed by (style, prompt_name).  All three styles (full, generalized,
# coding) are registered as first-class variants.

_REGISTRY: dict[tuple[str, str], str] = {}


def register(style: str, name: str, prompt: str) -> None:
    """Register a prompt variant.

    Parameters
    ----------
    style : str
        ``"full"``, ``"generalized"``, or ``"coding"``.
    name : str
        Dot-separated prompt name, e.g. ``"flow.develop.identity"``,
        ``"phase.bug.execute"``, ``"loop.identity"``.
    prompt : str
        The full prompt text for this variant.
    """
    _REGISTRY[(style, name)] = prompt


def get_variant(name: str, style: str | None = None) -> str | None:
    """Return a prompt variant, or *None* if not registered.

    When *style* is ``None`` the effective style is resolved automatically.
    """
    if style is None:
        style = resolve_style()
    return _REGISTRY.get((style, name))


# ── Style resolution ───────────────────────────────────────────────────

def resolve_style() -> str:
    """Return the effective prompt style: ``full``, ``generalized``, or ``coding``.

    Reads ``settings.PROMPT_STYLE``.  When set to ``"auto"`` (the default),
    small models (< 25 B params) get ``generalized`` and everything else
    gets ``full``.
    """
    from infinidev.config.settings import settings

    style = getattr(settings, "PROMPT_STYLE", "auto")
    if style != "auto":
        return style

    try:
        from infinidev.config.llm import _is_small_model
        return "generalized" if _is_small_model() else "full"
    except Exception:
        return "full"


def registered_names(style: str) -> set[str]:
    """Return all prompt names registered for *style*."""
    return {name for (s, name) in _REGISTRY if s == style}


# ── Auto-import variant modules so they self-register ──────────────────

def _load_variants() -> None:
    """Import variant modules to trigger their register() calls."""
    try:
        from infinidev.prompts.variants import full as _f  # noqa: F401
    except Exception as exc:
        logger.debug("Failed to load full variants: %s", exc)
    try:
        from infinidev.prompts.variants import generalized as _g  # noqa: F401
    except Exception as exc:
        logger.debug("Failed to load generalized variants: %s", exc)
    try:
        from infinidev.prompts.variants import coding as _c  # noqa: F401
    except Exception as exc:
        logger.debug("Failed to load coding variants: %s", exc)


_load_variants()
