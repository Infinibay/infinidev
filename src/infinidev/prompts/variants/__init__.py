"""Prompt style variant system.

Provides three prompt styles:
- ``full``  — current detailed prompts (DO/DON'T lists, examples, anti-patterns)
- ``generalized`` — condensed prose paragraphs capturing the same essence
- ``coding`` — behavioral rules expressed as pseudocode

The ``auto`` setting (default) selects ``generalized`` for all models.
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

#: What ``"auto"`` resolves to. One constant, because the resolver and the
#: settings dialog's own description of it had already drifted apart: the dialog
#: still advertised ``auto=generalized`` after the default changed to ``lean``.
DEFAULT_STYLE = "lean"

#: The order the picker shows styles in. A style registered but missing from
#: this tuple still appears, sorted at the end, so adding a variant can never
#: leave it unreachable from the UI again — which is exactly what happened to
#: ``lean``: it shipped, was measured, and was selectable only by editing the
#: settings file by hand.
_STYLE_ORDER = ("full", "generalized", "lean", "coding", "extra_simple")


def resolve_style() -> str:
    """Return the effective prompt style.

    Reads ``settings.PROMPT_STYLE``.  ``"auto"`` resolves to
    :data:`DEFAULT_STYLE`.  Set any other registered style explicitly to opt
    out; see :func:`registered_styles`.
    """
    from infinidev.config.settings import settings

    style = getattr(settings, "PROMPT_STYLE", "auto")
    if style != "auto":
        return style

    return DEFAULT_STYLE


def registered_styles() -> list[str]:
    """Return every style that registered a variant, in picker order.

    Derived from the registry rather than typed out: the literal that this
    replaces listed four styles and silently omitted the fifth.
    """
    found = {style for (style, _name) in _REGISTRY}
    ordered = [style for style in _STYLE_ORDER if style in found]
    ordered.extend(sorted(found - set(_STYLE_ORDER)))
    return ordered


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
    try:
        from infinidev.prompts.variants import extra_simple as _es  # noqa: F401
    except Exception as exc:
        logger.debug("Failed to load extra_simple variants: %s", exc)
    try:
        from infinidev.prompts.variants import lean as _ln  # noqa: F401
    except Exception as exc:
        logger.debug("Failed to load lean variants: %s", exc)


_load_variants()
