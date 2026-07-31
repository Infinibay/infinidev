"""The model catalog the ChatGPT subscription actually serves.

The subscription and the metered API expose the same model *names* with
different limits.  LiteLLM's cost map — correct for ``api.openai.com`` —
says ``gpt-5.5`` takes 1 050 000 input tokens; the Codex backend serves that
same model with a 272 000-token window.  Trusting the API number would make
the TUI's "context left" indicator claim ~800 000 tokens of headroom that do
not exist, and the loop would keep packing context until the backend
silently truncated it.

So the subscription gets its own source of truth: the catalog the Codex CLI
downloads and caches at ``~/.codex/models_cache.json``.  It is read, never
written — the CLI owns its refresh cycle, and piggybacking on it means the
list stays current without Infinidev inventing an undocumented endpoint to
poll.  When the file is missing (Codex installed but never run, or a
different machine) a small static fallback keeps the provider usable.

The catalog also carries ``effective_context_window_percent``, the share of
the window the client is expected to leave for input.  That is the number
the context indicator wants: the remaining ~5 % is the model's own output
budget, and counting it as available input is how a prompt ends up
overflowing at the last moment.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Used when the CLI's cache is unavailable.  Deliberately short: it exists so
# the provider still works, not to mirror a catalog that changes upstream.
# 272k is the window every listed Codex model shipped with at the time of
# writing; the real numbers arrive with the cache.
_FALLBACK_CONTEXT = 272_000
# Every catalog entry observed so far reserves the same 5 % for output;
# applying it here keeps the fallback on the same footing as the real thing
# instead of quietly reporting a slightly roomier window.
_FALLBACK_EFFECTIVE_PCT = 95
# Slugs, not families: the backend rejects ``gpt-5.6`` outright, because the
# 5.6 generation ships as three named variants. Anything written here is
# offered in a *closed* dropdown, so a slug that does not exist is not a
# harmless guess — it is a choice the product asserts will work.
_FALLBACK_MODELS: tuple[str, ...] = (
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.4",
    "gpt-5.4-mini",
)

# Parsed catalog, keyed by the cache file's mtime so an update by the CLI is
# picked up without a restart and an unchanged file costs one stat().
_cache: tuple[float, dict[str, dict[str, Any]]] | None = None


def catalog_path() -> Path:
    from infinidev.config.openai_oauth import codex_home

    return codex_home() / "models_cache.json"


def _load() -> dict[str, dict[str, Any]]:
    """The catalog as ``{slug: entry}``.  Empty when unavailable."""
    global _cache

    path = catalog_path()
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return {}

    if _cache is not None and _cache[0] == mtime:
        return _cache[1]

    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Codex model catalog unreadable (%s): %s", path, exc)
        return {}

    entries: dict[str, dict[str, Any]] = {}
    for entry in raw.get("models") or []:
        slug = entry.get("slug")
        if isinstance(slug, str) and slug:
            entries[slug] = entry

    _cache = (mtime, entries)
    return entries


# `priority` ranks *ascending* — it is a position, not a score. The catalog
# ships gpt-5.5 at 9 and gpt-5.2 at 29, so sorting it the intuitive way
# (highest first) puts the weakest model at the top of the picker.
_NO_PRIORITY = 1_000_000


def _priority(entry: dict[str, Any]) -> int:
    """Sort position for a catalog entry; unranked entries go last."""
    value = entry.get("priority")
    return int(value) if isinstance(value, int) else _NO_PRIORITY


def list_models() -> list[str]:
    """Selectable model slugs, best first.

    ``visibility != "list"`` hides the entries the CLI itself keeps out of
    the picker — internal review models and the like, which are reachable
    but not something to offer as a choice.

    A loaded catalog is the whole answer.  The fallback used to be *added*
    to it, on the reasoning that a cache nobody had refreshed in weeks would
    miss models the backend already served.  That trade only pays if the
    hardcoded tuple is fresher than the cache, and it never is: the CLI
    refreshes itself, while this constant ages with the release cycle.  What
    the union actually produced was the reverse — the catalog carried the
    three real 5.6 slugs and the fallback contributed ``gpt-5.6``, which the
    backend rejects with "not supported when using codex with ChatGPT
    account".  A settings dropdown that is *closed* turns every such entry
    into a selectable trap.

    So the fallback is what it says it is: what to offer when there is no
    catalog at all.
    """
    entries = _load()
    if not entries:
        return list(_FALLBACK_MODELS)

    listed = [e for e in entries.values() if e.get("visibility", "list") == "list"]
    listed.sort(key=lambda e: (_priority(e), str(e.get("slug"))))
    return [str(e["slug"]) for e in listed]


def display_name(slug: str) -> str:
    entry = _load().get(slug) or {}
    return str(entry.get("display_name") or slug)


def context_window(slug: str) -> int | None:
    """Usable *input* tokens for ``slug`` on the subscription.

    Returns the catalog window scaled by ``effective_context_window_percent``
    — what the client may fill — rather than the raw window, which the
    model's own output has to fit inside too.
    """
    entries = _load()
    entry = entries.get(slug)
    if entry is None:
        # An unknown slug is most often a model *newer* than the cache, not a
        # typo: the CLI only refreshes the catalog when it runs, so a machine
        # that has not opened `codex` in weeks is several releases behind
        # while the backend already serves the new model happily.
        #
        # Returning None there would be the pedantic answer and the worse
        # one — an unknown window disables the loop's context budget and
        # shows "?" in the status line for a model that works. Every entry
        # the catalog has ever carried is 272k, so assume it and say so.
        if entries:
            logger.info(
                "Model %r is not in the Codex catalog at %s (cache may predate "
                "it); assuming the standard %d-token window. Run `codex` once "
                "to refresh the catalog.",
                slug,
                catalog_path(),
                _FALLBACK_CONTEXT,
            )
        return int(_FALLBACK_CONTEXT * _FALLBACK_EFFECTIVE_PCT / 100)

    window = entry.get("context_window")
    if not isinstance(window, int) or window <= 0:
        return None

    pct = entry.get("effective_context_window_percent")
    if isinstance(pct, (int, float)) and 0 < pct <= 100:
        return int(window * pct / 100)
    return window


def supports_parallel_tool_calls(slug: str) -> bool:
    entry = _load().get(slug) or {}
    return bool(entry.get("supports_parallel_tool_calls", True))


def reasoning_levels(slug: str) -> list[str]:
    """Reasoning efforts the model accepts, e.g. ``["low","medium","high"]``.

    Empty when unknown — callers should then send no ``reasoning`` at all
    rather than guess a level the model will reject.
    """
    entry = _load().get(slug) or {}
    levels = entry.get("supported_reasoning_levels") or []
    out: list[str] = []
    for level in levels:
        if isinstance(level, dict) and isinstance(level.get("effort"), str):
            out.append(level["effort"])
        elif isinstance(level, str):
            out.append(level)
    return out
