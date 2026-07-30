"""Regression guard: every developer tool must be documented.

The static tool catalog in ``prompts/tool_hints.py`` (TOOL_DESCRIPTIONS) is
hand-written, so it silently goes stale every time a new tool is registered
without a matching entry. This test fails loudly when that happens, naming
the offending tools — add them to TOOL_DESCRIPTIONS (and the relevant
category in ``build_tool_usage_section``) to fix it.
"""

from infinidev.tools import get_tools_for_role
from infinidev.prompts.tool_hints import (
    MCP_TOOL_HINTS,
    TOOL_DESCRIPTIONS,
    build_tool_usage_section,
    get_available_tool_names,
)

# Engine pseudo-tools and backward-compat aliases that are intentionally not
# real registered tool classes but ARE valid catalog entries.
_PSEUDO_TOOLS = {"step_complete", "add_note", "add_session_note", "think"}


def _local_tools():
    """Tools defined in this repo — everything except the MCP bridge.

    ``supports_vision=True`` on purpose: this guards the *catalog*, not the
    runtime gating, and whether ``view_image`` is registered depends on the
    configured model. Letting that decide would make the guard pass or fail
    according to whoever's ``settings.json`` is on the machine.

    Tools discovered from an MCP server carry the server's own name and
    description, so the hand-written catalog cannot be the source of truth
    for them and this guard does not apply to them.
    """
    return [
        t for t in get_tools_for_role("developer", supports_vision=True)
        if getattr(t, "mcp_server", None) is None
    ]


def test_every_developer_tool_has_a_description():
    names = {t.name for t in _local_tools()}
    missing = sorted(n for n in names if n not in TOOL_DESCRIPTIONS)
    assert not missing, (
        "These registered developer tools are missing from "
        f"TOOL_DESCRIPTIONS in prompts/tool_hints.py: {missing}. "
        "Add a (description, example) entry for each."
    )


def test_every_mcp_tool_arrives_with_a_description():
    """The bridge must never register a tool the model cannot understand."""
    bridged = [
        t for t in get_tools_for_role("developer")
        if getattr(t, "mcp_server", None) is not None
    ]
    blank = sorted(t.name for t in bridged if not (t.description or "").strip())
    assert not blank, f"MCP tools registered with no description: {blank}"


def test_no_stale_catalog_entries():
    """Every catalog entry maps to a real tool or a known pseudo-tool."""
    valid = {t.name for t in _local_tools()} | _PSEUDO_TOOLS
    # Retired names still resolve through the dispatcher, so a hint carrying
    # one is documentation, not rot. Read the alias table rather than listing
    # them here — a hand-kept copy is what goes stale.
    from infinidev.engine.tool_dispatch import _TOOL_ALIASES

    valid |= set(_TOOL_ALIASES)
    # MCP hints describe tools that only exist when their server is
    # configured, so their absence is a deployment fact, not a stale entry.
    valid |= set(MCP_TOOL_HINTS)
    stale = sorted(k for k in TOOL_DESCRIPTIONS if k not in valid)
    assert not stale, (
        f"TOOL_DESCRIPTIONS has entries with no matching tool: {stale}. "
        "Remove them or fix the name."
    )


def test_usage_section_lists_every_available_tool():
    """build_tool_usage_section must place every available tool in a category."""
    tools = get_tools_for_role("developer")
    available = get_available_tool_names(tools)
    section = build_tool_usage_section(available)
    # Pseudo-tools are documented elsewhere (protocol section), so only the
    # real registered tools must surface in the usage section.
    real = {t.name for t in tools}
    not_shown = sorted(n for n in real if f"**{n}**" not in section)
    assert not not_shown, (
        "These tools are documented in TOOL_DESCRIPTIONS but never appear in "
        f"build_tool_usage_section (missing from its category list): {not_shown}."
    )


def test_mcp_hints_match_what_the_server_actually_publishes():
    """The hole the guard above deliberately leaves.

    ``test_no_stale_catalog_entries`` exempts MCP names, because a hint for
    a server nobody configured is a deployment fact rather than rot. That
    exemption also hides a rename: when ken collapsed 30 tools into 6, seven
    hints kept teaching the model names the server no longer published, and
    nothing failed.

    So ask the server. Skipped when it is not installed — a machine without
    ken cannot answer the question, and guessing is what got us here.
    """
    import shutil

    import pytest

    if not shutil.which("ken"):
        pytest.skip("ken is not installed; nothing to compare the hints against")

    from infinidev.tools.mcp_bridge import (
        discover_mcp_tool_classes,
        reset_discovery_cache,
    )

    reset_discovery_cache()
    try:
        published = {cls.model_fields["name"].default for cls in
                     discover_mcp_tool_classes(force=True, block=True)}
    finally:
        reset_discovery_cache()

    if not published:
        pytest.skip("no MCP server answered; cannot verify the hints")

    stale = sorted(name for name in MCP_TOOL_HINTS if name not in published)
    assert not stale, (
        f"MCP_TOOL_HINTS names tools the server no longer publishes: {stale}. "
        f"The live surface is {sorted(published)}."
    )
