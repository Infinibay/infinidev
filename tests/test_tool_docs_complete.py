"""Regression guard: every developer tool must be documented.

The static tool catalog in ``prompts/tool_hints.py`` (TOOL_DESCRIPTIONS) is
hand-written, so it silently goes stale every time a new tool is registered
without a matching entry. This test fails loudly when that happens, naming
the offending tools — add them to TOOL_DESCRIPTIONS (and the relevant
category in ``build_tool_usage_section``) to fix it.
"""

from infinidev.tools import get_tools_for_role
from infinidev.prompts.tool_hints import (
    TOOL_DESCRIPTIONS,
    build_tool_usage_section,
    get_available_tool_names,
)

# Engine pseudo-tools and backward-compat aliases that are intentionally not
# real registered tool classes but ARE valid catalog entries.
_PSEUDO_TOOLS = {"step_complete", "add_note", "add_session_note", "think"}


def test_every_developer_tool_has_a_description():
    tools = get_tools_for_role("developer")
    names = {t.name for t in tools}
    missing = sorted(n for n in names if n not in TOOL_DESCRIPTIONS)
    assert not missing, (
        "These registered developer tools are missing from "
        f"TOOL_DESCRIPTIONS in prompts/tool_hints.py: {missing}. "
        "Add a (description, example) entry for each."
    )


def test_no_stale_catalog_entries():
    """Every catalog entry maps to a real tool or a known pseudo-tool."""
    tools = get_tools_for_role("developer")
    valid = {t.name for t in tools} | _PSEUDO_TOOLS
    # find_definition is a backward-compat alias kept for prompt continuity.
    valid.add("find_definition")
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
