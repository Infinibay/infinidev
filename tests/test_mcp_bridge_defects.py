"""Four ways the MCP bridge quietly failed the model.

None of these raised. Each one degraded a capability while looking like it
worked, which is the failure mode that survives a green test suite: a writer
running in a parallel read batch, a `help` call that denies a tool the model
can plainly see in its toolbox, an "unknown tool" error that suggests
alphabetically, and an agent that spends its whole life without the project
index because it was built a second before the server answered.
"""

from __future__ import annotations

import pytest

from infinidev.engine.mcp_client import McpTool
from infinidev.tools.mcp_bridge import build_tool_class


def _remote(name: str, *, read_only: bool | None = None) -> McpTool:
    annotations = {} if read_only is None else {"readOnlyHint": read_only}
    return McpTool(
        server="ken",
        name=name,
        description=f"{name} does a thing.",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        annotations=annotations,
    )


@pytest.fixture(autouse=True)
def _clean_cache():
    from infinidev.tools.mcp_bridge import reset_discovery_cache

    reset_discovery_cache()
    yield
    reset_discovery_cache()


# ── a writer must never share a parallel batch with readers ──────────────


def test_discovered_writers_are_registered_as_write_tools(monkeypatch):
    """``batch_tool_calls`` decides serial-vs-parallel from a NAME set, and
    it only ever sees the call — never the tool. A discovered writer missing
    from that set gets batched in with the reads, so ``ken_remember`` would
    mutate the index concurrently with lookups reading it."""
    from infinidev.engine import tool_executor
    from infinidev.tools import mcp_bridge

    monkeypatch.setattr(tool_executor, "WRITE_TOOLS", set(tool_executor.WRITE_TOOLS))
    mcp_bridge._register_writers([
        build_tool_class(_remote("ken_recall", read_only=True)),
        build_tool_class(_remote("ken_remember", read_only=False)),
    ])

    assert "ken_remember" in tool_executor.WRITE_TOOLS
    assert "ken_recall" not in tool_executor.WRITE_TOOLS


def test_a_registered_writer_gets_its_own_batch(monkeypatch):
    from infinidev.engine import tool_executor
    from infinidev.tools import mcp_bridge

    monkeypatch.setattr(tool_executor, "WRITE_TOOLS", set(tool_executor.WRITE_TOOLS))
    mcp_bridge._register_writers([build_tool_class(_remote("ken_remember", read_only=False))])

    def call(name):
        return type("TC", (), {"function": type("F", (), {"name": name})()})()

    batches = tool_executor.batch_tool_calls(
        [call("ken_recall"), call("ken_remember"), call("ken_recall")]
    )
    assert [[c.function.name for c in b] for b in batches] == [
        ["ken_recall"], ["ken_remember"], ["ken_recall"],
    ]


# ── help must not deny a tool the model can see ──────────────────────────


def test_help_renders_a_discovered_tool_from_its_own_schema(monkeypatch):
    from infinidev.tools.meta import help_tool
    from infinidev.tools.meta.help_tool import HelpTool

    monkeypatch.setattr(
        help_tool, "discover_mcp_tool_classes",
        lambda **_: [build_tool_class(_remote("ken_rank"))],
        raising=False,
    )
    monkeypatch.setattr(
        "infinidev.tools.mcp_bridge.discover_mcp_tool_classes",
        lambda **_: [build_tool_class(_remote("ken_rank"))],
    )
    out = HelpTool()._run(context="ken_rank")
    assert "ken_rank" in out
    assert "does a thing" in out
    assert "query" in out          # the parameter is documented
    assert "ken" in out            # provenance: which server it came from
    assert "No help found" not in out


def test_help_still_reports_genuinely_unknown_topics(monkeypatch):
    from infinidev.tools.meta.help_tool import HelpTool

    monkeypatch.setattr(
        "infinidev.tools.mcp_bridge.discover_mcp_tool_classes", lambda **_: []
    )
    assert "No help found" in HelpTool()._run(context="zzz-not-a-tool")


# ── an unknown tool must suggest plausible ones, not alphabetical ones ───


def test_unknown_tool_suggests_by_similarity_not_alphabetically():
    """The old message answered with ``sorted(dispatch)[:15]`` — for this
    toolset, everything from add_content_after_line to delete_report, and
    never the tool that was wanted."""
    from infinidev.engine.tool_dispatch import _unknown_tool_message

    dispatch = dict.fromkeys([
        "add_content_after_line", "analyze_code", "code_search", "create_file",
        "delete_finding", "delete_report", "git_commit", "read_file",
        "ken_search_files", "ken_search_symbols", "ken_recall", "ken_rank",
    ])
    message = _unknown_tool_message(dispatch, "ken_search_file")
    assert "ken_search_files" in message
    assert "add_content_after_line" not in message


def test_unknown_tool_message_points_at_help():
    from infinidev.engine.tool_dispatch import _unknown_tool_message

    assert "describe_tool()" in _unknown_tool_message(
        {"read_file": None, "describe_tool": None}, "raed_file"
    )


def test_unknown_tool_does_not_recommend_unavailable_help():
    from infinidev.engine.tool_dispatch import _unknown_tool_message

    message = _unknown_tool_message({"read_file": None}, "raed_file")

    assert "describe_tool()" not in message
    assert "advertised for this turn" in message


# ── an agent built before warmup must not stay blind for its whole life ──


def test_agent_picks_up_mcp_tools_once_discovery_lands(monkeypatch):
    """Servers warm on a background thread. An agent constructed in the first
    moments of a session resolves its tools before any server has answered,
    and used to keep that empty result until the process exited."""
    from infinidev.agents import base as agent_base

    discovered: list = []
    monkeypatch.setattr(
        "infinidev.tools.mcp_bridge.discover_mcp_tool_classes",
        lambda **_: list(discovered),
    )

    built: list[int] = []

    def fake_get_tools_for_role(role, **kwargs):
        built.append(len(discovered))
        return [cls() for cls in discovered]

    monkeypatch.setattr(
        "infinidev.tools.get_tools_for_role", fake_get_tools_for_role
    )

    agent = agent_base.InfinidevAgent(role="developer", agent_id="a1")
    assert agent.tools == []

    # …the warmup thread finishes.
    discovered.append(build_tool_class(_remote("ken_rank")))

    assert [t.name for t in agent.tools] == ["ken_rank"]
    assert len(built) == 2, "the toolset must rebuild exactly once, not per access"

    agent.tools
    assert len(built) == 2
