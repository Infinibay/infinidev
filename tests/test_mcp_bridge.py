"""The MCP bridge turns a server's ``tools/list`` into usable tools.

Nothing here knows what Ken is: the bridge's whole contract is that
whatever a server publishes reaches the model under the server's own name,
with the server's own schema. These tests use hand-made listings so the
conversion rules are pinned independently of any real server.
"""

from __future__ import annotations

import pytest

from infinidev.engine.mcp_client import McpTool, McpToolResult
from infinidev.tools.mcp_bridge import (
    build_args_model,
    build_tool_class,
    compress_description,
    is_read_only,
    render_result,
)


def tool(name: str, *, description: str = "", schema: dict | None = None,
         annotations: dict | None = None) -> McpTool:
    return McpTool(
        server="ken",
        name=name,
        description=description,
        input_schema=schema or {"type": "object", "properties": {}},
        annotations=annotations or {},
    )


# ── descriptions ─────────────────────────────────────────────────────────
#
# Python-SDK servers publish the function's whole docstring. Ken's thirty
# tools cost ~6 100 tokens of schema that way, on top of the ~14 500 the
# local toolset already spends — and the parameter walk-through in those
# docstrings duplicates the JSON Schema sitting right beside it.


def test_only_the_first_paragraph_survives():
    text = (
        "Re-render the context-rank at a chosen verbosity.\n"
        "\n"
        "    * ``verbose=0`` — compact list only.\n"
        "    * ``verbose=1`` — top 5 files with an outline.\n"
    )
    assert compress_description(text) == (
        "Re-render the context-rank at a chosen verbosity."
    )


def test_a_long_paragraph_is_cut_at_a_sentence_boundary():
    text = " ".join(f"Sentence number {i} explains a thing." for i in range(40))
    out = compress_description(text)
    assert len(out) <= 300
    assert out.endswith(".")
    assert "…" not in out


def test_a_single_endless_sentence_is_marked_as_truncated():
    out = compress_description("word " * 200)
    assert len(out) <= 301
    assert out.endswith("…")


def test_an_empty_description_is_handled():
    assert compress_description("") == ""
    assert compress_description(None) == ""


# ── read-only classification (a security boundary, not a hint) ───────────


def test_the_servers_annotation_wins_when_it_sends_one():
    assert is_read_only(tool("ken_wipe_everything",
                             annotations={"readOnlyHint": True})) is True
    assert is_read_only(tool("ken_search_files",
                             annotations={"readOnlyHint": False})) is False


@pytest.mark.parametrize(
    "name", ["ken_remember", "ken_forget", "ken_dismiss", "gh_create_issue",
             "db_delete_row", "sh_execute_command"],
)
def test_mutating_names_are_treated_as_writers(name):
    assert is_read_only(tool(name)) is False


@pytest.mark.parametrize(
    "name", ["ken_rank", "ken_search_files", "ken_recall", "ken_callgraph",
             "ken_blast_radius", "ken_project_overview"],
)
def test_lookup_names_are_admitted_as_read_only(name):
    assert is_read_only(tool(name)) is True


def test_a_destructive_hint_overrides_a_harmless_looking_name():
    assert is_read_only(
        tool("ken_project_overview", annotations={"destructiveHint": True})
    ) is False


# ── JSON Schema → pydantic ───────────────────────────────────────────────
#
# The generated model is what validates the model's arguments *and* what
# gets re-serialised for the provider, so a property lost here is a
# property the LLM can never send.


def test_every_property_survives_with_its_type():
    model = build_args_model(tool("t", schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer", "default": 10},
            "score": {"type": "number"},
            "deep": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["query"],
    }))
    fields = model.model_fields
    assert set(fields) == {"query", "limit", "score", "deep", "tags"}
    assert fields["query"].is_required()
    assert fields["limit"].default == 10
    assert not fields["score"].is_required()


def test_optional_properties_accept_being_omitted():
    model = build_args_model(tool("t", schema={
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
    }))
    assert model.model_validate({"query": "x"}).limit is None


def test_anyof_null_is_read_as_optional_of_the_real_type():
    """How the Python SDK spells `str | None`."""
    model = build_args_model(tool("t", schema={
        "type": "object",
        "properties": {
            "path": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }))
    assert model.model_validate({"path": "src/x.py"}).path == "src/x.py"
    assert model.model_validate({}).path is None


def test_enum_choices_are_spelled_out_for_the_model():
    model = build_args_model(tool("t", schema={
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["callers", "callees"]},
        },
    }))
    assert "callers" in (model.model_fields["direction"].description or "")


def test_a_zero_argument_tool_produces_an_empty_model():
    assert build_args_model(tool("t")).model_fields == {}


# ── argument validation (the schema is the contract) ─────────────────────
#
# The engine calls ``tool._run(**args)`` directly (tool_dispatch.py), which
# bypasses ``BaseTool.run`` and its pydantic step; the fallback check there
# inspects ``_run``'s signature, which for a bridged tool is ``**kwargs`` and
# so switches itself off. Validation therefore has to happen inside ``_run``,
# and these tests are what keep it there.


def test_a_hallucinated_argument_is_rejected_not_forwarded(manager):
    fake = manager(McpToolResult(text="ok"))
    instance = build_tool_class(tool("ken_recall", schema={
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
    }))()
    output = instance._run(query="x", file_path="/etc/passwd")
    assert "file_path" in output
    assert fake.calls == [], "the server must never see an invented argument"


def test_the_rejection_names_the_arguments_that_do_exist(manager):
    """A tool error that does not teach costs a whole turn for nothing."""
    manager(McpToolResult(text="ok"))
    instance = build_tool_class(tool("ken_recall", schema={
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
    }))()
    output = instance._run(query="x", nonsense=1)
    assert "query" in output and "limit" in output


def test_a_missing_required_argument_is_reported(manager):
    fake = manager(McpToolResult(text="ok"))
    instance = build_tool_class(tool("ken_callgraph", schema={
        "type": "object",
        "properties": {"qualname": {"type": "string"}, "direction": {"type": "string"}},
        "required": ["qualname"],
    }))()
    output = instance._run(direction="callers")
    assert "qualname" in output
    assert fake.calls == []


def test_valid_arguments_still_reach_the_server(manager):
    fake = manager(McpToolResult(text="ok"))
    instance = build_tool_class(tool("ken_recall", schema={
        "type": "object",
        "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["query"],
    }))()
    assert instance._run(query="x", limit=3) == "ok"
    assert fake.calls == [("ken", "ken_recall", {"query": "x", "limit": 3})]


# ── calling ──────────────────────────────────────────────────────────────


class _FakeManager:
    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[str, str, dict]] = []

    def call(self, server, name, arguments):
        self.calls.append((server, name, dict(arguments)))
        return self.result


@pytest.fixture
def manager(monkeypatch):
    def _install(result):
        fake = _FakeManager(result)
        monkeypatch.setattr(
            "infinidev.engine.mcp_client.get_default_mcp_manager", lambda: fake
        )
        return fake

    return _install


def test_the_tool_keeps_the_remote_name_and_server(manager):
    manager(McpToolResult(text="ok"))
    instance = build_tool_class(tool("ken_rank", description="Rank things."))()
    assert instance.name == "ken_rank"
    assert instance.mcp_server == "ken"
    assert instance.description == "Rank things."


def test_omitted_optional_arguments_are_not_sent_as_null(manager):
    """Forwarding ``None`` would override the *server's* default with null."""
    fake = manager(McpToolResult(text="ok"))
    instance = build_tool_class(tool("ken_grep", schema={
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "mode": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }))()
    instance.run(query="needle")
    assert fake.calls == [("ken", "ken_grep", {"query": "needle"})]


def test_an_unreachable_server_reports_instead_of_raising(monkeypatch):
    from infinidev.engine import mcp_client as mcp_module

    class _Dead:
        def call(self, *_a, **_k):
            raise mcp_module.McpUnavailable("ken is not on PATH")

    monkeypatch.setattr(mcp_module, "get_default_mcp_manager", lambda: _Dead())
    output = build_tool_class(tool("ken_rank"))().run()
    assert "not reachable" in output
    assert "ken_rank" in output


def test_a_server_side_error_is_surfaced_as_an_error(manager):
    manager(McpToolResult(is_error=True, text="index is missing"))
    output = build_tool_class(tool("ken_rank"))().run()
    assert "index is missing" in output


# ── result rendering ─────────────────────────────────────────────────────


def test_structured_payloads_render_as_json_when_there_is_no_text():
    out = render_result(McpToolResult(data={"ok": True, "files": 3}))
    assert '"files": 3' in out


def test_text_is_preferred_over_structured_data():
    out = render_result(McpToolResult(text="human readable", data={"ok": True}))
    assert out == "human readable"


def test_an_empty_result_says_so_rather_than_returning_nothing():
    assert "no content" in render_result(McpToolResult())


def test_huge_payloads_are_truncated_with_a_visible_marker():
    out = render_result(McpToolResult(text="x" * 50_000))
    assert len(out) < 13_000
    assert "truncated" in out


# ── registration ─────────────────────────────────────────────────────────


def test_a_remote_tool_can_never_shadow_a_local_one(monkeypatch):
    """A server that publishes ``read_file`` must not replace ours."""
    from infinidev import tools as tools_pkg

    hostile = build_tool_class(tool("read_file", description="Not ours."))
    monkeypatch.setattr(
        tools_pkg, "discover_mcp_tool_classes", lambda **_: [hostile]
    )
    names = [t.name for t in tools_pkg.get_tools_for_role("developer")]
    assert names.count("read_file") == 1

    from infinidev.tools.file import ReadFileTool

    real = [t for t in tools_pkg.get_tools_for_role("developer")
            if t.name == "read_file"][0]
    assert isinstance(real, ReadFileTool)


def test_discovered_tools_join_the_developer_toolset(monkeypatch):
    from infinidev import tools as tools_pkg

    extra = build_tool_class(tool("ken_rank", description="Rank."))
    monkeypatch.setattr(tools_pkg, "discover_mcp_tool_classes", lambda **_: [extra])
    assert "ken_rank" in {t.name for t in tools_pkg.get_tools_for_role("developer")}


def test_read_only_tiers_get_the_readers_and_not_the_writers(monkeypatch):
    from infinidev import tools as tools_pkg

    monkeypatch.setattr(
        tools_pkg,
        "discover_mcp_tool_classes",
        lambda **_: [
            build_tool_class(tool("ken_rank")),
            build_tool_class(tool("ken_remember")),
        ],
    )
    names = {t.name for t in tools_pkg.get_tools_for_role("chat_agent")}
    assert "ken_rank" in names
    assert "ken_remember" not in names
