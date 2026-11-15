"""Source-derived modern and legacy Ken search transport, without a live server."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from infinidev.engine.ken_client import KenClient
from infinidev.engine.mcp_client import McpHit
from infinidev.tools.code_intel.search_symbols_tool import SearchSymbolsTool


FIXTURE = json.loads(
    (Path(__file__).parent / "fixtures/ken_code_search_contract.json").read_text()
)
CASES = {case["id"]: case["payload"] for case in FIXTURE["cases"]}


class SearchManager:
    def __init__(self, payload):
        self.payload = deepcopy(payload)

    def all_names(self):
        return ["ken"]

    def list_tools(self, server=None):
        return []

    def try_call(self, server, tool, arguments):
        return None if self.payload is None else SimpleNamespace(data=self.payload)


@pytest.mark.parametrize("case", [
    "modern-owner", "modern-piece-symbol", "modern-piece-file-null-qualname",
    "modern-partial-disjoint", "modern-stale", "modern-extension-preservation",
])
@pytest.mark.parametrize("method", ["search_files", "search_symbols"])
def test_modern_results_keep_evidence_identity_and_complete_authority(case, method):
    payload = CASES[case]
    manager = SearchManager(payload)
    hits = getattr(KenClient(manager), method)("measurement")
    assert len(hits) == len(payload["results"])
    row = payload["results"][0]
    hit = hits[0]
    assert hit.metadata == {
        "result": row, "response": {k: v for k, v in payload.items() if k != "results"},
    }
    assert hits.response == hit.metadata["response"]
    assert hit.score == row.get("ranking_score")
    assert hit.line == row["entity_span"]["start_line"]
    for evidence in row["evidence"]:
        assert evidence["content"] in hit.snippet
        assert f"{evidence['start_line']}-{evidence['end_line']}" in hit.snippet
    if method == "search_symbols":
        assert hit.qualname == (row["entity"]["qualname"] or row["path"])
        assert hit.kind == row["entity"]["kind"]
        assert hit.file == row["path"]
    else:
        assert hit.target == row["path"]
    rendered = hit.render()
    assert row["source_hash"] in rendered
    assert '"stale": ' + str(row["stale"]).lower() in rendered
    assert '"truncated": ' + str(row["truncated"]).lower() in rendered
    # Returned mutable structures must not mutate the transport's source record.
    hit.metadata["result"]["entity"]["qualname"] = "changed"
    hits.response["query"] = "changed"
    assert manager.payload == payload


@pytest.mark.parametrize("case", [
    "modern-empty", "modern-budget-empty", "modern-preparation-required",
    "evidence-index-required",
])
@pytest.mark.parametrize("method", ["search_files", "search_symbols"])
def test_valid_empty_and_preparation_errors_keep_response_without_fallback(case, method, monkeypatch):
    monkeypatch.setattr(
        "infinidev.engine.ken_client.keyword_search",
        lambda *a, **kw: pytest.fail("A declared response must not become local matches"),
    )
    payload = CASES[case]
    hits = getattr(KenClient(SearchManager(payload)), method)("measurement")
    assert hits == []
    assert hits.response == {k: v for k, v in payload.items() if k != "results"}
    assert json.dumps(hits.response, ensure_ascii=False) in hits.render_diagnostics()


@pytest.mark.parametrize("case, method", [
    ("legacy-files", "search_files"), ("legacy-symbols", "search_symbols"),
])
def test_legacy_results_retain_existing_meaning(case, method):
    hits = getattr(KenClient(SearchManager(CASES[case])), method)("measurement")
    assert hits[0].score == 0.5
    if method == "search_files":
        assert hits[0].target == "source.py"
        assert "retain_measurement" in hits[0].snippet
    else:
        assert hits[0].qualname == "retain_measurement"
        assert hits[0].line == 1
        assert hits[0].docstring == "Retain observed CPU."


def test_unavailable_server_file_fallback_is_explicit(monkeypatch):
    monkeypatch.setattr(
        "infinidev.engine.ken_client.keyword_search",
        lambda *a, **kw: [McpHit("local.py", "local excerpt", 1.0, "fallback")],
    )
    hits = KenClient(SearchManager(None)).search_files("measurement")
    assert hits[0].source == "fallback"
    assert hits.response["source"] == "fallback"
    assert "unavailable" in hits.render_diagnostics().lower()


@pytest.mark.parametrize("case", [
    "modern-piece-symbol", "modern-partial-disjoint", "modern-stale",
    "modern-budget-empty", "modern-preparation-required",
])
def test_semantic_tool_delivers_public_evidence_and_diagnostics(case, monkeypatch):
    client = KenClient(SearchManager(CASES[case]))
    monkeypatch.setattr("infinidev.engine.ken_client.get_ken_client", lambda: client)
    rendered = SearchSymbolsTool._semantic_fallback("measurement")
    assert rendered
    payload = CASES[case]
    if payload.get("ok") is False:
        assert json.loads(rendered)["error"] == payload["error"]
        assert payload["code"] in rendered
        assert payload["next_action"] in rendered
        assert "semantically related symbol(s)" not in rendered
    else:
        assert payload["profile_version"] in rendered
        assert json.dumps({k: v for k, v in payload.items() if k != "results"},
                          ensure_ascii=False) in rendered
    for row in payload["results"]:
        assert row["source_hash"] in rendered
        for evidence in row["evidence"]:
            assert evidence["content"] in rendered


@pytest.mark.parametrize("case, method", [
    ("legacy-files", "search_files"), ("legacy-symbols", "search_symbols"),
])
def test_single_legacy_record_remains_supported(case, method):
    hits = getattr(KenClient(SearchManager(CASES[case][0])), method)("measurement")
    assert len(hits) == 1
    assert hits[0].score == 0.5


def test_server_order_and_slice_diagnostics_are_preserved_without_inventing_scores():
    payload = deepcopy(CASES["modern-owner"])
    second = deepcopy(payload["results"][0])
    second.pop("ranking_score")
    second["entity"]["qualname"] = "second"
    payload["results"].append(second)
    hits = KenClient(SearchManager(payload)).search_symbols("measurement")
    assert [hit.qualname for hit in hits] == ["retain_measurement", "second"]
    assert hits[1].score is None
    subset = hits[:1]
    assert subset.response == hits.response
    subset.response["query"] = "changed"
    assert hits.response["query"] != "changed"


def test_kind_filter_does_not_discard_response_diagnostics(monkeypatch):
    client = KenClient(SearchManager(CASES["modern-partial-disjoint"]))
    monkeypatch.setattr("infinidev.engine.ken_client.get_ken_client", lambda: client)
    rendered = SearchSymbolsTool._semantic_fallback("measurement", kind="class")
    response = {k: v for k, v in CASES["modern-partial-disjoint"].items() if k != "results"}
    assert json.dumps(response, ensure_ascii=False) in rendered
    assert "semantically related symbol(s)" not in rendered
