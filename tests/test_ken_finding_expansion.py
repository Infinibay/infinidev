"""Bounded session finding expansion using only a fake CLI response."""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from copy import deepcopy
from types import SimpleNamespace

import pytest

from infinidev.engine.ken_session import KenSession


@pytest.fixture
def expand(tmp_path, monkeypatch):
    (tmp_path / ".ken").mkdir()
    (tmp_path / ".ken/meta.json").write_text('{}')
    session = KenSession(tmp_path, "offline-findings")
    monkeypatch.setattr("infinidev.engine.ken_session.shutil.which", lambda _: "/fake/ken")

    def render(rows, brief=None):
        original = deepcopy(rows)
        def run(command, **kwargs):
            assert command == ["/fake/ken", "findings", "--path", str(session._root),
                               "--json", "-n", "3"]
            return SimpleNamespace(returncode=0, stdout=json.dumps(rows))
        monkeypatch.setattr("infinidev.engine.ken_session.subprocess.run", run)
        if brief is None:
            brief = "<ken-session-brief>" + ";".join(
                row.get("topic", "") for row in rows if isinstance(row, dict)
            ) + "</ken-session-brief>"
        rendered = session._expanded_recent_findings(brief)
        assert rows == original
        return rendered
    render.root = tmp_path
    render.session = session
    return render


def decode(rendered):
    root = ET.fromstring(rendered)
    return [(finding.attrib["topic"], json.loads(finding.text))
            for finding in root.findall("finding")]


@pytest.mark.parametrize("state", ["unchanged", "stale", "unknown", "untracked"])
def test_complete_metadata_and_content_preserved_without_promoting_validity(expand, state):
    row = {"topic": "CPU isolation", "content": "Only verified with ABI v2.",
           "created_at": "2026-11-14", "type": "finding",
           "justification": {"kind": "hypothesis", "assumptions": ["ABI v2"],
                             "evidence": [{"path": "source.rs", "note": "focal gate"}],
                             "recheck": "Recheck caller after changes.", "extension": [1]},
           "validity": {"state": state, "issues": ["caller.rs"]}}
    rendered = expand([row])
    topic, payload = decode(rendered)[0]
    assert topic == row["topic"]
    assert payload["content"] == row["content"]
    assert payload["metadata"] == {k: v for k, v in row.items() if k not in {"topic", "content"}}
    assert payload["validity_status"] == "provided"
    assert payload["content_omitted"] is False
    assert payload["metadata_omitted"] is False
    assert "Full versions" not in rendered
    assert "one matching current read" not in rendered


@pytest.mark.parametrize("present, validity", [
    (False, None), (True, None), (True, "stale"), (True, []), (True, 1),
])
def test_missing_or_malformed_validity_stays_unknown(expand, present, validity):
    row = {"topic": "legacy", "content": "old note"}
    if present:
        row["validity"] = validity
    payload = decode(expand([row]))[0][1]
    assert payload["validity_status"] == "unknown"
    assert payload["metadata"] == {k: v for k, v in row.items() if k not in {"topic", "content"}}
    assert "unchanged" not in json.dumps(payload)


def test_long_conclusion_is_omitted_whole_and_assumptions_remain(expand):
    row = {"topic": "negation at end", "content": "A" * 3000 + " NOT safe for live reuse.",
           "justification": {"kind": "decision", "assumptions": ["not production"]},
           "validity": {"state": "stale"}}
    payload = decode(expand([row]))[0][1]
    assert payload["content_omitted"] is True
    assert "content" not in payload
    assert payload["metadata"]["justification"] == row["justification"]
    assert payload["expand"] == {"topic": row["topic"], "detail": "full"}


def test_oversized_metadata_is_omitted_whole_with_reference(expand):
    row = {"topic": "large scope", "content": "A claim with qualifications.",
           "justification": {"assumptions": ["scope" * 1000]},
           "validity": {"state": "unchanged"}}
    payload = decode(expand([row]))[0][1]
    assert payload["content_omitted"] is True
    assert payload["metadata_omitted"] is True
    assert "content" not in payload and "metadata" not in payload
    assert payload["validity_status"] == "omitted"
    assert payload["expand"] == {"topic": row["topic"], "detail": "full"}
    assert "unchanged" not in json.dumps(payload)


def test_final_escaped_representation_obeys_per_finding_and_total_budget(expand):
    rows = [{"topic": f'entry{i}<&"', "content": "<&東京>" * 35,
             "justification": {"assumptions": ["<&東京>" * 20]},
             "validity": {"state": "stale"}} for i in range(3)]
    rendered = expand(rows)
    assert len(rendered) <= 5000
    assert len(decode(rendered)) >= 2
    for finding in ET.fromstring(rendered).findall("finding"):
        assert len(ET.tostring(finding, encoding="unicode")) <= 1800
    for topic, payload in decode(rendered):
        assert topic in [row["topic"] for row in rows]
        assert payload["metadata"]["validity"] == {"state": "stale"}


def test_identifier_too_large_has_explicit_record_omission(expand):
    rendered = expand([{"topic": "X" * 6000, "content": "must not be detached from identity"}])
    assert len(rendered) <= 5000
    assert "omitted" in rendered
    assert decode(rendered) == []


def test_global_budget_omits_content_without_chopping_a_claim(expand):
    rows = [{"topic": f"finding{i}", "content": "X" * 1500} for i in range(3)]
    rendered = expand(rows)
    payloads = [payload for _, payload in decode(rendered)]
    assert len(rendered) <= 5000
    assert len(payloads) == 3
    assert any(payload["content_omitted"] for payload in payloads)
    assert any(not payload["content_omitted"] for payload in payloads)
    assert all("content" not in payload or payload["content"] == "X" * 1500
               for payload in payloads)


def test_only_findings_selected_by_brief_are_expanded(expand):
    rendered = expand([{"topic": "selected", "content": "keep"},
                       {"topic": "other", "content": "omit"}],
                      brief="<ken-session-brief>selected</ken-session-brief>")
    assert [topic for topic, _ in decode(rendered)] == ["selected"]
    assert ET.fromstring(rendered).find("finding-source").attrib["root"] == str(expand.root)


@pytest.mark.parametrize("rows", [None, {}, "bad", [None, {}, {"topic": "no content"}]])
def test_malformed_rows_degrade_without_a_claim(expand, rows):
    assert expand(rows, brief="<ken-session-brief>no content</ken-session-brief>") == ""


@pytest.mark.parametrize("failure", [
    OSError("offline failure"), subprocess.TimeoutExpired("fake ken", 3),
    SimpleNamespace(returncode=1, stdout="failed"),
    SimpleNamespace(returncode=0, stdout="{broken"),
])
def test_cli_errors_do_not_invent_findings(expand, monkeypatch, failure):
    def run(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return failure
    monkeypatch.setattr("infinidev.engine.ken_session.subprocess.run", run)
    assert expand.session._expanded_recent_findings("<ken-session-brief>x</ken-session-brief>") == ""


def test_escaped_source_identity_cannot_overflow_total_budget(expand):
    from pathlib import Path
    expand.session._root = Path("/" + "<&" * 3000)
    rendered = expand([{"topic": "example", "content": "no unattributed claim"}])
    assert len(rendered) <= 5000
    assert 'omitted="budget"' in rendered
    assert decode(rendered) == []


@pytest.mark.parametrize("identity", ["a\nb", "a\rb", "a\tb"])
def test_topic_identity_survives_xml_attribute_normalization(expand, identity):
    rendered = expand([{"topic": identity, "content": "X" * 3000}])
    topic, payload = decode(rendered)[0]
    assert topic == identity
    assert payload["expand"]["topic"] == identity


def test_source_identity_survives_xml_attribute_normalization(expand):
    from pathlib import Path
    expand.session._root = Path("/source\nwith\rwhitespace\tinside")
    rendered = expand([{"topic": "example", "content": "known"}])
    assert ET.fromstring(rendered).find("finding-source").attrib["root"] == str(expand.session._root)


@pytest.mark.parametrize("identity", ["bad\x00", "bad\ud800", "bad\ufffe"])
def test_invalid_xml_topic_is_explicitly_omitted(expand, identity):
    rendered = expand([{"topic": identity, "content": "must stay attributed"}])
    assert len(rendered) <= 5000
    assert "invalid-xml-identity" in rendered
    assert decode(rendered) == []


def test_both_omission_markers_share_the_total_budget(expand):
    from pathlib import Path
    rows = [{"topic": "tiny", "content": "known"},
            {"topic": "invalid\x00", "content": "never unattributed"},
            {"topic": "X" * 6000, "content": "never unattributed"}]
    expand.session._root = Path("/scope")
    baseline = expand(rows)
    # Grow only the escaped source identity to the boundary with both omissions.
    padding = 5001 - len(baseline)
    expand.session._root = Path("/scope" + "&" * (padding // 5) + "X" * (padding % 5))
    rendered = expand(rows)
    assert 'reason="budget"' in rendered
    assert 'reason="invalid-xml-identity"' in rendered
    assert len(rendered) <= 5000
