"""``KenClient`` calls ken by name, and nothing told us when that stopped working.

The client is the internal path behind the deterministic tools, and by design
it degrades to a local implementation when ken cannot answer — "ken augments,
never gates". That is the right behaviour and it is also why renaming ken's
tool surface broke every one of these calls in silence: `try_call` returned
None, the fallback ran, and the only symptom was worse results.

Two failure modes, one guard each. The names have to exist on the server, and
the payloads the client reads have to be the ones the server sends.
"""

from __future__ import annotations

import json
import select
import shutil
import subprocess
import time

import pytest

_LIVE_RESPONSE_TIMEOUT_SECONDS = 10.0


def _read_live_message(proc: subprocess.Popen[str], deadline: float) -> dict:
    """Read one JSON-RPC message without letting an optional server hang pytest."""
    remaining = deadline - time.monotonic()
    if remaining <= 0 or not select.select([proc.stdout], [], [], remaining)[0]:
        pytest.skip(
            "ken mcp did not answer initialize within "
            f"{_LIVE_RESPONSE_TIMEOUT_SECONDS:.0f}s; live contract unverified"
        )
    line = proc.stdout.readline()
    if not line:
        pytest.skip("ken mcp closed the pipe")
    return json.loads(line)


def _live_surface() -> dict[str, dict]:
    """``tools/list`` from a real ``ken mcp``, as ``{name: inputSchema}``."""
    binary = shutil.which("ken")
    if binary is None:
        pytest.skip("ken is not installed; nothing to check the contract against")

    proc = subprocess.Popen(
        [binary, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )

    def send(obj):
        proc.stdin.write(json.dumps(obj) + "\n")
        proc.stdin.flush()

    try:
        deadline = time.monotonic() + _LIVE_RESPONSE_TIMEOUT_SECONDS
        send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
            "protocolVersion": "2024-11-05", "capabilities": {},
            "clientInfo": {"name": "contract-test", "version": "0"}}})
        while True:
            msg = _read_live_message(proc, deadline)
            if msg.get("id") == 1:
                send({"jsonrpc": "2.0", "method": "notifications/initialized",
                      "params": {}})
                send({"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                      "params": {}})
            elif msg.get("id") == 2:
                return {t["name"]: t["inputSchema"] for t in msg["result"]["tools"]}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


# Every ``_call`` in KenClient, as (tool, arguments it sends). Kept here rather
# than scraped from the source: a literal list is what makes a rename show up
# as a failing test instead of a test that quietly checks nothing.
CALLS = [
    ("ken_find", {"query", "scope", "limit"}),
    ("ken_find", {"query", "scope", "literal", "limit", "language"}),
    ("ken_related", {"target", "relation", "limit"}),
    ("ken_read", {"path", "include"}),
    ("ken_rank", {"query", "verbose", "max_chars"}),
    ("ken_rank", {"scope", "verbose"}),
    ("ken_remember", {"topic", "content", "tags"}),
]


@pytest.fixture(scope="module")
def live_surface() -> dict[str, dict]:
    """Probe the optional live server once for all contract assertions."""
    return _live_surface()


def test_every_tool_the_client_calls_still_exists(live_surface):
    """The break itself: six names went away and the fallback hid it."""
    live = live_surface
    missing = sorted({name for name, _ in CALLS} - set(live))
    assert not missing, (
        f"KenClient calls tools ken no longer publishes: {missing}. "
        f"The live surface is {sorted(live)}."
    )


def test_every_argument_the_client_sends_is_accepted(live_surface):
    """A surviving name with renamed arguments fails the same silent way."""
    live = live_surface
    problems = []
    for name, args in CALLS:
        schema = live.get(name)
        if schema is None:
            continue
        accepted = set(schema.get("properties", {}))
        unknown = sorted(args - accepted)
        if unknown:
            problems.append(f"{name}: {unknown} (accepts {sorted(accepted)})")
    assert not problems, "KenClient sends arguments ken does not accept:\n" + "\n".join(problems)


def test_enum_arguments_are_sent_valid_values(live_surface):
    """``scope`` and ``relation`` are closed sets; a stale value returns an
    error payload the client reads as "no results", which is the same silence
    as a missing tool."""
    live = live_surface
    sent = {
        ("ken_find", "scope"): {"files", "symbols", "text"},
        ("ken_related", "relation"): {"callers"},
        ("ken_rank", "scope"): {"project"},
    }
    problems = []
    for (tool, arg), values in sent.items():
        prop = live.get(tool, {}).get("properties", {}).get(arg, {})
        allowed = set(prop.get("enum") or [])
        if allowed and (bad := sorted(values - allowed)):
            problems.append(f"{tool}.{arg}={bad} not in {sorted(allowed)}")
    assert not problems, "\n".join(problems)
