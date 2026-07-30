"""A minimal but *protocol-conformant* MCP stdio server, used by tests.

Speaks the same JSON-RPC dialect as a real SDK server: it refuses
``tools/list`` and ``tools/call`` until ``initialize`` has been answered,
emits an unsolicited notification (no ``id``) to prove the client
correlates responses instead of trusting line order, and wraps results in
``content`` / ``structuredContent`` / ``isError``.

Behaviour is driven by argv so one script covers every scenario:

    python mcp_fake_server.py                 # well-behaved
    python mcp_fake_server.py --no-handshake  # rejects initialize
    python mcp_fake_server.py --hang          # accepts, never answers calls
    python mcp_fake_server.py --crash-after 2 # exits mid-session
    python mcp_fake_server.py --noise         # writes junk to stdout/stderr
"""

from __future__ import annotations

import json
import sys
import time


def _write(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _ok(request_id, result) -> None:
    _write({"jsonrpc": "2.0", "id": request_id, "result": result})


def _text_result(payload) -> dict:
    """Mirror the SDK: JSON text blocks plus a structuredContent mirror."""
    blocks = payload if isinstance(payload, list) else [payload]
    return {
        "content": [
            {"type": "text", "text": json.dumps(block)} for block in blocks
        ],
        "structuredContent": {"result": payload},
        "isError": False,
    }


# Mirrors ken's real surface. The fake exists so the client is exercised
# against the protocol without a live server — which only holds if the names
# and payload shapes are the ones ken actually publishes. When they drifted,
# the tests kept passing against a ken that no longer existed.
_SCOPE = {"enum": ["files", "symbols", "text", "tests", "wiring", "intent"],
          "type": "string", "default": "files"}

TOOLS = [
    {
        "name": "ken_find",
        "description": "Find things by describing them, over one of six scopes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}, "scope": _SCOPE,
                "limit": {"type": "integer"}, "literal": {"type": "boolean"},
                "language": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "ken_read",
        "description": "Read an indexed file's structure.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "include": {"type": "array", "items": {"type": "string"}},
                "qualname": {"type": "string"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "ken_related",
        "description": "What else is connected to a target.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "target": {"type": "string"},
                "relation": {"enum": ["neighbors", "imports", "callers", "callees",
                                      "subtypes", "supertypes", "cochange",
                                      "blast_radius", "clones"], "type": "string"},
                "limit": {"type": "integer"}, "depth": {"type": "integer"},
                "min_confidence": {"type": "number"},
            },
            "required": ["target", "relation"],
        },
    },
    {
        "name": "ken_rank",
        "description": "Ranked context block.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scope": {"enum": ["session", "changes", "project", "architecture"],
                          "type": "string", "default": "session"},
                "query": {"type": "string"}, "verbose": {"type": "integer"},
                "explain": {"type": "boolean"}, "max_chars": {"type": "integer"},
            },
        },
    },
    {
        "name": "ken_recall",
        "description": "Recall saved findings.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}, "path": {"type": "string"},
                "topic": {"type": "string"}, "tag": {"type": "string"},
                "limit": {"type": "integer"}, "min_score": {"type": "number"},
                "anchor_file": {"type": "string"}, "anchor_symbol": {"type": "string"},
                "anchor_tool": {"type": "string"}, "anchor_error": {"type": "string"},
            },
        },
    },
    {
        "name": "ken_remember",
        "description": "Save a finding.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "topic": {"type": "string"}, "content": {"type": "string"},
                "action": {"enum": ["save", "forget", "dismiss"], "type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
                "anchor_file": {"type": "string"}, "anchor_symbol": {"type": "string"},
                "anchor_tool": {"type": "string"}, "anchor_error": {"type": "string"},
            },
            "required": ["topic"],
        },
    },
]


def _dispatch(name: str, arguments: dict):
    if name == "ken_find":
        scope = arguments.get("scope", "files")
        if scope == "files":
            return _text_result(
                [
                    {
                        "path": "src/auth.py",
                        "language": "python",
                        "score": 0.71,
                        "symbols": [{"kind": "function", "name": "verify_token", "line": 12}],
                    }
                ]
            )
        if scope == "symbols":
            return _text_result(
                [
                    {
                        "qualname": "verify_token",
                        "kind": "function",
                        "file": "src/auth.py",
                        "line": 12,
                        "docstring": "Validate a JWT.",
                        "score": 0.66,
                    }
                ]
            )
        if scope == "text":
            literal = bool(arguments.get("literal", False))
            if literal:
                return _text_result({
                    "ok": True, "mode": "literal",
                    "results": [{
                        "path": "src/auth.py", "count": 1,
                        "snippets": [{"line": 12, "text": "def verify_token(tok):"}],
                    }],
                })
            # BM25 reports one snippet string per hit, with no line to cite.
            return _text_result({
                "ok": True, "mode": "bm25",
                "results": [{"path": "src/auth.py", "score": -9.1,
                             "snippet": "… def [verify_token](tok): …"}],
            })
        return _text_result([])
    if name == "ken_read":
        return _text_result({
            "ok": True, "path": arguments.get("path", ""),
            "outline": {
                "ok": True, "path": arguments.get("path", ""), "language": "python",
                "symbols": [{"kind": "function", "name": "verify_token",
                             "qualname": "verify_token", "line": 12, "line_end": 20,
                             "docstring": "Validate a JWT."}],
            },
        })
    if name == "ken_related":
        return _text_result({
            "ok": True, "qualname": arguments.get("target", ""),
            "callers": [{"from_qualname": "login", "file": "src/routes.py",
                         "line": 40, "confidence_tier": "T1"}],
        })
    if name == "ken_recall":
        return _text_result(
            [
                {
                    "topic": "jwt-clock-skew",
                    "content": "Tokens allow 60s skew.",
                    "tags": ["auth"],
                    "score": 0.52,
                    "created_at": "2026-01-01T00:00:00Z",
                }
            ]
        )
    if name == "ken_rank":
        if arguments.get("scope") == "project":
            return _text_result({"ok": True, "files": 3, "languages": ["python"]})
        return _text_result({"ok": True, "context_block": "<context-rank>x</context-rank>"})
    if name == "ken_remember":
        return _text_result({"ok": True, "topic": arguments.get("topic", "")})
    return {
        "content": [{"type": "text", "text": f"Unknown tool {name}"}],
        "isError": True,
    }


def main() -> int:
    argv = sys.argv[1:]
    no_handshake = "--no-handshake" in argv
    hang = "--hang" in argv
    noise = "--noise" in argv
    crash_after = 0
    if "--crash-after" in argv:
        crash_after = int(argv[argv.index("--crash-after") + 1])

    if noise:
        sys.stdout.write("starting up, not JSON\n")
        sys.stdout.flush()
        sys.stderr.write("fake server booting\n")
        sys.stderr.flush()

    initialized = False
    handled = 0

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            continue

        method = message.get("method")
        request_id = message.get("id")
        handled += 1

        if crash_after and handled > crash_after:
            return 3

        if method == "initialize":
            if no_handshake:
                _write(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32600, "message": "initialize refused"},
                    }
                )
                continue
            initialized = True
            _ok(
                request_id,
                {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": "fake-ken", "version": "1.0"},
                },
            )
            continue

        if method == "notifications/initialized":
            continue

        if not initialized:
            _write(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32002, "message": "not initialized"},
                }
            )
            continue

        if hang:
            time.sleep(30)
            continue

        # An unsolicited notification before the real answer: a client that
        # trusts line order instead of matching ids will pick this up.
        _write({"jsonrpc": "2.0", "method": "notifications/message", "params": {}})

        if method == "tools/list":
            _ok(request_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = message.get("params", {})
            _ok(request_id, _dispatch(params.get("name", ""), params.get("arguments", {})))
        else:
            _ok(request_id, {})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
