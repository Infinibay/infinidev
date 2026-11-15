"""Ken facade — semantic index, grep, and durable memory over MCP.

`ken <https://pypi.org/project/ken-rank/>`_ indexes the workspace (symbols,
files, call graph, saved findings) and exposes it as an MCP server. This
module is the single place that knows Ken's tool names and result shapes;
everything else in Infinidev talks to :class:`KenClient`.

Design rule: **Ken augments, never gates.** Every method degrades to a
local implementation when the server is missing, the project has no
``.ken`` index, or a call times out — callers get slightly worse results,
never an exception and never an empty list where data exists.

Code search accepts both legacy flat rows and modern SearchResponse evidence.
Modern entity identities, disjoint excerpts and source-authority metadata remain
available to callers, including diagnostics on an empty response. Ken tool names
and JSON negotiation are owned by this facade; the transport stays generic.

"""

from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, TypeVar

from infinidev.engine.mcp_client import (
    McpHit,
    McpManager,
    get_default_mcp_manager,
    keyword_search,
)

logger = logging.getLogger(__name__)

KEN_SERVER = "ken"


@dataclass(slots=True)
class SymbolHit:
    """A symbol located by Ken (or by the local index in fallback mode)."""

    qualname: str
    kind: str
    file: str
    line: int = 0
    docstring: str = ""
    score: float | None = 0.0
    source: str = "ken"
    snippet: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        doc = f" — {self.docstring}" if self.docstring else ""
        text = f"{self.kind:10} {self.qualname}  {self.file}:{self.line}{doc}"
        return "\n".join(part for part in (text, self.snippet, _code_metadata(self.metadata)) if part)


@dataclass(slots=True)
class GrepMatch:
    """One line-cited match from Ken's worktree grep."""

    path: str
    line: int
    text: str


@dataclass(slots=True)
class MemoryHit:
    """A finding recalled from Ken's durable memory."""

    topic: str
    content: str
    tags: list[str] = field(default_factory=list)
    score: float = 0.0
    created_at: str = ""
    justification: dict[str, Any] | None = None
    validity: dict[str, Any] | None = None

    def render(self, width: int = 240) -> str:
        body = self.content if len(self.content) <= width else self.content[:width] + "…"
        tags = f"  [{', '.join(self.tags)}]" if self.tags else ""
        lines = [f"{self.topic}{tags}\n{body}"]
        if self.justification is not None:
            lines.append("Justification: " + json.dumps(self.justification, ensure_ascii=False))
        if self.validity is not None:
            lines.append("Validity: " + json.dumps(self.validity, ensure_ascii=False))
        else:
            lines.append("Validity: not supplied.")
        lines.append(
            "Validity describes declared inputs; it does not certify the conclusion "
            "or its assumptions."
        )
        return "\n".join(lines)


SearchHit = TypeVar("SearchHit")


class CodeSearchResults(list[SearchHit]):
    """List-compatible results retaining diagnostics even when no hits exist."""

    def __init__(self, hits: list[SearchHit], *, response: dict[str, Any]) -> None:
        super().__init__(hits)
        self.response = deepcopy(response)

    def __getitem__(self, index: Any) -> Any:
        result = super().__getitem__(index)
        return type(self)(result, response=self.response) if isinstance(index, slice) else result

    def render_diagnostics(self) -> str:
        return ("Ken search response: " + json.dumps(self.response, ensure_ascii=False)
                if self.response else "")


@dataclass(slots=True)
class CodeFileHit(McpHit):
    """A code match whose score may be absent and whose source status is explicit."""

    score: float | None
    metadata: dict[str, Any] = field(default_factory=dict)

    def render(self) -> str:
        return "\n".join(part for part in (
            f"{self.target}:{self.line}", self.snippet, _code_metadata(self.metadata),
        ) if part)


def _code_evidence(row: dict[str, Any]) -> str:
    return "\n\n".join(
        f"{fragment.get('start_line', '?')}-{fragment.get('end_line', '?')}\n"
        + fragment["content"]
        for fragment in row.get("evidence", [])
        if isinstance(fragment, dict) and isinstance(fragment.get("content"), str)
    )


def _code_metadata(metadata: dict[str, Any]) -> str:
    # Evidence already appears with its disjoint spans in the snippet above.
    row = metadata.get("result")
    if not row:
        return ""
    return "Ken source metadata: " + json.dumps(
        {key: value for key, value in row.items() if key != "evidence"}, ensure_ascii=False,
    )


def _code_rows(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        return _as_rows(payload), {}
    if isinstance(payload, dict):
        response = deepcopy({key: value for key, value in payload.items()
                             if key not in {"results", "hits", "items"}})
        if payload.get("ok") is False:
            return [], response
        if any(payload.get(key) for key in ("path", "qualname", "name")):
            return [payload], {}
        if any(isinstance(payload.get(key), list)
               for key in ("results", "hits", "items", "symbols", "files")):
            return _as_rows(payload), response
    return [], {"ok": False, "error": "Unsupported Ken search response"}


class KenClient:
    """Typed access to Ken's tools, with local fallbacks for every call."""

    def __init__(self, manager: McpManager | None = None) -> None:
        self._manager = manager if manager is not None else get_default_mcp_manager()
        self._server = self._resolve_server()

    # ── availability ─────────────────────────────────────────────────

    def _resolve_server(self) -> str | None:
        for name in self._manager.all_names():
            if name.lower() == KEN_SERVER:
                return name
        return None

    @property
    def available(self) -> bool:
        """Whether a Ken server is registered *and* its binary is present.

        This does not prove the index exists — ``ken mcp`` exits when the
        workspace has no ``.ken`` directory. That case surfaces as a failed
        call and is absorbed by the fallbacks below.
        """
        if self._server is None:
            return False
        client = self._manager.get(self._server)
        return bool(client and client.available)

    @property
    def reason(self) -> str:
        if self._server is None:
            return "no 'ken' server in the MCP config"
        client = self._manager.get(self._server)
        return client.unavailable_reason if client else "server not registered"

    def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Call a Ken tool, returning its structured payload or ``None``."""
        if self._server is None:
            return None
        for published in self._manager.list_tools(self._server):
            if published.name != tool:
                continue
            output_format = published.input_schema.get("properties", {}).get("output_format", {})
            if output_format.get("type") == "string" and (
                "enum" not in output_format or "json" in output_format["enum"]
            ):
                arguments = {**arguments, "output_format": "json"}
            break
        result = self._manager.try_call(self._server, tool, arguments)
        return None if result is None else result.data

    # ── semantic search ──────────────────────────────────────────────

    def search_files(self, query: str, limit: int = 8) -> CodeSearchResults[McpHit]:
        """Files in server rank order, with evidence and declared source status."""
        payload = self._call(
            "ken_find", {"query": query, "scope": "files", "limit": limit},
        )
        if payload is None:
            return CodeSearchResults(keyword_search(query, limit, kinds={"files"}), response={
                "source": "fallback", "reason": "Ken call failed or was unavailable",
            })
        rows, response = _code_rows(payload)
        hits: list[McpHit] = []
        for row in rows:
            path = str(row.get("path") or "")
            if not path:
                continue
            modern = isinstance(row.get("entity"), dict)
            snippet = _code_evidence(row) if modern else ", ".join(
                str(symbol.get("name", ""))
                for symbol in (row.get("symbols") or [])[:6] if isinstance(symbol, dict)
            )
            hits.append(CodeFileHit(
                target=path, snippet=snippet, source="ken",
                score=row.get("ranking_score", row.get("score")) if modern
                else float(row.get("score", 0.0) or 0.0),
                line=int((row.get("entity_span") or {}).get("start_line", 0)) if modern else 0,
                metadata={"result": deepcopy(row), "response": deepcopy(response)},
            ))
        return CodeSearchResults(hits[:limit], response=response)

    def search_symbols(self, query: str, limit: int = 10) -> CodeSearchResults[SymbolHit]:
        """Symbols or requested pieces, retaining evidence and response diagnostics."""
        payload = self._call(
            "ken_find", {"query": query, "scope": "symbols", "limit": limit},
        )
        if payload is None:
            return CodeSearchResults([], response={
                "ok": False, "error": "Ken call failed or was unavailable",
            })
        rows, response = _code_rows(payload)
        hits: list[SymbolHit] = []
        for row in rows:
            modern = isinstance(row.get("entity"), dict)
            entity = row["entity"] if modern else row
            path = str(row.get("file") or row.get("path") or "")
            qualname = str(entity.get("qualname") or entity.get("name") or
                           (path if modern else ""))
            if not qualname:
                continue
            hits.append(SymbolHit(
                qualname=qualname, kind=str(entity.get("kind", "symbol")), file=path,
                line=int((row.get("entity_span") or {}).get("start_line", 0)) if modern
                else int(row.get("line", 0) or 0),
                docstring=str(row.get("docstring") or "").strip(),
                score=row.get("ranking_score", row.get("score")) if modern
                else float(row.get("score", 0.0) or 0.0),
                snippet=_code_evidence(row) if modern else "",
                metadata={"result": deepcopy(row), "response": deepcopy(response)},
            ))
        return CodeSearchResults(hits[:limit], response=response)

    def grep(
        self,
        query: str,
        *,
        mode: str = "literal",
        language: str | None = None,
        limit: int = 20,
    ) -> list[GrepMatch]:
        """Line-cited matches from the live worktree (never stale).

        ``mode='literal'`` is an exact substring scan; ``mode='bm25'`` is
        ranked relevance over Ken's FTS index.
        """
        # ken's text scope carries the mode as a boolean, not a string.
        payload: dict[str, Any] = {
            "query": query,
            "scope": "text",
            "literal": mode == "literal",
            "limit": limit,
        }
        if language:
            payload["language"] = language
        data = self._call("ken_find", payload)
        matches: list[GrepMatch] = []
        if isinstance(data, dict):
            for row in data.get("results", []) or []:
                if not isinstance(row, dict):
                    continue
                path = str(row.get("path", ""))
                # The two modes report differently, and only the literal
                # shape was handled: an exact scan knows which lines matched
                # and returns ``snippets`` with line numbers, while a ranked
                # BM25 hit is one ``snippet`` string with nothing to cite.
                # Reading only the plural form made bm25 silently return
                # nothing at all.
                for snippet in row.get("snippets", []) or []:
                    if not isinstance(snippet, dict):
                        continue
                    matches.append(
                        GrepMatch(
                            path=path,
                            line=int(snippet.get("line", 0) or 0),
                            text=str(snippet.get("text", "")).rstrip(),
                        )
                    )
                single = row.get("snippet")
                if isinstance(single, str) and single.strip():
                    matches.append(
                        GrepMatch(path=path, line=0, text=single.strip())
                    )
        return matches[:limit]

    def callers_of(self, qualname: str, limit: int = 30) -> list[SymbolHit]:
        """Resolved call-sites that reference *qualname*."""
        data = self._call(
            "ken_related",
            {"target": qualname, "relation": "callers", "limit": limit},
        )
        if not isinstance(data, dict):
            return []
        hits: list[SymbolHit] = []
        for row in data.get("callers", []) or []:
            if not isinstance(row, dict):
                continue
            hits.append(
                SymbolHit(
                    # ``from_qualname`` is what the call graph reports; the
                    # older spellings stay as fallbacks.
                    qualname=str(
                        row.get("from_qualname")
                        or row.get("qualname")
                        or row.get("caller")
                        or ""
                    ),
                    kind=str(row.get("kind", "call")),
                    file=str(row.get("file") or row.get("path") or ""),
                    line=int(row.get("line", 0) or 0),
                    docstring=str(row.get("confidence", "") or ""),
                )
            )
        return [hit for hit in hits if hit.qualname][:limit]

    def file_symbols(self, path: str) -> list[SymbolHit]:
        """Indexed symbol structure for one file."""
        data = self._call(
            "ken_read", {"path": path, "include": ["symbols", "docstrings"]},
        )
        rows: list[dict[str, Any]] = []
        if isinstance(data, dict):
            # ken_read nests the symbol list under ``outline`` when it renders
            # a file's structure; older shapes put it at the top level.
            outline = data.get("outline")
            raw = (
                outline.get("symbols") if isinstance(outline, dict)
                else data.get("symbols")
            ) or []
            rows = [row for row in raw if isinstance(row, dict)]
        elif isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
        return [
            SymbolHit(
                qualname=str(row.get("qualname") or row.get("name") or ""),
                kind=str(row.get("kind", "symbol")),
                file=path,
                line=int(row.get("line", 0) or 0),
                docstring=str(row.get("docstring") or "").strip(),
            )
            for row in rows
            if row.get("name") or row.get("qualname")
        ]

    # ── durable memory ───────────────────────────────────────────────

    def recall(
        self, query: str, limit: int = 5, min_score: float = 0.25
    ) -> list[MemoryHit]:
        """Findings saved in earlier sessions, ranked by similarity."""
        rows = self._call(
            "ken_recall", {"query": query, "limit": limit, "min_score": min_score}
        )
        hits = [
            MemoryHit(
                topic=str(row.get("topic", "")),
                content=str(row.get("content", "")),
                tags=[str(tag) for tag in (row.get("tags") or [])],
                score=float(row.get("score", 0.0) or 0.0),
                created_at=str(row.get("created_at", "")),
                justification=deepcopy(row["justification"])
                if isinstance(row.get("justification"), dict) else None,
                validity=deepcopy(row["validity"])
                if isinstance(row.get("validity"), dict) else None,
            )
            for row in _as_rows(rows)
            if row.get("topic") or row.get("content")
        ]
        if hits:
            return hits[:limit]
        return [
            MemoryHit(topic=hit.target, content=hit.snippet, score=hit.score)
            for hit in keyword_search(query, limit, kinds={"memory"})
        ]

    def remember(
        self, topic: str, content: str, tags: list[str] | None = None,
        *, justification: dict[str, Any] | None = None,
    ) -> bool:
        """Persist a finding; omitting justification revokes any previous one in Ken."""
        payload: dict[str, Any] = {"topic": topic, "content": content}
        if tags:
            payload["tags"] = list(tags)
        if justification is not None:
            payload["justification"] = deepcopy(justification)
        result = self._call("ken_remember", payload)
        return isinstance(result, dict) and result.get("ok") is True

    # ── context ranking (index history) ──────────────────────────────

    def rank(self, query: str = "", verbose: int = 0, max_chars: int = 0) -> str:
        """Ken's ranked context block for *query* — empty string if absent.

        This is the "index history" channel: Ken tracks which files and
        symbols mattered for previous prompts in this project and folds
        that history into the ranking.
        """
        data = self._call(
            "ken_rank",
            {"query": query, "verbose": int(verbose), "max_chars": int(max_chars)},
        )
        if isinstance(data, dict) and data.get("ok"):
            return str(data.get("context_block", "") or "")
        return ""

    def project_overview(self, depth: int = 2, limit: int = 20) -> dict[str, Any]:
        data = self._call("ken_rank", {"scope": "project", "verbose": depth})
        return data if isinstance(data, dict) else {}

    def status(self) -> dict[str, Any]:
        """Health snapshot for the `/mcp` panel and diagnostics."""
        client = self._manager.get(self._server) if self._server else None
        return {
            "server": self._server,
            "available": self.available,
            "running": bool(client and client.running),
            "reason": self.reason,
            "fallback": not self.available,
        }

    # ── legacy aliases (kept so older call sites keep working) ───────

    def search(self, query: str, limit: int = 10) -> list[McpHit]:
        return self.search_files(query, limit)

    def memory_search(self, query: str, limit: int = 5) -> list[McpHit]:
        return [
            McpHit(
                target=hit.topic, snippet=hit.render(width=len(hit.content)),
                score=hit.score, source="ken",
            )
            for hit in self.recall(query, limit=limit)
        ]

    def index_status(self) -> dict[str, Any]:
        return self.status()


def _as_rows(value: Any) -> list[dict[str, Any]]:
    """Coerce a Ken payload into a list of dict rows."""
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("results", "hits", "items", "symbols", "files"):
            nested = value.get(key)
            if isinstance(nested, list):
                return [row for row in nested if isinstance(row, dict)]
        return [value]
    return []


_client: KenClient | None = None
_client_lock = threading.Lock()


def get_ken_client() -> KenClient:
    """Return the process-wide Ken facade."""
    global _client
    with _client_lock:
        if _client is None:
            _client = KenClient()
        return _client


def reset_ken_client() -> None:
    """Drop the cached facade — used by tests and after config reloads."""
    global _client
    with _client_lock:
        _client = None
