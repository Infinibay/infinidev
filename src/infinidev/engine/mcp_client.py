"""Model Context Protocol (MCP) client — stdio JSON-RPC, spec-conformant.

Servers are declared in a JSON config (``.mcp.json`` in the workspace or
``~/.infinidev/mcp.json``) using the standard shape every MCP host uses::

    {
      "mcpServers": {
        "ken": {"command": "ken", "args": ["mcp"], "timeout": 30, "tool_ttl": 300}
      }
    }

Protocol notes (why this file looks the way it does):

* **Handshake is mandatory.** A server built on the official SDK rejects
  ``tools/list`` until the client has sent ``initialize`` and the
  ``notifications/initialized`` notification. The old implementation
  skipped both.
* **Responses must be matched by ``id``.** Servers are free to interleave
  notifications (``{"method": "notifications/message"}``, no ``id``) with
  responses, so ``readline()`` is not "the answer to my request".
* **stdout/stderr must be drained continuously.** A 64 KiB pipe buffer that
  nobody reads deadlocks the child. One reader thread per stream: stdout
  feeds a queue of parsed messages, stderr feeds the log plus a small ring
  buffer we can show the user when a server refuses to start.
* **Timeouts must be real.** ``readline()`` on a hung server blocks forever;
  reading from a queue with a deadline does not.

Tool results come back as ``{"content": [...], "structuredContent": {...},
"isError": bool}``. :class:`McpToolResult` normalises that into text + data
so callers never parse MCP envelopes by hand.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from infinidev.config.settings import get_base_dir, settings

logger = logging.getLogger(__name__)

# The protocol revision we negotiate. Servers echo back their own version;
# we accept whatever they answer (the SDK downgrades gracefully).
PROTOCOL_VERSION = "2025-06-18"

CLIENT_INFO = {"name": "infinidev", "version": "0.13.0"}

# stderr lines kept per server for diagnostics (`/mcp` panel, logs).
_STDERR_RING = 40


@dataclass(slots=True)
class McpTool:
    """A single tool exposed by an MCP server.

    ``annotations`` carries the spec's optional behaviour hints
    (``readOnlyHint``, ``destructiveHint``, …). Servers are not required to
    send them — ``ken`` currently does not — so consumers must treat an
    empty dict as "unknown", never as "safe".
    """

    server: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class McpHit:
    """A structured search hit returned by an MCP-backed tool."""

    target: str
    snippet: str
    score: float
    source: str
    line: int = 0


@dataclass(slots=True)
class McpToolResult:
    """Normalised ``tools/call`` result.

    ``data`` holds the server's structured payload when it provides one
    (SDK servers return ``structuredContent: {"result": ...}`` for typed
    return values); otherwise it holds the JSON parsed out of the text
    blocks, or ``None`` when the content is plain prose.
    """

    is_error: bool = False
    text: str = ""
    data: Any = None

    def rows(self) -> list[dict[str, Any]]:
        """Return ``data`` as a list of dicts, whatever shape it arrived in."""
        if isinstance(self.data, list):
            return [row for row in self.data if isinstance(row, dict)]
        if isinstance(self.data, dict):
            for key in ("results", "hits", "matches", "items", "symbols", "files"):
                value = self.data.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            return [self.data]
        return []


class McpUnavailable(RuntimeError):
    """Raised when an MCP server is unreachable or answers with an error."""


class McpServerClient:
    """JSON-RPC client owning one MCP server subprocess."""

    def __init__(
        self,
        name: str,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout: float | None = None,
        startup_timeout: float | None = None,
        tool_ttl: float | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.name = name
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd
        self._timeout = timeout
        self._startup_timeout = startup_timeout
        self._tool_ttl = tool_ttl
        self._on_event = on_event

        self._process: subprocess.Popen[str] | None = None
        self._inbox: queue.Queue[dict[str, Any] | None] | None = None
        self._stderr_lines: deque[str] = deque(maxlen=_STDERR_RING)
        self._pending: dict[int, dict[str, Any]] = {}
        self._next_id = 1
        self._initialized = False
        self._server_info: dict[str, Any] = {}

        self._lock = threading.RLock()
        self._tools: list[McpTool] | None = None
        self._tools_loaded_at: float | None = None
        self._failure_count = 0
        self._last_failure: float | None = None
        self._next_retry_at: float | None = None
        self._unavailable_reason: str = ""

    # ── introspection ────────────────────────────────────────────────

    @property
    def available(self) -> bool:
        """Whether the server's executable exists on PATH."""
        return self._resolve_executable() is not None

    @property
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    @property
    def unavailable_reason(self) -> str:
        if self._unavailable_reason:
            return self._unavailable_reason
        if not self.available:
            return f"{self.command!r} not found on PATH"
        return ""

    def stderr_tail(self, limit: int = 10) -> list[str]:
        """Last stderr lines — what to show when a server refuses to start."""
        return list(self._stderr_lines)[-limit:]

    # ── events / failure bookkeeping ─────────────────────────────────

    def _emit(self, event: str, **payload: Any) -> None:
        if self._on_event is None:
            return
        try:
            self._on_event({"server": self.name, "event": event, **payload})
        except Exception:
            logger.debug("MCP event handler failed", exc_info=True)

    def _backoff_delay(self) -> float:
        """Exponential backoff: 0.5s, 1s, 2s, 4s, capped at 8s."""
        return min(8.0, 0.5 * (2**self._failure_count))

    def _record_failure(self, exc: Exception | str) -> None:
        self._failure_count += 1
        self._last_failure = time.monotonic()
        self._next_retry_at = self._last_failure + self._backoff_delay()
        self._unavailable_reason = str(exc)
        self._teardown()
        self._emit("failure", error=str(exc), count=self._failure_count)

    def _resolve_executable(self) -> str | None:
        if not self.command:
            return None
        # Absolute/relative paths are honoured as-is so a config can point at
        # a venv binary that isn't on PATH.
        if os.path.sep in self.command:
            path = Path(self.command).expanduser()
            return str(path) if path.is_file() and os.access(path, os.X_OK) else None
        return shutil.which(self.command)

    # ── process lifecycle ────────────────────────────────────────────

    def _teardown(self) -> None:
        proc, self._process = self._process, None
        self._inbox = None
        self._pending.clear()
        self._initialized = False
        self._tools = None
        self._tools_loaded_at = None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    logger.debug("Failed to kill MCP server %s", self.name)

    def _pump_stdout(
        self, proc: subprocess.Popen[str], inbox: queue.Queue[dict[str, Any] | None]
    ) -> None:
        """Parse every stdout line into the inbox; ``None`` marks EOF."""
        stream = proc.stdout
        if stream is None:
            inbox.put(None)
            return
        try:
            for line in stream:
                line = line.strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    # Servers occasionally print banners on stdout before
                    # the JSON-RPC stream starts. Log and keep reading.
                    logger.debug("MCP %s: non-JSON stdout: %s", self.name, line[:200])
                    continue
                if isinstance(message, dict):
                    inbox.put(message)
        except Exception:
            logger.debug("MCP %s stdout pump ended", self.name, exc_info=True)
        finally:
            inbox.put(None)

    def _pump_stderr(self, proc: subprocess.Popen[str]) -> None:
        stream = proc.stderr
        if stream is None:
            return
        try:
            for line in stream:
                line = line.rstrip()
                if not line:
                    continue
                self._stderr_lines.append(line)
                logger.debug("MCP %s: %s", self.name, line[:400])
        except Exception:
            logger.debug("MCP %s stderr pump ended", self.name, exc_info=True)

    def _spawn(self) -> subprocess.Popen[str] | None:
        exe = self._resolve_executable()
        if exe is None:
            self._unavailable_reason = f"{self.command!r} not found on PATH"
            return None
        env = os.environ.copy()
        env.update(self.env)
        try:
            proc = subprocess.Popen(
                [exe, *self.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=self.cwd,
                env=env,
            )
        except OSError as exc:
            logger.warning("Failed to start MCP server %s: %s", self.name, exc)
            self._record_failure(exc)
            return None

        inbox: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self._process = proc
        self._inbox = inbox
        self._pending.clear()
        self._initialized = False
        threading.Thread(
            target=self._pump_stdout,
            args=(proc, inbox),
            name=f"mcp-{self.name}-stdout",
            daemon=True,
        ).start()
        threading.Thread(
            target=self._pump_stderr,
            args=(proc,),
            name=f"mcp-{self.name}-stderr",
            daemon=True,
        ).start()
        self._emit("started", pid=proc.pid)
        return proc

    def _ensure_process(self) -> subprocess.Popen[str] | None:
        """Return a live, initialized process — spawning/handshaking if needed."""
        with self._lock:
            if self.running and self._initialized:
                return self._process
            if self.running and not self._initialized:
                # Spawned but the handshake failed earlier: retry it.
                if self._handshake():
                    return self._process
                return None
            if (
                self._next_retry_at is not None
                and time.monotonic() < self._next_retry_at
            ):
                return None
            if self._spawn() is None:
                return None
            if not self._handshake():
                return None
            self._failure_count = 0
            self._next_retry_at = None
            self._unavailable_reason = ""
            return self._process

    def _handshake(self) -> bool:
        """Run ``initialize`` + ``notifications/initialized``."""
        timeout = self._startup_timeout
        if timeout is None:
            timeout = float(getattr(settings, "MCP_STARTUP_TIMEOUT", 60) or 60)
        try:
            result = self._rpc(
                "initialize",
                {
                    "protocolVersion": PROTOCOL_VERSION,
                    "capabilities": {},
                    "clientInfo": CLIENT_INFO,
                },
                timeout=timeout,
            )
        except McpUnavailable as exc:
            detail = "; ".join(self.stderr_tail(3))
            self._record_failure(f"{exc}{f' — {detail}' if detail else ''}")
            return False
        self._server_info = dict(result.get("serverInfo", {}) or {})
        self._initialized = True
        try:
            self._notify("notifications/initialized", {})
        except McpUnavailable:
            # Some servers close the notification path; the session is
            # already usable, so a failure here is not fatal.
            logger.debug("MCP %s: initialized notification failed", self.name)
        self._emit("ready", server_info=self._server_info)
        return True

    # ── JSON-RPC plumbing ────────────────────────────────────────────

    def _write(self, payload: dict[str, Any]) -> None:
        proc = self._process
        if proc is None or proc.stdin is None:
            raise McpUnavailable(f"MCP server {self.name!r} is not running")
        try:
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()
        except Exception as exc:  # BrokenPipeError, ValueError on closed pipe
            raise McpUnavailable(
                f"MCP server {self.name!r} write failed: {exc}"
            ) from exc

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _rpc(
        self, method: str, params: dict[str, Any], *, timeout: float
    ) -> dict[str, Any]:
        """Send a request and wait for the response with a matching ``id``."""
        inbox = self._inbox
        if inbox is None:
            raise McpUnavailable(f"MCP server {self.name!r} is not running")
        request_id = self._next_id
        self._next_id += 1
        self._write(
            {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        )

        deadline = time.monotonic() + max(1.0, timeout)
        while True:
            cached = self._pending.pop(request_id, None)
            if cached is not None:
                return self._unwrap(cached)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise McpUnavailable(
                    f"MCP server {self.name!r} timed out after {timeout:.0f}s "
                    f"on {method}"
                )
            try:
                message = inbox.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                if not self.running:
                    raise McpUnavailable(
                        f"MCP server {self.name!r} exited "
                        f"(rc={self._process.returncode if self._process else '?'})"
                    ) from None
                continue
            if message is None:
                detail = "; ".join(self.stderr_tail(3))
                raise McpUnavailable(
                    f"MCP server {self.name!r} closed the connection"
                    + (f" — {detail}" if detail else "")
                )
            message_id = message.get("id")
            if message_id is None:
                # A notification from the server: nothing to correlate.
                continue
            if message_id == request_id:
                return self._unwrap(message)
            # Late response to an abandoned request — keep it in case a
            # caller is still waiting, but never grow without bound.
            if len(self._pending) < 32:
                self._pending[message_id] = message

    def _unwrap(self, message: dict[str, Any]) -> dict[str, Any]:
        error = message.get("error")
        if error:
            if isinstance(error, dict):
                raise McpUnavailable(
                    f"{self.name}: {error.get('message', error)} "
                    f"(code {error.get('code', '?')})"
                )
            raise McpUnavailable(f"{self.name}: {error}")
        result = message.get("result")
        return result if isinstance(result, dict) else {}

    def _request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Public request path: ensures the session, then round-trips."""
        proc = self._ensure_process()
        if proc is None:
            raise McpUnavailable(
                f"MCP server {self.name!r} is not available"
                + (f": {self._unavailable_reason}" if self._unavailable_reason else "")
            )
        timeout = self._timeout
        if timeout is None:
            timeout = float(getattr(settings, "MCP_REQUEST_TIMEOUT", 30) or 30)
        try:
            return self._rpc(method, params, timeout=float(timeout))
        except McpUnavailable:
            # A dead session must not poison every later call: tear it down
            # so the next request re-spawns (subject to backoff).
            if not self.running:
                self._record_failure(f"{method} lost the session")
            raise

    # ── high-level API ───────────────────────────────────────────────

    def list_tools(self, *, force: bool = False) -> list[McpTool]:
        """Return the server's tools, cached with an optional TTL."""
        now = time.monotonic()
        ttl = self._tool_ttl
        if not force and self._tools is not None:
            if ttl is None or (
                self._tools_loaded_at is not None and (now - self._tools_loaded_at) < ttl
            ):
                return self._tools
        result = self._request("tools/list", {})
        tools = [
            McpTool(
                server=self.name,
                name=str(entry.get("name", "")),
                description=str(entry.get("description", "") or "").strip(),
                input_schema=dict(entry.get("inputSchema", {}) or {}),
                annotations=dict(entry.get("annotations", {}) or {}),
            )
            for entry in result.get("tools", [])
            if entry.get("name")
        ]
        self._tools = tools
        self._tools_loaded_at = time.monotonic()
        return tools

    def call_tool(self, tool: str, arguments: dict[str, Any]) -> McpToolResult:
        """Invoke *tool* and return its normalised result."""
        self._emit("tool_call", tool=tool, arguments=arguments)
        started = time.monotonic()
        try:
            raw = self._request(
                "tools/call", {"name": tool, "arguments": dict(arguments or {})}
            )
        except McpUnavailable as exc:
            self._emit("tool_error", tool=tool, error=str(exc))
            raise
        result = parse_tool_result(raw)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        if result.is_error:
            self._emit("tool_error", tool=tool, error=result.text[:500])
        else:
            self._emit("tool_result", tool=tool, elapsed_ms=elapsed_ms)
        return result

    def invalidate_tools_cache(self) -> None:
        """Drop the cached tools so the next list_tools call refreshes."""
        self._tools = None
        self._tools_loaded_at = None

    def close(self) -> None:
        with self._lock:
            self._teardown()


def parse_tool_result(raw: dict[str, Any]) -> McpToolResult:
    """Normalise an MCP ``tools/call`` payload into text + structured data."""
    is_error = bool(raw.get("isError"))
    chunks: list[str] = []
    for block in raw.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            chunks.append(str(block.get("text", "")))
    text = "\n".join(chunk for chunk in chunks if chunk)

    data: Any = None
    structured = raw.get("structuredContent")
    if isinstance(structured, dict):
        # SDK servers wrap non-dict return values as {"result": ...}.
        data = structured.get("result", structured) if structured else None
    if data is None and chunks:
        # Each content block is one JSON document for list-returning tools.
        decoded = []
        for chunk in chunks:
            try:
                decoded.append(json.loads(chunk))
            except (json.JSONDecodeError, TypeError):
                decoded = []
                break
        if decoded:
            data = decoded[0] if len(decoded) == 1 else decoded
    return McpToolResult(is_error=is_error, text=text, data=data)


class McpManager:
    """Owns the configured servers and dispatches tool calls to them."""

    def __init__(
        self,
        servers: dict[str, dict[str, Any]] | None = None,
        *,
        on_event: Callable[[dict[str, Any]], None] | None = None,
        default_tool_ttl: float | None = None,
    ) -> None:
        self._servers: dict[str, McpServerClient] = {}
        self._on_event = on_event
        self._default_tool_ttl = default_tool_ttl
        for name, config in (servers or {}).items():
            self.add_server(name, config)

    def add_server(self, name: str, config: dict[str, Any]) -> McpServerClient:
        client = McpServerClient(
            name=name,
            command=str(config.get("command", "")),
            args=list(config.get("args", []) or []),
            env=dict(config.get("env", {}) or {}),
            cwd=config.get("cwd"),
            timeout=config.get("timeout"),
            startup_timeout=config.get("startup_timeout"),
            tool_ttl=config.get("tool_ttl", self._default_tool_ttl),
            on_event=self._on_event,
        )
        self._servers[name] = client
        return client

    def set_event_handler(
        self, handler: Callable[[dict[str, Any]], None] | None
    ) -> None:
        """Route every server's events to *handler* (used by the UI bridge)."""
        self._on_event = handler
        for client in self._servers.values():
            client._on_event = handler

    def names(self) -> list[str]:
        """Servers whose executable exists (i.e. worth trying)."""
        return [name for name, client in self._servers.items() if client.available]

    def all_names(self) -> list[str]:
        return list(self._servers.keys())

    def status(self) -> dict[str, dict[str, Any]]:
        """Per-server health, for the `/mcp` panel."""
        return {
            name: {
                "available": client.available,
                "running": client.running,
                "initialized": client._initialized,
                "failure_count": client._failure_count,
                "next_retry_at": client._next_retry_at,
                "tools_loaded": len(client._tools) if client._tools else 0,
                "reason": client.unavailable_reason,
                "command": " ".join([client.command, *client.args]).strip(),
                "stderr": client.stderr_tail(3),
            }
            for name, client in self._servers.items()
        }

    def get(self, name: str) -> McpServerClient | None:
        return self._servers.get(name)

    def list_tools(self, server: str | None = None) -> list[McpTool]:
        """List tools across every (or one) reachable server, never raising."""
        tools: list[McpTool] = []
        targets = [server] if server else self.all_names()
        for name in targets:
            client = self._servers.get(name)
            if client is None or not client.available:
                continue
            try:
                tools.extend(client.list_tools())
            except McpUnavailable as exc:
                logger.debug("MCP %s tools/list failed: %s", name, exc)
        return tools

    def call(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> McpToolResult:
        client = self._servers.get(server)
        if client is None:
            raise McpUnavailable(f"MCP server {server!r} is not registered")
        return client.call_tool(tool, arguments)

    def try_call(
        self, server: str, tool: str, arguments: dict[str, Any]
    ) -> McpToolResult | None:
        """Call a tool, returning ``None`` instead of raising when unreachable."""
        try:
            result = self.call(server, tool, arguments)
        except McpUnavailable as exc:
            logger.debug("MCP %s.%s unavailable: %s", server, tool, exc)
            return None
        if result.is_error:
            logger.debug("MCP %s.%s errored: %s", server, tool, result.text[:200])
            return None
        return result

    def warmup(self, server: str | None = None) -> None:
        """Spawn + handshake servers ahead of first use (non-blocking)."""
        for name in [server] if server else self.all_names():
            client = self._servers.get(name)
            if client is None or not client.available or client.running:
                continue
            threading.Thread(
                target=self._warmup_one,
                args=(client,),
                name=f"mcp-{name}-warmup",
                daemon=True,
            ).start()

    @staticmethod
    def _warmup_one(client: McpServerClient) -> None:
        try:
            client.list_tools()
        except McpUnavailable as exc:
            logger.debug("MCP %s warmup failed: %s", client.name, exc)

    def start(self, name: str) -> bool:
        """Start (or restart) the named MCP server."""
        client = self._servers.get(name)
        if client is None:
            return False
        client.close()
        client._failure_count = 0
        client._next_retry_at = None
        return client._ensure_process() is not None

    def stop(self, name: str) -> bool:
        """Stop the named MCP server (closes its subprocess)."""
        client = self._servers.get(name)
        if client is None:
            return False
        client.close()
        return True

    def restart(self, name: str) -> bool:
        """Stop and start the named MCP server."""
        self.stop(name)
        return self.start(name)

    def close(self) -> None:
        for client in self._servers.values():
            client.close()
        self._servers.clear()


# ── configuration ─────────────────────────────────────────────────────────


def _config_candidates() -> list[Path]:
    return [
        Path.cwd() / ".mcp.json",
        Path.cwd() / ".infinidev" / "mcp.json",
        Path(get_base_dir()) / "mcp.json",
    ]


def load_mcp_config() -> dict[str, dict[str, Any]]:
    """Load MCP server config from the first candidate file that parses.

    Later files *extend* earlier ones: a workspace ``.mcp.json`` wins for
    servers it declares, while the user-level file still contributes the
    servers the workspace doesn't mention.
    """
    merged: dict[str, dict[str, Any]] = {}
    for path in _config_candidates():
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("MCP config %s is not usable: %s", path, exc)
            continue
        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            continue
        for name, config in servers.items():
            if not isinstance(config, dict):
                continue
            merged.setdefault(str(name), dict(config))
    return merged


DEFAULT_SERVERS: dict[str, dict[str, Any]] = {
    "ken": {"command": "ken", "args": ["mcp"], "tool_ttl": 300.0},
}


def resolve_mcp_servers() -> dict[str, dict[str, Any]]:
    """Config on disk, with Ken as the default server when none is declared."""
    if not getattr(settings, "MCP_ENABLED", True):
        return {}
    servers = load_mcp_config() if getattr(settings, "MCP_AUTOLOAD_CONFIG", True) else {}
    if not servers:
        return dict(DEFAULT_SERVERS)
    for name, config in DEFAULT_SERVERS.items():
        servers.setdefault(name, dict(config))
    return servers


_default_manager: McpManager | None = None
_manager_lock = threading.Lock()


def get_default_mcp_manager() -> McpManager:
    """Return the process-wide MCP manager built from the resolved config."""
    global _default_manager
    with _manager_lock:
        if _default_manager is None:
            _default_manager = McpManager(resolve_mcp_servers())
        return _default_manager


def reset_default_mcp_manager() -> None:
    """Close and drop the cached manager — used by tests and `/mcp restart`."""
    global _default_manager
    with _manager_lock:
        if _default_manager is not None:
            _default_manager.close()
        _default_manager = None


# ── Keyword fallback used when no MCP server can answer ───────────────────

_TOKEN_RE = re.compile(r"\w+")

_FALLBACK_SKIP_DIRS = {
    ".git",
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "target",
}

_FALLBACK_MAX_FILES = 400
_FALLBACK_MAX_BYTES = 512 * 1024


def keyword_search(
    query: str,
    limit: int = 10,
    *,
    kinds: Iterable[str] | None = None,
    root: Path | None = None,
) -> list[McpHit]:
    """Token-frequency search over memory notes and source files.

    Deliberately dumb but dependency-free: it exists so a workspace with
    no Ken index still gets *something* back instead of an empty list.
    """
    tokens = [token.lower() for token in _TOKEN_RE.findall(query or "") if token]
    if not tokens:
        return []
    kind_set = set(kinds) if kinds is not None else {"memory", "files"}
    corpus: list[tuple[str, str]] = []  # (target, text)

    if "memory" in kind_set:
        try:
            memory_dir = Path(get_base_dir()) / "memory"
            if memory_dir.is_dir():
                for entry in sorted(memory_dir.glob("*.md")):
                    corpus.append((str(entry), entry.read_text(errors="ignore")))
        except Exception:
            logger.debug("fallback memory scan failed", exc_info=True)

    if "files" in kind_set:
        try:
            base = root or Path.cwd()
            count = 0
            for path in base.rglob("*"):
                if count >= _FALLBACK_MAX_FILES:
                    break
                if not path.is_file():
                    continue
                if any(part in _FALLBACK_SKIP_DIRS for part in path.parts):
                    continue
                if path.suffix not in {
                    ".py",
                    ".js",
                    ".ts",
                    ".tsx",
                    ".jsx",
                    ".go",
                    ".rs",
                    ".java",
                    ".rb",
                    ".md",
                    ".toml",
                    ".json",
                }:
                    continue
                try:
                    if path.stat().st_size > _FALLBACK_MAX_BYTES:
                        continue
                    corpus.append((str(path), path.read_text(errors="ignore")))
                except OSError:
                    continue
                count += 1
        except Exception:
            logger.debug("fallback file scan failed", exc_info=True)

    scored: list[McpHit] = []
    for target, text in corpus:
        haystack = text.lower()
        score = sum(haystack.count(token) for token in tokens)
        if not score:
            continue
        line = 0
        snippet = text[:200]
        for index, raw_line in enumerate(text.splitlines(), start=1):
            lowered = raw_line.lower()
            if any(token in lowered for token in tokens):
                line = index
                snippet = raw_line.strip()[:200]
                break
        scored.append(
            McpHit(
                target=target,
                snippet=snippet,
                score=float(score),
                source="fallback",
                line=line,
            )
        )
    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored[:limit]
