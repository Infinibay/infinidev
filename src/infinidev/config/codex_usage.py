"""Read ChatGPT quota through Codex's documented app-server protocol."""

from __future__ import annotations

import json
import os
import selectors
import shutil
import subprocess
import time


def read_codex_limits(timeout: float = 10) -> dict:
    """Initialize a short-lived read-only client, fetch limits, then reap it."""
    executable = shutil.which("codex")
    if not executable:
        raise RuntimeError("Install the Codex CLI and sign in with codex login to read quota.")
    proc = subprocess.Popen([executable, "app-server"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
    deadline = time.monotonic() + timeout
    buffered = bytearray()
    total = 0
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)

    def send(message: dict) -> None:
        proc.stdin.write((json.dumps(message) + "\n").encode())
        proc.stdin.flush()

    def receive(request_id: int) -> dict:
        nonlocal total
        while time.monotonic() < deadline:
            while b"\n" in buffered:
                line, _, remaining = buffered.partition(b"\n")
                buffered[:] = remaining
                try:
                    message = json.loads(line)
                except (ValueError, UnicodeError):
                    continue
                if not isinstance(message, dict) or message.get("id") != request_id:
                    continue
                if "error" in message:
                    raise RuntimeError("Codex could not read quota. Check codex login and CLI version.")
                result = message.get("result")
                if not isinstance(result, dict):
                    raise RuntimeError("Codex returned an invalid quota response.")
                return result
            if not selector.select(max(0, deadline - time.monotonic())):
                break
            chunk = os.read(proc.stdout.fileno(), 65536)
            if not chunk:
                raise RuntimeError("Codex app-server closed before returning quota.")
            total += len(chunk)
            if total > 1024 * 1024:
                raise RuntimeError("Codex quota response exceeded the read limit.")
            buffered.extend(chunk)
        raise TimeoutError("Codex quota query timed out.")

    try:
        send({"id": 1, "method": "initialize", "params": {
            "clientInfo": {"name": "infinidev", "title": "Infinidev", "version": "1"},
        }})
        receive(1)
        send({"method": "initialized", "params": {}})
        send({"id": 2, "method": "account/rateLimits/read"})
        return receive(2)
    finally:
        selector.close()
        proc.stdin.close()
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1)
        proc.stdout.close()
