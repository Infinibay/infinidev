"""Authenticated, local web harness with reconnectable session streams."""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from infinidev.server.routes import api_router
from infinidev.server.runtime import WebRuntime


def _trusted_origin(origin: str | None, host: str) -> bool:
    if not origin:
        return True
    parsed = urlsplit(origin)
    return parsed.scheme in {"http", "https"} and (
        parsed.netloc == host or origin in {
            "http://127.0.0.1:5173", "http://localhost:5173",
        }
    )


def _trusted_host(host: str) -> bool:
    return urlsplit("http://" + host).hostname in {"127.0.0.1", "localhost", "::1", "testserver"}


def create_app(workdir: str | None = None, *, token: str | None = None,
               initialize: bool = True, static_dir: Path | None = None,
               on_ready: Callable[[], None] | None = None) -> FastAPI:
    """Create one workspace server; CLI must set cwd before importing settings."""
    root = Path(workdir or Path.cwd()).resolve()
    if root != Path.cwd().resolve():
        raise ValueError("Change into the workspace before creating the server.")
    access_token = token or secrets.token_urlsafe(32)

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        from infinidev.db.service import init_db
        from infinidev.server.bootstrap import bootstrap_runtime, shutdown_runtime
        from infinidev.tools.permission import set_permission_handler

        init_db()
        app.state.runtime = WebRuntime(root, asyncio.get_running_loop())
        set_permission_handler(app.state.runtime.permission)
        if initialize:
            await asyncio.to_thread(bootstrap_runtime)
        try:
            if on_ready:
                on_ready()
            yield
        finally:
            app.state.runtime.close()
            set_permission_handler(lambda *_: False)
            if initialize:
                shutdown_runtime()

    app = FastAPI(title="Infinidev", version=version("infinidev"), lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def authenticate(request, call_next):
        host = request.headers.get("host", "")
        if not _trusted_host(host) or not _trusted_origin(request.headers.get("origin"), host):
            return JSONResponse({"detail": "Untrusted browser origin."}, status_code=403)
        if request.url.path.startswith("/api"):
            supplied = request.headers.get("authorization", "").removeprefix("Bearer ")
            if not secrets.compare_digest(supplied, access_token):
                return JSONResponse({"detail": "Open the authenticated launch URL."}, status_code=401)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        if request.url.path.startswith("/api"):
            response.headers["Cache-Control"] = "no-store"
        return response

    app.include_router(api_router, prefix="/api")

    @app.websocket("/ws")
    async def stream(websocket: WebSocket) -> None:
        host = websocket.headers.get("host", "")
        protocols = [p.strip() for p in websocket.headers.get("sec-websocket-protocol", "").split(",")]
        authorized = any(secrets.compare_digest(p, "token." + access_token) for p in protocols)
        if (not authorized or not _trusted_host(host)
                or not _trusted_origin(websocket.headers.get("origin"), host)):
            await websocket.close(code=1008)
            return
        try:
            session = app.state.runtime.session(websocket.query_params.get("session_id", ""))
        except KeyError:
            await websocket.close(code=1008)
            return
        await websocket.accept(subprotocol="infinidev")
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        session.listeners.add(queue)

        async def pump():
            while True:
                envelope = await queue.get()
                if envelope["type"] == "resync":
                    envelope = {"type": "snapshot", "session": session.snapshot()}
                await websocket.send_json(envelope)

        task = None
        try:
            await websocket.send_json({"type": "snapshot", "session": session.snapshot()})
            task = asyncio.create_task(pump())
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            session.listeners.discard(queue)
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, WebSocketDisconnect, RuntimeError):
                    await task

    static = static_dir or Path(__file__).parent / "static"
    if (static / "index.html").is_file():
        app.mount("/", StaticFiles(directory=static, html=True), name="web")
    else:
        @app.get("/", response_class=HTMLResponse)
        def missing_build():
            return """<!doctype html><title>Infinidev · Build the web interface</title>
            <main style="font:16px system-ui;max-width:650px;margin:15vh auto;padding:24px">
            <h1>The workspace server is ready.</h1><p>Build the browser interface from the
            Infinidev checkout, then restart this server:</p>
            <pre>cd web\nnpm ci\nnpm run build</pre>
            <p>For development, run <code>npm run dev</code> and open the Vite address
            with the same <code>#token=…</code> fragment from the launch URL.</p></main>"""
    return app
