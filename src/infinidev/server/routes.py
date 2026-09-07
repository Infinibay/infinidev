"""Workspace APIs backed by the existing engine, team store and process manager."""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field, ValidationError

from infinidev.server.runtime import public
from infinidev.tools.base.db import execute_with_retry

api_router = APIRouter()
_SKIP = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".pytest_cache",
         "dist", "build", ".infinidev", "target", ".cache", ".ken", ".mypy_cache"}
_EDITABLE = {"LLM_MODEL", "LLM_PROVIDER", "LLM_API_KEY", "LLM_BASE_URL", "THINKING_BUDGET",
             "THINKING_ENABLED", "THINKING_BUDGET_TOKENS", "LOOP_MAX_ITERATIONS",
             "TEAM_MAX_WORKERS", "TEAM_WORKER_MAX_ITERATIONS"}


def _session(request: Request, session_id: str):
    try:
        return request.app.state.runtime.session(session_id)
    except KeyError:
        raise HTTPException(404, "Session not found in this workspace.") from None


def _path(request: Request, rel: str) -> Path:
    root = request.app.state.runtime.root
    target = (root / rel).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(400, "Path escapes the workspace.")
    if any(part in {".git", ".infinidev"} for part in target.relative_to(root).parts):
        raise HTTPException(403, "Internal runtime files are not editable through the browser.")
    return target


def _idle(request: Request) -> None:
    if any(s.busy or (s.future and not s.future.done())
           for s in request.app.state.runtime.sessions.values()):
        raise HTTPException(409, "Wait for active and queued turns before changing configuration.")


@api_router.get("/info")
def info(request: Request) -> dict:
    from infinidev.config.settings import settings

    runtime = request.app.state.runtime
    return {"cwd": str(runtime.root), "workspace": runtime.root.name, "version": version("infinidev"),
            "model": settings.LLM_MODEL, "provider": settings.LLM_PROVIDER,
            "active_session": runtime.active.session_id if runtime.active else None}


@api_router.get("/sessions")
def sessions(request: Request) -> dict:
    runtime = request.app.state.runtime
    rows = execute_with_retry(lambda conn: [dict(row) for row in conn.execute(
        "SELECT session_id, title, last_active_at, turn_count FROM sessions "
        "WHERE workspace_path = ? ORDER BY last_active_at DESC LIMIT 200", (str(runtime.root),),
    ).fetchall()])
    for row in rows:
        current = runtime.sessions.get(row["session_id"])
        row.update(current.summary() if current else {"status": "idle", "phase": "idle"})
        row["title"] = row.get("title") or "New session"
    return {"sessions": rows}


@api_router.post("/sessions")
def create_session(request: Request) -> dict:
    return request.app.state.runtime.create_session().summary()


@api_router.get("/sessions/{session_id}")
def snapshot(request: Request, session_id: str) -> dict:
    return _session(request, session_id).snapshot()


class Title(BaseModel):
    title: str = Field(min_length=1, max_length=80)


@api_router.patch("/sessions/{session_id}")
def rename(request: Request, session_id: str, body: Title) -> dict:
    from infinidev.db.service import rename_session

    session = _session(request, session_id)
    if not rename_session(session_id, body.title):
        raise HTTPException(400, "A title cannot be empty.")
    session.state(title=" ".join(body.title.split()))
    return session.summary()


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=100000)
    client_id: str = Field(min_length=1, max_length=100)


@api_router.post("/sessions/{session_id}/messages")
def send(request: Request, session_id: str, body: Message) -> dict:
    if not body.text.strip():
        raise HTTPException(400, "A message cannot be empty.")
    return _session(request, session_id).submit(body.text.strip(), body.client_id)


class Answer(BaseModel):
    request_id: str
    text: str | None = Field(default=None, max_length=100000)


@api_router.post("/sessions/{session_id}/answers")
def answer(request: Request, session_id: str, body: Answer) -> dict:
    if not _session(request, session_id).resolve(body.request_id, body.text):
        raise HTTPException(409, "This question was already answered or cancelled.")
    return {"ok": True}


@api_router.post("/sessions/{session_id}/cancel")
def cancel(request: Request, session_id: str) -> dict:
    session = _session(request, session_id)
    session.cancel()
    return session.summary()


@api_router.get("/sessions/{session_id}/team")
def team(request: Request, session_id: str, after: int = Query(0, ge=0)) -> dict:
    from infinidev.engine.team.store import TeamStore

    session = _session(request, session_id)
    key = json.dumps([1, str(request.app.state.runtime.root), session_id])
    store = TeamStore(hashlib.sha256(key.encode()).hexdigest())
    state = store.snapshot()
    live = bool(session.engine and getattr(session.engine, "_team_runtime", None))
    events = store.events(after=after, limit=100)
    for event in events:
        if event["kind"] in {"note", "delegation", "ticket"}:
            try:
                detail = json.loads(event["content"])
            except (TypeError, json.JSONDecodeError):
                continue
            if not isinstance(detail, dict):
                continue
            event["data"] = detail
            if event["kind"] == "note":
                event["content"] = detail.get("text", event["content"])
            elif event["kind"] == "delegation":
                event["content"] = f"Assigned {detail.get('name', 'specialist')} · " + str(
                    detail.get("role", "Specialist"),
                )
            elif event["kind"] == "ticket":
                event["content"] = f"Created ticket: {detail.get('title', '')}"
    return public({"board": state, "live": live, "events": events,
                   "next": events[-1]["id"] if events else after, "has_more": len(events) == 100})


@api_router.get("/processes")
def processes() -> dict:
    from infinidev.tools.shell.background_manager import get_background_manager

    return public({"processes": [{"id": t.id, "description": t.description, "command": t.command,
                                  "cwd": t.cwd, "status": t.status, "exit_code": t.exit_code,
                                  "runtime_seconds": round(t.runtime_seconds(), 1)}
                                 for t in get_background_manager().list()]})


class Note(BaseModel):
    content: str = Field(min_length=1, max_length=20000)


@api_router.post("/sessions/{session_id}/notes")
def add_note(request: Request, session_id: str, body: Note) -> dict:
    from infinidev.engine.team.store import TeamStore

    _session(request, session_id)
    key = json.dumps([1, str(request.app.state.runtime.root), session_id])
    store = TeamStore(hashlib.sha256(key.encode()).hexdigest())
    if not store.snapshot():
        raise HTTPException(409, "Start a team before adding shared notes.")
    content = body.content.strip()
    if not content:
        raise HTTPException(400, "A note cannot be empty.")
    event_id = store.update(lambda state, emit: emit(
        "note", "user", json.dumps({"kind": "observation", "text": content}),
    ))
    return public(store.event(event_id))


@api_router.get("/processes/{task_id}")
def process(task_id: str) -> dict:
    from infinidev.tools.shell.background_manager import get_background_manager

    task = get_background_manager().get(task_id)
    if task is None:
        raise HTTPException(404, "Process not found in this server instance.")
    output, discarded = task.combined_output()
    return public({"id": task.id, "output": output[-512000:], "discarded": discarded,
                   "truncated": len(output) > 512000, "status": task.status,
                   "exit_code": task.exit_code})


@api_router.post("/processes/{task_id}/stop")
def stop_process(task_id: str) -> dict:
    from infinidev.tools.shell.background_manager import get_background_manager

    if get_background_manager().stop(task_id) is None:
        raise HTTPException(404, "Process not found.")
    return {"ok": True}


@api_router.get("/settings")
def get_settings() -> dict:
    from infinidev.config.secrets import is_secret
    from infinidev.config.settings import settings

    values = {key: value for key, value in settings.model_dump().items()
              if key in _EDITABLE and not (is_secret(key) and isinstance(value, str))}
    return public({"values": values, "secrets": {key: bool(getattr(settings, key))
                   for key in _EDITABLE if is_secret(key)
                   and isinstance(getattr(settings, key), str)}, "editable": sorted(_EDITABLE)})


class SettingsPatch(BaseModel):
    updates: dict[str, Any]


@api_router.patch("/settings")
def patch_settings(request: Request, body: SettingsPatch) -> dict:
    from infinidev.config.model_capabilities import _reset_capabilities
    from infinidev.config.providers import get_provider, list_provider_ids
    from infinidev.config.reasoning import effort_profile
    from infinidev.config.settings import Settings, reload_all, settings

    with request.app.state.runtime.lock:
        _idle(request)
        updates = dict(body.updates)
        if not updates or set(updates) - _EDITABLE:
            raise HTTPException(400, "Unsupported setting.")
        provider = updates.get("LLM_PROVIDER", settings.LLM_PROVIDER)
        if provider not in list_provider_ids():
            raise HTTPException(400, "Unknown provider.")
        selected = get_provider(provider)
        if provider != settings.LLM_PROVIDER:
            # A previous provider's credential must never travel to the new endpoint.
            updates.setdefault("LLM_API_KEY", "")
            updates.setdefault("LLM_BASE_URL", selected.default_base_url)
            if "LLM_MODEL" not in updates:
                raise HTTPException(400, "Select a model when changing provider.")
        if "LLM_MODEL" in updates:
            model = str(updates["LLM_MODEL"]).strip()
            if not model:
                raise HTTPException(400, "A model is required.")
            updates["LLM_MODEL"] = (model if model.startswith(selected.prefix)
                                    else selected.prefix + model)
        if "THINKING_BUDGET" in updates:
            profile = effort_profile(provider, updates.get("LLM_MODEL", settings.LLM_MODEL))
            if updates["THINKING_BUDGET"] not in profile.choices:
                raise HTTPException(400, "Effort is not supported by this model and provider.")
            updates["THINKING_ENABLED"] = updates["THINKING_BUDGET"] not in {"off", "none"}
        for key in {"LOOP_MAX_ITERATIONS", "TEAM_MAX_WORKERS", "TEAM_WORKER_MAX_ITERATIONS",
                    "THINKING_BUDGET_TOKENS"} & updates.keys():
            if not isinstance(updates[key], int) or not 1 <= updates[key] <= 1000000:
                raise HTTPException(400, f"Invalid positive integer: {key}")
        try:
            Settings.model_validate({**settings.model_dump(), **updates})
        except ValidationError:
            raise HTTPException(400, "Invalid configuration values.") from None
        settings.save_user_settings(updates)
        reload_all()
        _reset_capabilities()
        mismatched = [key for key, value in updates.items() if getattr(settings, key) != value]
        if mismatched:
            raise HTTPException(409, "Some values are overridden by the environment or could "
                                "not be saved: " + ", ".join(sorted(mismatched)))
        return get_settings()


@api_router.get("/models")
def models(provider: str | None = None, model: str | None = None,
           refresh: bool = False) -> dict:
    from infinidev.config.providers import PROVIDERS, fetch_models
    from infinidev.config.reasoning import effort_profile, resolve_effort
    from infinidev.config.settings import settings

    provider = provider or settings.LLM_PROVIDER
    if provider not in PROVIDERS:
        raise HTTPException(400, "Unknown provider.")
    selected = PROVIDERS[provider]
    model = model or (settings.LLM_MODEL if provider == settings.LLM_PROVIDER else "")
    available = selected.static_models
    error = None
    if refresh:
        try:
            available = fetch_models(provider, api_key=(settings.LLM_API_KEY if
                                     provider == settings.LLM_PROVIDER else ""),
                                     base_url=(settings.LLM_BASE_URL if provider ==
                                     settings.LLM_PROVIDER else selected.default_base_url),
                                     raise_on_error=True)
        except Exception as exc:
            error = str(exc)[:300]
    profile = effort_profile(provider, model) if model else effort_profile(provider, "unknown")
    return public({"current": settings.LLM_MODEL, "provider": provider, "models": available,
                   "providers": [asdict(p) for p in PROVIDERS.values()], "error": error,
                   "effort": {**asdict(profile), "value": resolve_effort(
                       profile, settings.THINKING_BUDGET, enabled=settings.THINKING_ENABLED)}})


@api_router.get("/usage")
def usage() -> dict:
    from infinidev.config.usage import render_usage

    return public({"report": render_usage()})


@api_router.get("/findings")
def findings(limit: int = Query(200, ge=1, le=1000)) -> dict:
    from infinidev.db.service import get_all_findings

    return public({"findings": get_all_findings(project_id=1, limit=limit)})


@api_router.get("/tools")
def tools() -> dict:
    from infinidev.tools import get_tools_for_role

    return {"tools": [{"name": tool.name, "description": tool.description,
                        "read_only": bool(getattr(tool, "is_read_only", False))}
                       for tool in get_tools_for_role("developer", supports_vision=False)]}


@api_router.get("/files/tree")
def files_tree(request: Request, path: str = "") -> dict:
    base = _path(request, path)
    if not base.is_dir():
        raise HTTPException(404, "Directory not found.")
    root = request.app.state.runtime.root
    entries = [{"name": p.name, "path": str(p.relative_to(root)), "is_dir": p.is_dir()}
               for p in sorted(base.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
               if p.name not in _SKIP and p.resolve().is_relative_to(root)]
    return {"path": str(base.relative_to(root)), "entries": entries[:1000]}


@api_router.get("/files/read")
def files_read(request: Request, path: str) -> dict:
    target = _path(request, path)
    if not target.is_file():
        raise HTTPException(404, "File not found.")
    if target.stat().st_size > 2 * 1024 * 1024:
        raise HTTPException(413, "File exceeds the 2 MB browser preview limit.")
    content = target.read_bytes()
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError:
        decoded = None
    return {"path": path, "revision": hashlib.sha256(content).hexdigest(),
            "binary": b"\0" in content or decoded is None,
            "text": "" if b"\0" in content or decoded is None else decoded}


class FileWrite(BaseModel):
    path: str
    text: str = Field(max_length=2000000)
    revision: str


@api_router.put("/files/write")
def files_write(request: Request, body: FileWrite) -> dict:
    from infinidev.tools.base.permissions import check_file_permission

    target = _path(request, body.path)
    if not target.is_file():
        raise HTTPException(404, "File not found.")
    with request.app.state.runtime.file_lock:
        if hashlib.sha256(target.read_bytes()).hexdigest() != body.revision:
            raise HTTPException(409, "The file changed on disk. Reload before saving.")
        error = check_file_permission("edit_file", str(target))
        if error:
            raise HTTPException(403, error)
        if hashlib.sha256(target.read_bytes()).hexdigest() != body.revision:
            raise HTTPException(409, "The file changed while awaiting approval. Reload it.")
        target.write_text(body.text, encoding="utf-8")
    return files_read(request, body.path)


def _git(request: Request, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=request.app.state.runtime.root,
                            capture_output=True, timeout=15)
    if result.returncode:
        raise HTTPException(400, result.stderr.decode(errors="replace")[:400])
    return result.stdout.decode("utf-8", errors="replace")


@api_router.get("/changes")
def changes(request: Request) -> dict:
    raw = _git(request, "status", "--porcelain=v1", "-z", "--untracked-files=normal")
    parts = iter(raw.split("\0"))
    files = []
    for part in parts:
        if len(part) < 4:
            continue
        row = {"status": part[:2], "path": part[3:]}
        if "R" in part[:2] or "C" in part[:2]:
            row["original"] = next(parts, "")
        files.append(row)
    return {"files": files, "branch": _git(request, "branch", "--show-current").strip()}


@api_router.get("/changes/diff")
def diff(request: Request, path: str) -> dict:
    _path(request, path)
    return {"path": path, "unstaged": _git(request, "diff", "--no-ext-diff", "--", path)[:512000],
            "staged": _git(request, "diff", "--cached", "--no-ext-diff", "--", path)[:512000]}
