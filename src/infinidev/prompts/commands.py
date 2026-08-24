"""Shared command support for discovering and toggling optional prompt capabilities."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
from typing import Iterator

from infinidev.prompts._filesystem import fsync_directory
from infinidev.prompts.catalog import (
    USER_OVERRIDES_FILE,
    materialize_starter_prompt_profiles,
)
from infinidev.prompts.optional_capabilities import OPTIONAL_CAPABILITIES
from infinidev.prompts.profiles import (
    EffectivePromptConfiguration,
    PromptProfileError,
    get_prompt_catalog_path,
)


_USER_OVERRIDE_LOCK = Lock()


@contextmanager
def _user_override_file_lock(path: Path) -> Iterator[None]:
    """Serialize override updates across Infinidev processes.

    Atomic replacement prevents torn JSON, but without an advisory lock two
    processes can still read the same old document and overwrite one another's
    changes. The stable sidecar remains in place because locking the replaced
    JSON inode itself would not protect later openers.
    """
    lock_path = path.parent.parent / f".{path.parent.name}.{path.name}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0)
            if lock_file.tell() == lock_file.seek(0, os.SEEK_END):
                lock_file.write(b"0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def handle_prompts_command(parts: list[str]) -> str:
    """Return the user-facing result for ``/prompts [enable|disable] [name]``."""
    try:
        return _handle_prompts_command(parts)
    except (OSError, PromptProfileError, ValueError) as err:
        return f"Prompt configuration error: {err}"


def _handle_prompts_command(parts: list[str]) -> str:
    """Execute a validated prompt command and let expected configuration errors propagate."""
    if len(parts) == 1:
        return _render_capabilities()
    if len(parts) == 2 and parts[1].lower() in {"list", "status"}:
        return _render_capabilities()

    action = parts[1].lower()
    valid_actions = {"enable", "disable", "reset", "search", "show"}
    if action not in valid_actions or len(parts) < 3:
        return (
            "Usage: /prompts [list] | /prompts search <term> | "
            "/prompts show <name> | /prompts enable <name> | "
            "/prompts disable <name> | /prompts reset <name>"
        )

    if action == "search":
        term = " ".join(parts[2:]).strip()
        if not term:
            return (
                "Usage: /prompts [list] | /prompts search <term> | "
                "/prompts show <name> | /prompts enable <name> | "
                "/prompts disable <name> | /prompts reset <name>"
            )
        return _render_capability_search(term)
    if len(parts) != 3:
        return (
            "Usage: /prompts [list] | /prompts search <term> | "
            "/prompts show <name> | /prompts enable <name> | "
            "/prompts disable <name> | /prompts reset <name>"
        )

    name = _normalize_name(parts[2])
    capabilities = dict(OPTIONAL_CAPABILITIES)
    if name not in capabilities:
        return f"Unknown prompt capability: {parts[2]}\nUse /prompts to list available names."
    if action == "show":
        return _render_capability(name, capabilities[name])

    catalog_path = get_prompt_catalog_path()
    materialize_starter_prompt_profiles(catalog_path)
    path = catalog_path / USER_OVERRIDES_FILE
    if action == "reset":
        removed = _write_user_override(path, name, None)
        effective = EffectivePromptConfiguration.compile().resolve(
            "develop", name, default_enabled=False
        ).enabled
        state = "enabled" if effective else "disabled"
        concise_name = name.removeprefix("capability.")
        if not removed:
            return (
                f"Prompt capability {concise_name} already uses its inherited state "
                f"({state})."
            )
        return (
            f"Prompt capability {concise_name} reset to its inherited state "
            f"({state}). Applies from the next task."
        )

    enabled = action == "enable"
    _write_user_override(path, name, enabled)
    effective = EffectivePromptConfiguration.compile().resolve(
        "develop", name, default_enabled=False
    ).enabled
    state = "enabled" if enabled else "disabled"
    message = f"Prompt capability {name.removeprefix('capability.')} {state}."
    if effective != enabled:
        message += (
            " A higher-precedence prompt profile controls the effective state "
            "(for example, a later shared file or the project override)."
        )
    else:
        message += " Applies from the next task."
    return message


def _render_capabilities() -> str:
    """Render all optional capabilities and their effective states."""
    configuration = EffectivePromptConfiguration.compile()
    lines = [
        "Optional prompt capabilities:",
        "  Enable or disable with /prompts; reset removes your override.",
    ]
    for name, body in OPTIONAL_CAPABILITIES:
        enabled = configuration.resolve("develop", name, default_enabled=False).enabled
        marker = "on " if enabled else "off"
        summary = _capability_summary(body)
        lines.append(f"  {marker}  {name.removeprefix('capability.')} — {summary}")
    lines.append(f"\nUser overrides: {get_prompt_catalog_path() / USER_OVERRIDES_FILE}")
    return "\n".join(lines)


def _render_capability_search(term: str) -> str:
    """Render capabilities whose name, condition, or guidance contains ``term``."""
    normalized_term = term.casefold()
    matches = [
        (name, body)
        for name, body in OPTIONAL_CAPABILITIES
        if normalized_term in name.removeprefix("capability.").casefold()
        or any(
            normalized_term in part.casefold()
            for part in _capability_parts(body)
        )
    ]
    if not matches:
        return f"No prompt capabilities match: {term}"

    configuration = EffectivePromptConfiguration.compile()
    lines = [f"Prompt capabilities matching {term!r}:"]
    for name, body in matches:
        enabled = configuration.resolve("develop", name, default_enabled=False).enabled
        marker = "on " if enabled else "off"
        lines.append(
            f"  {marker}  {name.removeprefix('capability.')} — {_capability_summary(body)}"
        )
    return "\n".join(lines)


def _render_capability(name: str, body: str) -> str:
    """Render one capability's condition and guidance for inspection before activation."""
    configuration = EffectivePromptConfiguration.compile()
    enabled = configuration.resolve("develop", name, default_enabled=False).enabled
    reason, guidance = _capability_parts(body)
    state = "enabled" if enabled else "disabled"
    concise_name = name.removeprefix("capability.")
    return (
        f"Prompt capability: {concise_name} ({state})\n"
        f"ID: {name}\n"
        f"Applies when: {reason}\n\n"
        f"{guidance}"
    )


def _capability_summary(body: str) -> str:
    """Extract the catalog entry's activation condition for command discovery."""
    reason, _guidance = _capability_parts(body)
    return reason


def _capability_parts(body: str) -> tuple[str, str]:
    """Extract normalized condition and guidance text from a capability fragment."""
    prefix, separator, remainder = body.partition('reason="')
    if not separator or "<if" not in prefix:
        return "additional workflow guidance", " ".join(body.split())
    reason, separator, content = remainder.partition('">')
    if not separator:
        return "additional workflow guidance", " ".join(body.split())
    guidance, closing, _suffix = content.rpartition("</if>")
    if not closing:
        guidance = content
    return " ".join(reason.split()), " ".join(guidance.split())


def _normalize_name(name: str) -> str:
    """Accept case-insensitive concise names while retaining canonical identifiers."""
    normalized = name.lower()
    return normalized if normalized.startswith("capability.") else f"capability.{normalized}"


def _write_user_override(path: Path, name: str, enabled: bool | None) -> bool:
    """Set or remove one override without losing concurrent process updates.

    Returns whether a reset removed an existing override. Set operations always
    return ``True`` after they are persisted.
    """
    with _USER_OVERRIDE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _user_override_file_lock(path):
            document: dict[str, object] = {}
            if path.exists():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError(f"Prompt override root in {path} must be an object")
                document = loaded

            develop = document.setdefault("develop", {})
            if not isinstance(develop, dict):
                raise ValueError(
                    f"Prompt override 'develop' section in {path} must be an object"
                )
            if enabled is None:
                if name not in develop:
                    return False
                del develop[name]
            else:
                develop[name] = {"enabled": enabled, "enabled_by_default": False}

            content = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
            temporary_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    dir=path.parent,
                    prefix=f".{path.name}.",
                    suffix=".tmp",
                    delete=False,
                ) as temporary_file:
                    temporary_file.write(content)
                    temporary_file.flush()
                    os.fsync(temporary_file.fileno())
                    temporary_path = Path(temporary_file.name)
                temporary_path.replace(path)
                fsync_directory(path.parent)
            finally:
                if temporary_path is not None:
                    temporary_path.unlink(missing_ok=True)
            return True
