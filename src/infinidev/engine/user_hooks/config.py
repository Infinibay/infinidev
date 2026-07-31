"""Reading ``hooks.json`` — what the user declared, and where.

The file is a plain JSON object mapping an event name to the list of
hooks bound to it::

    {
      "hooks": {
        "step_end_instruction": [
          {"prompt": "Review the diff you just produced and fix what is wrong."}
        ],
        "task_end_summary": [
          {"command": "git diff --stat", "timeout": 30}
        ]
      }
    }

A hook declares **either** ``command`` (a shell command whose stdout is
the hook's output) **or** ``prompt`` (fixed text, no process). The prompt
form exists because the common case — injecting one fixed instruction —
should not pay for a subprocess, and because ``echo`` quoting is a bad
first experience for a feature people configure by hand.

Two files are consulted, workspace first::

    <cwd>/.infinidev/hooks.json      the project's hooks
    ~/.infinidev/hooks.json          the user's hooks, for every project

They merge **per event, not per entry**: the first file that declares an
event owns it outright. Concatenating instead would mean a user-level
hook silently firing inside a project that already configured that event
— surprising, and impossible to turn off from the project. Declaring an
event as ``[]`` is therefore meaningful: it switches the global hooks off
for that event here.

The top-level ``"hooks"`` wrapper is optional; a file whose keys are
already event names is accepted as-is, since that is what people write.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infinidev.engine.user_hooks.events import UserHookEvent

logger = logging.getLogger(__name__)

#: Fallback per-hook timeout, overridable per hook and via settings.
DEFAULT_TIMEOUT_SECONDS = 60.0


@dataclass(frozen=True)
class HookSpec:
    """One configured hook: what to run, and how long to wait for it."""

    event: UserHookEvent
    command: str = ""
    prompt: str = ""
    timeout: float = DEFAULT_TIMEOUT_SECONDS
    name: str = ""
    enabled: bool = True

    @property
    def is_literal(self) -> bool:
        """True when the hook is fixed text and no process needs spawning."""
        return not self.command and bool(self.prompt)

    def label(self) -> str:
        """Short identifier for logs — the user's name, else the payload."""
        if self.name:
            return self.name
        source = self.command or self.prompt
        return source[:60] + ("…" if len(source) > 60 else "")


def _config_candidates() -> list[Path]:
    """Config files in precedence order, workspace before user."""
    from infinidev.config.settings import get_base_dir

    candidates = [Path(get_base_dir()) / "hooks.json"]
    user_level = Path.home() / ".infinidev" / "hooks.json"
    # get_base_dir() is cwd/.infinidev, so running Infinidev from the home
    # directory would otherwise read the same file twice and log about it.
    if user_level not in candidates:
        candidates.append(user_level)
    return candidates


def _event_map(data: Any, source: Path) -> dict[UserHookEvent, list[Any]]:
    """Pull the event → raw-entries mapping out of a parsed config file."""
    if not isinstance(data, dict):
        logger.warning("hooks config %s is not a JSON object — ignored", source)
        return {}
    section = data.get("hooks", data)
    if not isinstance(section, dict):
        logger.warning("hooks config %s has a non-object 'hooks' key", source)
        return {}

    result: dict[UserHookEvent, list[Any]] = {}
    for raw_name, entries in section.items():
        event = UserHookEvent.parse(raw_name)
        if event is None:
            logger.warning(
                "hooks config %s declares unknown event %r — ignored",
                source, raw_name,
            )
            continue
        if entries is None:
            entries = []
        # A bare object or string is accepted where a list is expected;
        # writing one hook as a list of one is a papercut, not a contract.
        if not isinstance(entries, list):
            entries = [entries]
        result[event] = entries
    return result


def _build_spec(
    event: UserHookEvent, entry: Any, source: Path, default_timeout: float,
) -> HookSpec | None:
    """Turn one config entry into a :class:`HookSpec`, or reject it loudly."""
    if isinstance(entry, str):
        entry = {"prompt": entry}
    if not isinstance(entry, dict):
        logger.warning(
            "hooks config %s: %s entry is neither an object nor a string",
            source, event.value,
        )
        return None

    command = str(entry.get("command") or "").strip()
    prompt = str(entry.get("prompt") or "").strip()
    if command and prompt:
        logger.warning(
            "hooks config %s: %s hook declares both 'command' and 'prompt' "
            "— using 'command'", source, event.value,
        )
        prompt = ""
    if not command and not prompt:
        logger.warning(
            "hooks config %s: %s hook declares neither 'command' nor 'prompt'",
            source, event.value,
        )
        return None

    try:
        timeout = float(entry.get("timeout", default_timeout))
    except (TypeError, ValueError):
        timeout = default_timeout
    if timeout <= 0:
        timeout = default_timeout

    return HookSpec(
        event=event,
        command=command,
        prompt=prompt,
        timeout=timeout,
        name=str(entry.get("name") or "").strip(),
        enabled=bool(entry.get("enabled", True)),
    )


def load_hooks_config() -> dict[UserHookEvent, list[HookSpec]]:
    """Read every candidate file and return the resolved hooks per event.

    Never raises. A malformed file costs the user their hooks and a
    warning in the log, not their session.
    """
    from infinidev.config.settings import settings

    default_timeout = float(
        getattr(settings, "HOOKS_TIMEOUT", DEFAULT_TIMEOUT_SECONDS)
    )

    resolved: dict[UserHookEvent, list[HookSpec]] = {}
    for path in _config_candidates():
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("hooks config %s is not usable: %s", path, exc)
            continue
        for event, entries in _event_map(data, path).items():
            if event in resolved:
                continue  # first file to declare the event owns it
            specs = [
                spec
                for entry in entries
                if (spec := _build_spec(event, entry, path, default_timeout))
                is not None and spec.enabled
            ]
            resolved[event] = specs
    return {event: specs for event, specs in resolved.items() if specs}


def _config_fingerprint() -> tuple:
    """(path, mtime, size) for each existing config file.

    Cheap enough to stat on every step, and it lets an edit to
    ``hooks.json`` take effect without restarting — the same promise
    ``settings.json`` already makes.
    """
    stamps: list[tuple] = []
    for path in _config_candidates():
        try:
            stat = path.stat()
        except OSError:
            continue
        stamps.append((str(path), stat.st_mtime_ns, stat.st_size))
    return tuple(stamps)


_cache: dict[UserHookEvent, list[HookSpec]] | None = None
_cache_key: tuple | None = None


def get_hooks(event: UserHookEvent) -> list[HookSpec]:
    """Hooks bound to ``event``, reloading the config when it changed."""
    from infinidev.config.settings import settings

    if not getattr(settings, "HOOKS_ENABLED", True):
        return []

    global _cache, _cache_key
    key = _config_fingerprint()
    if _cache is None or key != _cache_key:
        _cache = load_hooks_config()
        _cache_key = key
    return list(_cache.get(event, ()))


def has_hooks(event: UserHookEvent) -> bool:
    """Whether anything is bound to ``event`` — the cheap pre-check."""
    return bool(get_hooks(event))


def invalidate_cache() -> None:
    """Drop the cached config. For tests and for ``/reload``-style commands."""
    global _cache, _cache_key
    _cache = None
    _cache_key = None
