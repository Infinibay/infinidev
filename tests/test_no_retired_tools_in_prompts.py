"""No prompt may name a tool that was retired from the surface.

This drift is silent and expensive. When the editing cluster collapsed from
nine tools to ``edit_file``, twenty-odd files kept telling the model to call
``replace_lines`` — including the ``unknown_tool`` guidance entry whose whole
job is to get a model out of a loop of calling tools that don't exist.

The test derives both sides rather than hardcoding either: the retired names
come from ``_RETIRED_TOOLS`` (the dispatcher's own list) and the prompt text
comes from walking the modules, so retiring the next tool makes this fail on
its own without anyone remembering to update a list here.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
import re

import pytest

from infinidev.engine.tool_dispatch import _RETIRED_TOOLS


# Modules whose module-level strings become prompt text. Walked as packages
# so a new prompt file is covered the day it is added.
PROMPT_PACKAGES = [
    "infinidev.prompts",
    "infinidev.engine.loop.prompt",
    "infinidev.engine.guidance",
]

# Two modules classify names the model *emitted* rather than telling it what
# to call, so a retired name is load-bearing there: ``tool_dispatch`` holds
# the message explaining the tool is gone, and ``fingerprint`` needs a stable
# letter for the dead name so "model keeps calling a retired tool" reads as a
# repeating pattern instead of an unfingerprintable gap.
EXEMPT_MODULES = {
    "infinidev.engine.tool_dispatch",
    "infinidev.engine.guidance.fingerprint",
}


def _iter_prompt_modules():
    for pkg_name in PROMPT_PACKAGES:
        pkg = importlib.import_module(pkg_name)
        yield pkg_name, pkg
        if not hasattr(pkg, "__path__"):
            continue
        for mod in pkgutil.walk_packages(pkg.__path__, prefix=f"{pkg_name}."):
            if mod.name in EXEMPT_MODULES:
                continue
            try:
                yield mod.name, importlib.import_module(mod.name)
            except Exception:  # pragma: no cover - optional deps
                continue


def _module_strings(mod) -> list[tuple[str, str]]:
    """Return ``(attr_name, text)`` for every module-level string."""
    out = []
    for name, value in vars(mod).items():
        if name.startswith("__"):
            continue
        if isinstance(value, str):
            out.append((name, value))
        # Lists and tuples hold prompt fragments; sets hold tool-name
        # lookups, which name a retired tool legitimately.
        elif isinstance(value, (list, tuple)):
            out.extend(
                (name, v) for v in value if isinstance(v, str)
            )
        elif isinstance(value, dict):
            out.extend(
                (name, v) for v in value.values() if isinstance(v, str)
            )
    return out


@pytest.mark.parametrize("retired", sorted(_RETIRED_TOOLS))
def test_no_prompt_module_names_a_retired_tool(retired: str) -> None:
    word = re.compile(rf"\b{re.escape(retired)}\b")
    offenders: list[str] = []

    for mod_name, mod in _iter_prompt_modules():
        for attr, text in _module_strings(mod):
            if word.search(text):
                offenders.append(f"{mod_name}.{attr}")

    assert not offenders, (
        f"{retired!r} was retired but is still named in prompt text: "
        + ", ".join(sorted(set(offenders)))
        + f". Rewrite those around the live tool — see _RETIRED_TOOLS"
        f'[{retired!r}]: "{_RETIRED_TOOLS[retired]}"'
    )


@pytest.mark.parametrize("retired", sorted(_RETIRED_TOOLS))
def test_no_prompt_source_names_a_retired_tool(retired: str) -> None:
    """Also catch names inside f-strings and builder functions.

    ``_module_strings`` only sees strings that survive to module level. The
    editing rules and examples are assembled inside functions, so their text
    exists only at call time — reading the source catches those too.
    """
    word = re.compile(rf"\b{re.escape(retired)}\b")
    offenders: list[str] = []

    for mod_name, mod in _iter_prompt_modules():
        try:
            src = inspect.getsource(mod)
        except (OSError, TypeError):  # pragma: no cover
            continue
        for i, line in enumerate(src.splitlines(), 1):
            if word.search(line):
                offenders.append(f"{mod_name}:{i}")

    assert not offenders, (
        f"{retired!r} was retired but still appears in prompt-module source: "
        + ", ".join(sorted(set(offenders)))
    )
