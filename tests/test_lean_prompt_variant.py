"""The `lean` prompt style, and the payload it removes.

The loop rebuilds the whole prompt on every model round, so every character of
the static prefix is paid again per round. The measurements here are exact
rather than statistical: they compare the strings the engine assembles, which
is the part of "fewer tokens" that does not depend on how a model behaves.

What the style must not do is drop a rule. ``tests/test_prompt_style_rules.py``
enforces the language rules over the same files; these tests enforce that the
compact statement still says each thing the verbose one said.
"""

from __future__ import annotations

from infinidev.config.settings import settings
from infinidev.engine.loop.context import build_system_prompt
from infinidev.engine.loop.prompt.text import BEHAVIOR_GUIDELINES
from infinidev.prompts.flows.develop import _DEVELOP_IDENTITY_BASE, get_develop_identity
from infinidev.prompts.profiles import EffectivePromptConfiguration
from infinidev.prompts.variants import get_variant
from infinidev.tools import get_tools_for_role


def _flatten(text: str) -> str:
    """Casefold and collapse whitespace so a wrapped phrase still matches."""
    return " ".join(text.casefold().split())


def _system_prompt(style: str) -> str:
    original = settings.PROMPT_STYLE
    settings.PROMPT_STYLE = style
    try:
        tools = {tool.name for tool in get_tools_for_role("developer")}
        return build_system_prompt(
            "backstory",
            identity_override=get_develop_identity(tools),
            workspace_path="/tmp",
            prompt_configuration=EffectivePromptConfiguration.compile(),
        )
    finally:
        settings.PROMPT_STYLE = original


def test_lean_cuts_a_quarter_of_the_static_system_prompt() -> None:
    # Against `generalized` explicitly, not "auto": since 0.29.0 "auto" *is*
    # lean, so comparing them would assert that a thing is smaller than itself.
    default = _system_prompt("generalized")
    lean = _system_prompt("lean")

    assert len(lean) < len(default) * 0.8, (
        f"the lean style was supposed to remove at least a fifth of the static "
        f"prefix and produced {len(lean)} against {len(default)} characters"
    )


def test_lean_is_the_only_style_that_changes_the_develop_core() -> None:
    for style in ("auto", "full", "generalized", "coding", "extra_simple"):
        assert get_variant("flow.develop.core", style) is None, (
            f"{style} must keep the original develop core; a second copy of the "
            "engineering rules drifting from the first is worse than the bytes"
        )
    assert get_variant("flow.develop.core", "lean")
    assert get_variant("loop.behavior_guidelines", "lean")


def test_the_compact_core_keeps_every_engineering_rule() -> None:
    """Shorter, not thinner: each rule survives in some wording."""
    lean = get_variant("flow.develop.core", "lean") or ""
    lowered = _flatten(lean)

    for rule, marker in (
        ("read before editing", "before you change them"),
        ("follow existing patterns", "pattern the project already uses"),
        ("implement only what was asked", "do not add unrelated features"),
        ("report what you do not fix", "leave it alone"),
        ("regression test for changed behavior", "regression test"),
        ("readability over cleverness", "clever trick"),
        ("single responsibility", "one thing"),
        ("parameterized queries", "parameterized queries"),
        ("no secrets in output", "never print a secret"),
        ("constant-time secret comparison", "constant time"),
        ("path validation", "validate a path"),
        ("do not reorganize the project", "do not reorganize"),
        ("no dependency for trivial work", "beat a new dependency"),
        ("patterns need a trigger", "trigger is present"),
        ("no commit unless asked", "do not branch, commit or push"),
    ):
        assert marker in lowered, f"the compact core lost: {rule}"


def test_the_compact_bars_keep_the_product_bars() -> None:
    lean = get_variant("loop.behavior_guidelines", "lean") or ""
    lowered = _flatten(lean)

    for bar, marker in (
        ("honesty", "report results exactly as they are"),
        ("no exaggeration", "never exaggerate a success"),
        ("show failures", "show the whole picture"),
        ("no claim without running it", "unless you ran it"),
        ("no faked tests", "never fake a test"),
        ("no hidden placeholders", "no `todo`"),
        ("literal scope is authoritative", "literal active task is authoritative"),
        ("assumptions are not requirements", "never becomes an acceptance criterion"),
        ("bounded retries", "bound the retries"),
        ("ask on unresolved reference", "if it stays non-unique, ask"),
    ):
        assert marker in lowered, f"the compact bars lost: {bar}"


def test_the_original_constants_are_untouched() -> None:
    """Other styles and the reviewer must keep reading the originals."""
    assert "## Product bars and working guidance" in BEHAVIOR_GUIDELINES
    assert len(_DEVELOP_IDENTITY_BASE) > 7000
