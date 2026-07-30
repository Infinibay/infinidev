"""Style and correctness rules for prompt text, enforced instead of remembered.

Three properties, measured against every prompt string in ``prompts/`` and in
the loop's own prompt modules:

1. **Every tool the prompt tells the model to call exists**, and exists *for
   that role*. A prompt naming a tool the model cannot call teaches it to
   hallucinate that call, and the model has no way to discover the mistake.
2. **No hedging in an instruction.** "prefer X", "you should Y", "generally Z"
   give the model room to not do the thing. An instruction either applies or
   it is not an instruction.
3. **No arrow glyphs.** ``→`` and ``=>`` carry a meaning ("then", "produces",
   "instead") that words carry better, because words are what the training
   corpus is made of.

Two things keep rule 1 precise. It looks only at invocation contexts — "call
X", "use X(", a ``**X**:`` catalog entry, a numbered example step — because
matching every snake_case token drowns the signal in example functions
(``verify_token``) and the pseudo-code DSL in ``variants/coding.py``. And both
the tool names and the *parameter* names come from the live schemas, so
``**final_answer**:`` documenting a parameter is not mistaken for a tool and
nothing here has to be updated when the surface changes.

Against the tree before the editing-cluster surgery, these rules found all five
dead tool names and nothing else.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
import re

import pytest

PROMPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src/infinidev/prompts"

# Prompt modules that live outside ``prompts/`` because the loop builds them.
LOOP_PROMPT_MODULES = [
    "infinidev.engine.loop.prompt.text",
    "infinidev.engine.guidance.library",
]

# ken's published surface. The MCP bridge discovers these from ``tools/list``
# at runtime, so no server runs here. ``test_tool_docs_complete.py`` is what
# checks this list against a live server.
KEN_TOOLS = frozenset({
    "ken_find", "ken_read", "ken_related", "ken_rank", "ken_recall",
    "ken_remember",
})

# Tools with no schema anywhere: the model reaches them through the engine.
UNSCHEMA_TOOLS = frozenset({
    "respond", "escalate", "emit_plan", "emit_spec", "emit_verdict",
    "emit_questions", "send_message", "web_search", "web_fetch",
    "code_search_web",
})

# Which directory under prompts/ belongs to which tool tier.
ROLE_BY_DIR = {"chat_agent": "chat_agent", "analyst": "planner"}

_NAME = r"([a-z][a-z0-9]*(?:_[a-z0-9]+)+)"
_INVOCATION = [
    re.compile(rf"\*\*{_NAME}\*\*\s*[:(]"),          # catalog: **tool**(args)
    re.compile(rf"\bcall(?:ing|s)?\s+`?{_NAME}`?"),
    re.compile(rf"\buse\s+`?{_NAME}`?\s*\("),
    re.compile(rf"\bthe\s+`{_NAME}`\s+tool\b"),
    re.compile(rf"^\s*\d+\.\s+{_NAME}:", re.M),      # example step "1. read_file:"
]

HEDGE = re.compile(
    r"\b(?:could|should|might|perhaps|possibly|probably|prefer(?:ably|s|red)?"
    r"|try to|ideally|generally|typically|usually|sometimes|tends to)\b",
    re.I,
)
ARROW = re.compile(r"→|=>")

# Files whose prompt text legitimately carries these words, each with the
# reason. A new entry has to be argued for rather than slipped in.
HEDGE_EXEMPT = {
    # Quoted example questions the model is told to generate or to avoid.
    # A question about the unknown cannot be phrased as an imperative.
    "phases/questions.py",
    # A pseudo-code DSL: ``prefer(simple_obvious_code, over=clever_tricks)``
    # is a call in the variant's own notation, not English hedging.
    "variants/coding.py",
}

# Phrases where the word is a noun or a quotation, not a hedged instruction.
HEDGE_ALLOWED_SUBSTRINGS = (
    "preference",                 # "user preferences", "stylistic preferences"
    '"It should work"',           # quoting the phrase the rule forbids
    "what the code should do",    # describing what a test states
    "I should now read",          # quoting filler the model must not emit
)


# ── Deriving the surface ──────────────────────────────────────────────────

def _pseudo_schemas() -> list[dict]:
    """The engine-handled tools, read from their live schema constants."""
    import infinidev.engine.schema_sanitizer as ss

    return [
        getattr(ss, name)
        for name in dir(ss)
        if name.endswith("_SCHEMA") and isinstance(getattr(ss, name), dict)
    ]


def _tool_names(role: str) -> set[str]:
    from infinidev.tools import get_tools_for_role

    names = {t.name for t in get_tools_for_role(role)}
    names |= {s.get("function", {}).get("name", "") for s in _pseudo_schemas()}
    return (names | set(KEN_TOOLS) | set(UNSCHEMA_TOOLS)) - {""}


def _parameter_names() -> set[str]:
    """Every argument name any tool accepts.

    Documentation writes a parameter the same way it writes a tool
    (``**final_answer**:``), so the only way to tell them apart is to know
    which names are parameters.
    """
    from infinidev.tools import get_tools_for_role

    params: set[str] = set()
    for role in ("developer", "chat_agent", "planner"):
        for tool in get_tools_for_role(role):
            schema = getattr(tool, "args_schema", None)
            params |= set(getattr(schema, "model_fields", {}) or {})
    for schema in _pseudo_schemas():
        props = schema.get("function", {}).get("parameters", {}).get("properties", {})
        params |= set(props)
    return params


# ── Reading prompt text ───────────────────────────────────────────────────

def _docstring_nodes(tree: ast.AST) -> set[int]:
    """ids of Constant nodes that are docstrings, not prompt text."""
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                out.add(id(first.value))
    return out


def _log_message_nodes(tree: ast.AST) -> set[int]:
    """ids of Constant nodes that are log messages, not prompt text.

    A ``logger.warning("could not read ...")`` reads like a hedge and never
    reaches the model.
    """
    out: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
        if target not in {"debug", "info", "warning", "error", "exception", "critical"}:
            continue
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.add(id(arg))
    return out


def _prompt_strings(path: pathlib.Path) -> list[str]:
    """String literals that reach the model: not docstrings, not log lines."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return []
    skip = _docstring_nodes(tree) | _log_message_nodes(tree)
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and len(node.value) > 40
        and id(node) not in skip
    ]


def _prompt_files() -> list[pathlib.Path]:
    return sorted(PROMPTS_DIR.rglob("*.py"))


def _rel(path: pathlib.Path) -> str:
    return str(path.relative_to(PROMPTS_DIR))


def _role_for(path: pathlib.Path) -> str:
    for part in path.parts:
        if part in ROLE_BY_DIR:
            return ROLE_BY_DIR[part]
    return "developer"


def _named_tools(texts: list[str]) -> set[str]:
    found: set[str] = set()
    for text in texts:
        for pattern in _INVOCATION:
            found |= set(pattern.findall(text))
    return found - _parameter_names()


def _hedged_lines(texts: list[str]) -> list[str]:
    return [
        line.strip()
        for text in texts
        for line in text.splitlines()
        if HEDGE.search(line)
        and not any(s in line for s in HEDGE_ALLOWED_SUBSTRINGS)
    ]


# ── The rules ─────────────────────────────────────────────────────────────

def test_the_surface_is_actually_derivable():
    """A guard on the guard: an empty surface would pass everything."""
    assert len(_tool_names("developer")) > 40
    assert len(_parameter_names()) > 40
    assert "read_file" in _tool_names("developer")
    assert "final_answer" in _parameter_names()


@pytest.mark.parametrize("path", _prompt_files(), ids=_rel)
def test_every_tool_the_prompt_tells_the_model_to_call_exists(path: pathlib.Path) -> None:
    role = _role_for(path)
    offenders = _named_tools(_prompt_strings(path)) - _tool_names(role)
    assert not offenders, (
        f"{_rel(path)} tells the model to call tools that do not exist for "
        f"role {role!r}: {', '.join(sorted(offenders))}"
    )


@pytest.mark.parametrize("path", _prompt_files(), ids=_rel)
def test_no_prompt_names_a_tool_from_another_role(path: pathlib.Path) -> None:
    """A chat-agent prompt naming a write tool teaches it to try one."""
    role = _role_for(path)
    everywhere = set().union(
        *(_tool_names(r) for r in ("developer", "chat_agent", "planner"))
    )
    named = _named_tools(_prompt_strings(path))
    offenders = (named - _tool_names(role)) & everywhere
    assert not offenders, (
        f"{_rel(path)} is a {role!r} prompt but names tools that role cannot "
        f"call: {', '.join(sorted(offenders))}"
    )


@pytest.mark.parametrize("path", _prompt_files(), ids=_rel)
def test_no_hedging_in_prompt_text(path: pathlib.Path) -> None:
    if _rel(path) in HEDGE_EXEMPT:
        pytest.skip(f"{_rel(path)} is exempt — see HEDGE_EXEMPT")
    offenders = _hedged_lines(_prompt_strings(path))
    assert not offenders, (
        "Hedging leaves the model room to skip the instruction. Rewrite as an "
        "imperative or an IF/THEN:\n  " + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("path", _prompt_files(), ids=_rel)
def test_no_arrow_glyphs_in_prompt_text(path: pathlib.Path) -> None:
    offenders = [
        line.strip()
        for text in _prompt_strings(path)
        for line in text.splitlines()
        if ARROW.search(line)
    ]
    assert not offenders, (
        'Arrows read worse than the word does. Write "then", "Output:" or '
        '"INSTEAD:":\n  ' + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("module", LOOP_PROMPT_MODULES)
def test_the_loops_own_prompts_follow_the_same_rules(module: str) -> None:
    """text.py and library.py build prompts too, and drift the same way."""
    path = pathlib.Path(importlib.import_module(module).__file__)
    texts = _prompt_strings(path)

    bad_tools = _named_tools(texts) - _tool_names("developer")
    hedges = _hedged_lines(texts)
    arrows = [
        line.strip()
        for text in texts
        for line in text.splitlines()
        if ARROW.search(line)
    ]

    assert not bad_tools, f"{module} names missing tools: {sorted(bad_tools)}"
    assert not hedges, f"{module} hedges:\n  " + "\n  ".join(hedges)
    assert not arrows, f"{module} uses arrows:\n  " + "\n  ".join(arrows)
