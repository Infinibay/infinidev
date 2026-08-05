"""Style and correctness rules for prompt text, enforced instead of remembered.

Four properties, measured against every prompt string in ``prompts/`` and in
the loop's own prompt modules:

1. **Every tool the prompt tells the model to call exists**, and exists *for
   that role*. A prompt naming a tool the model cannot call teaches it to
   hallucinate that call, and the model has no way to discover the mistake.
2. **No evasive uncertainty in an instruction.** Words such as "perhaps" or
   "try to" avoid making a recommendation. Calibrated method guidance such as
   "prefer X when Y; depart when Z" is allowed: methods are defaults, unlike
   tool contracts and product bars.
3. **No arrow glyphs.** ``→`` and ``=>`` carry a meaning ("then", "produces",
   "instead") that words carry better, because words are what the training
   corpus is made of.
4. **No threshold-free instructions.** Words such as "relevant", "proper",
   and "sufficient" force the model to invent a decision criterion. State the
   observable criterion instead.

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

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
PROMPTS_DIR = REPO_ROOT / "src/infinidev/prompts"

# Model-facing instructions also live outside ``prompts/``. Keep this list
# explicit: every entry either builds an LLM message, contributes text to one,
# or appends instructions to a tool result the model reads next.
RUNTIME_PROMPT_MODULES = [
    "finetune.build_dataset",
    "infinidev.config.model_capabilities",
    "infinidev.engine.analysis.adversarial_verifier",
    "infinidev.engine.analysis.grounded_spec",
    "infinidev.engine.analysis.planner",
    "infinidev.engine.analysis.review_engine",
    "infinidev.engine.analysis.spec_elaborator",
    "infinidev.engine.behavior.batched_runner",
    "infinidev.engine.behavior.eval_context",
    "infinidev.engine.behavior.llm_checker",
    "infinidev.engine.behavior.registry",
    "infinidev.engine.council.agent_loop",
    "infinidev.engine.council.member",
    "infinidev.engine.council.moderator",
    "infinidev.engine.council.personas",
    "infinidev.engine.council.prompts",
    "infinidev.engine.council.runner",
    "infinidev.engine.guidance.library",
    "infinidev.engine.loop.behavior_rules",
    "infinidev.engine.loop.context",
    "infinidev.engine.loop.context_builder",
    "infinidev.engine.loop.critic",
    "infinidev.engine.loop.critic_liaison",
    "infinidev.engine.loop.engine",
    "infinidev.engine.loop.guardrail_runner",
    "infinidev.engine.loop.llm_caller",
    "infinidev.engine.loop.loop_guard",
    "infinidev.engine.loop.prompt.text",
    "infinidev.engine.loop.prompt.tools_section",
    "infinidev.engine.loop.step_complete_gate",
    "infinidev.engine.loop.step_summarizer",
    "infinidev.engine.loop.tool_runner",
    "infinidev.engine.loop.user_message_injector",
    "infinidev.engine.loop.work_summary",
    "infinidev.engine.orchestration.chat_agent",
    "infinidev.engine.orchestration.pipeline",
    "infinidev.engine.phases.investigator",
    "infinidev.engine.phases.plan_executor",
    "infinidev.engine.phases.plan_generator",
    "infinidev.engine.phases.question_generator",
    "infinidev.engine.tree.brainstorm",
    "infinidev.engine.tree.brainstorm_context",
    "infinidev.engine.tree.context",
    "infinidev.engine.tree.engine",
    "infinidev.gather.classifier",
    "infinidev.gather.mini_agent",
    "infinidev.gather.question",
    "infinidev.gather.runner",
    "infinidev.gather.templates.bug",
    "infinidev.gather.templates.feature",
    "infinidev.gather.templates.other",
    "infinidev.gather.templates.refactor",
    "infinidev.gather.templates.sysadmin",
    "infinidev.tools.docs.doc_flow",
    "infinidev.tools.meta.help_content",
    "infinidev.tools.shell.execute_command_tool",
]

RUNTIME_PROMPT_ROLES = {
    "infinidev.engine.analysis.planner": {"planner"},
    "infinidev.engine.council.agent_loop": {
        "council_member",
        "council_moderator",
    },
    "infinidev.engine.council.member": {"council_member"},
    "infinidev.engine.council.moderator": {"council_moderator"},
    "infinidev.engine.council.prompts": {"council_member", "council_moderator"},
    "infinidev.engine.council.runner": {
        "council_member",
        "council_moderator",
    },
    "infinidev.engine.orchestration.chat_agent": {"chat_agent"},
}

RUNTIME_SCHEMA_MODULES = {
    "infinidev.engine.analysis.spec_elaborator": {
        "infinidev.engine.analysis.spec_elaborator",
    },
    "infinidev.engine.tree.brainstorm_context": {
        "infinidev.engine.tree.context",
    },
    "infinidev.engine.tree.context": {"infinidev.engine.tree.context"},
    "infinidev.engine.tree.engine": {"infinidev.engine.tree.context"},
}

RUNTIME_API_MODULES = {
    "infinidev.tools.meta.help_content": {
        "infinidev.code_intel.interpreter_api",
    },
}

RUNTIME_TEXT_ATTRIBUTES = {
    "infinidev.tools.meta.help_content": {"HELP_CONTENT"},
}

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

_NAME = r"([a-z][a-z0-9]*(?:_[a-z0-9]+)*)"
_COMPOUND_NAME = r"([a-z][a-z0-9]*(?:_[a-z0-9]+)+)"
_INVOCATION = [
    re.compile(rf"\*\*{_NAME}\*\*(?:\(|\s*:)"),      # catalog: **tool**(args)
    # A single word after "call" is often prose ("call it", "call for").
    # Compound names retain enough shape to identify a tool invocation.
    re.compile(rf"\bcall(?:ing|s)?\s+`?{_COMPOUND_NAME}`?"),
    re.compile(rf"\buse\s+`?{_NAME}`?\s*\("),
    re.compile(rf"\bthe\s+`{_NAME}`\s+tool\b"),
    re.compile(rf"^\s*\d+\.\s+{_NAME}:", re.M),      # example step "1. read_file:"
]

_DISABLED_TOOL_INVOCATIONS = {
    "think": re.compile(
        r"(?:\bthink\s*\(|\bthe\s+`think`\s+tool\b|^\s*\d+\.\s+think:)",
        re.M,
    ),
}

EVASIVE_HEDGE = re.compile(
    r"\b(?:might|perhaps|possibly|probably|try to|ideally|sometimes|tends to)\b",
    re.I,
)
ARROW = re.compile(r"→|=>|->")

# Words that hand the model a decision and withhold the criterion to decide
# it. Distinct from a hedge: a hedge gives permission to skip the
# instruction, while one of these leaves the instruction standing and makes
# the model invent the threshold. "Run the relevant test" reads as guidance
# and specifies nothing — relevant to what? The fix is never a synonym; it is
# the criterion the word stood in for ("the test covering the file you
# edited").
UNKNOWN = re.compile(
    r"\b(?:appropriate(?:ly)?|relevant|as needed|as required|if necessary"
    r"|suitable|adequate(?:ly)?|reasonable|sufficient|meaningful|various"
    r"|where applicable|as applicable|and so on|etc\.|and more"
    r"|proper(?:ly)?|enough information|non-trivial|significant)\b",
    re.I,
)

# Phrases where the word is a noun or a quotation, not a hedged instruction.
HEDGE_ALLOWED_SUBSTRINGS = (
    "preference",                 # "user preferences", "stylistic preferences"
    '"It should work"',           # quoting the phrase the rule forbids
    "what the code should do",    # describing what a test states
    "I should now read",          # quoting filler the model must not emit
)

# The unknown-word rule also permits a prompt that quotes a header the
# engine emits verbatim is not the one choosing the wording.
UNKNOWN_ALLOWED_SUBSTRINGS = (
    "Known lessons relevant to this action",  # tool_executor.py:378, verbatim
)

# ── Deriving the surface ──────────────────────────────────────────────────

def _pseudo_schemas() -> list[dict]:
    """The engine-handled tools, read from their live schema constants."""
    import infinidev.engine.schema_sanitizer as ss

    return [
        ss.STEP_COMPLETE_SCHEMA,
        ss.ADD_NOTE_SCHEMA,
        ss.ADD_SESSION_NOTE_SCHEMA,
    ]


def _active_engine_schemas() -> list[dict]:
    """All schema constants sent by an engine phase."""
    import infinidev.engine.schema_sanitizer as ss

    return _pseudo_schemas() + [
        ss.GENERATE_QUESTION_SCHEMA,
        ss.ADD_STEP_SCHEMA,
        ss.MODIFY_STEP_SCHEMA,
        ss.REMOVE_STEP_SCHEMA,
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
    for schema in _active_engine_schemas():
        props = schema.get("function", {}).get("parameters", {}).get("properties", {})
        params |= set(props)
    return params


def _module_schema_tool_names(module_names: set[str]) -> set[str]:
    """Read custom runtime tool names from their live module schemas."""
    names: set[str] = set()
    for module_name in module_names:
        module = importlib.import_module(module_name)
        for value in vars(module).values():
            if not isinstance(value, dict) or value.get("type") != "function":
                continue
            name = value.get("function", {}).get("name", "")
            if isinstance(name, str) and name:
                names.add(name)
    return names


def _module_api_names(module_names: set[str]) -> set[str]:
    """Read names exposed inside model-operated interpreter environments."""
    names: set[str] = set()
    for module_name in module_names:
        module = importlib.import_module(module_name)
        names |= set(getattr(module, "__all__", ()))
    return names


def _runtime_attribute_strings(module_name: str) -> list[str]:
    """Read prompt text assembled dynamically at module import time."""
    module = importlib.import_module(module_name)
    strings: list[str] = []
    for attribute in RUNTIME_TEXT_ATTRIBUTES.get(module_name, set()):
        value = getattr(module, attribute)
        if isinstance(value, str):
            strings.append(value)
        elif isinstance(value, dict):
            strings.extend(item for item in value.values() if isinstance(item, str))
        elif isinstance(value, (list, tuple)):
            strings.extend(item for item in value if isinstance(item, str))
    return strings


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
            if isinstance(arg, ast.JoinedStr):
                out.add(id(arg))
            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                out.add(id(arg))
    return out


def _prompt_strings(path: pathlib.Path) -> list[str]:
    """String literals that reach the model: not docstrings, not log lines."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
        return []
    skip = _docstring_nodes(tree) | _log_message_nodes(tree)
    strings: list[str] = []
    for node in ast.walk(tree):
        text = ""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                value.value if isinstance(value, ast.Constant) else "{value}"
                for value in node.values
                if isinstance(value, (ast.Constant, ast.FormattedValue))
            )
        if len(text) > 40 and id(node) not in skip:
            strings.append(text)
    return strings


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


def _disabled_tool_invocations(texts: list[str]) -> set[str]:
    """Return disabled single-word tools invoked in model-facing text."""
    return {
        name
        for name, pattern in _DISABLED_TOOL_INVOCATIONS.items()
        if any(pattern.search(text) for text in texts)
    }


def _hedged_lines(texts: list[str]) -> list[str]:
    return [
        line.strip()
        for text in texts
        for line in text.splitlines()
        if EVASIVE_HEDGE.search(line)
        and not any(s in line for s in HEDGE_ALLOWED_SUBSTRINGS)
    ]


# ── The rules ─────────────────────────────────────────────────────────────

def test_the_surface_is_actually_derivable():
    """A guard on the guard: an empty surface would pass everything."""
    assert len(_tool_names("developer")) > 40
    assert len(_parameter_names()) > 40
    assert "read_file" in _tool_names("developer")
    assert "final_answer" in _parameter_names()


def test_primary_loop_prompt_separates_bars_from_adaptable_methods() -> None:
    """The default executor must not disguise execution heuristics as contracts."""
    from infinidev.engine.loop.prompt.text import (
        BEHAVIOR_GUIDELINES,
        BEHAVIOR_GUIDELINES_SMALL,
        CLI_AGENT_IDENTITY,
    )

    assert "## Product bars and working guidance" in BEHAVIOR_GUIDELINES
    assert "## Product bars and working guidance" in BEHAVIOR_GUIDELINES_SMALL
    assert "Execution methods are guidance" in BEHAVIOR_GUIDELINES
    assert "unresolved product\n  choice" in CLI_AGENT_IDENTITY
    assert "missing authorization" in CLI_AGENT_IDENTITY
    assert "active preference profile" in CLI_AGENT_IDENTITY
    assert "The live `record_finding` schema defines" in CLI_AGENT_IDENTITY
    assert "| `anchor_file=" not in CLI_AGENT_IDENTITY
    assert "NEVER pause mid-loop" not in CLI_AGENT_IDENTITY
    assert "NEVER re-open it" not in CLI_AGENT_IDENTITY


def test_execute_prompt_does_not_turn_heuristics_into_universal_rules() -> None:
    from infinidev.prompts.tool_hints import build_execute_prompt

    prompt = build_execute_prompt(
        available_tools={"read_file", "edit_file", "execute_command"},
        step_num=1,
        total_steps=1,
        step_title="Repair behavior",
        step_files="src/example.py",
    )

    assert "## Scope contract" in prompt
    assert "## Working guidance" in prompt
    assert "## Failure Patterns and Corrections" in prompt
    assert "Repeat the same failed action without new evidence" in prompt
    assert "blocked only when no in-scope action can produce new evidence" in prompt
    assert "ALWAYS run a test or an import check after every edit" not in prompt
    assert "Keep trying after 3 consecutive failures" not in prompt
    assert "Read the same file twice in one step" not in prompt


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
def test_no_evasive_hedging_in_prompt_text(path: pathlib.Path) -> None:
    offenders = _hedged_lines(_prompt_strings(path))
    assert not offenders, (
        "Evasive uncertainty provides no usable recommendation. State a hard "
        "contract, or give a method default with its decision criterion:\n  "
        + "\n  ".join(offenders)
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


@pytest.mark.parametrize("path", _prompt_files(), ids=_rel)
def test_no_unknown_implying_words_in_prompt_text(path: pathlib.Path) -> None:
    """A word that names no criterion makes the model invent one."""
    offenders = [
        line.strip()
        for text in _prompt_strings(path)
        for line in text.splitlines()
        if UNKNOWN.search(line)
    ]
    assert not offenders, (
        "These words hand the model a decision without the criterion to "
        "decide it. Replace each with what it stood in for:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("path", _prompt_files(), ids=_rel)
def test_no_prompt_invokes_a_disabled_single_word_tool(path: pathlib.Path) -> None:
    offenders = _disabled_tool_invocations(_prompt_strings(path))
    assert not offenders, f"{_rel(path)} invokes disabled tools: {sorted(offenders)}"


# ── The same rules, where nobody was looking ──────────────────────────────
#
# A tool's ``description=`` strings are serialised into the function-call
# schema and reach the model in the SAME request as the system prompt — often
# with more weight, since they sit beside the parameter it is about to fill.
# Scanning only ``prompts/`` left that surface unguarded, and it is where a
# live "Prefer a DETERMINISTIC kind" survived every pass of this file.


def _schema_strings(tool: object) -> list[str]:
    """Every description string in the schema this tool sends to the model."""
    from infinidev.engine.schema_sanitizer import tool_to_openai_schema

    out: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "description" and isinstance(value, str):
                    out.append(value)
                elif isinstance(value, (dict, list)):
                    walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    try:
        walk(tool_to_openai_schema(tool))
    except Exception:  # pragma: no cover - a tool that cannot serialise
        return []
    return out


def _every_bound_tool() -> list:
    from infinidev.tools import get_tools_for_role

    seen: set[str] = set()
    tools = []
    for role in ("developer", "chat_agent", "planner"):
        for tool in get_tools_for_role(role):
            if tool.name not in seen:
                seen.add(tool.name)
                tools.append(tool)
    return tools


def test_tool_schemas_follow_the_same_rules() -> None:
    offenders: list[str] = []
    for tool in _every_bound_tool():
        for text in _schema_strings(tool):
            for line in text.splitlines():
                for label, rx in (("hedge", EVASIVE_HEDGE), ("arrow", ARROW),
                                  ("unknown", UNKNOWN)):
                    if not rx.search(line):
                        continue
                    if label == "hedge" and any(
                        s in line for s in HEDGE_ALLOWED_SUBSTRINGS
                    ):
                        continue
                    if label == "unknown" and any(
                        s in line for s in UNKNOWN_ALLOWED_SUBSTRINGS
                    ):
                        continue
                    offenders.append(f"[{tool.name}/{label}] {line.strip()}")
    assert not offenders, (
        "Tool descriptions reach the model in the same request as the system "
        "prompt:\n  " + "\n  ".join(offenders)
    )


def test_engine_pseudo_tool_schemas_follow_the_same_rules() -> None:
    """Engine-handled schemas are prompt text even without a bound tool."""
    offenders: list[str] = []
    for schema in _pseudo_schemas():
        function = schema.get("function", {})
        name = function.get("name", "unknown")
        stack: list[object] = [function]
        while stack:
            value = stack.pop()
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "description" and isinstance(child, str):
                        for label, pattern in (
                            ("hedge", EVASIVE_HEDGE),
                            ("arrow", ARROW),
                            ("unknown", UNKNOWN),
                        ):
                            if pattern.search(child):
                                offenders.append(f"[{name}/{label}] {child}")
                    elif isinstance(child, (dict, list)):
                        stack.append(child)
            elif isinstance(value, list):
                stack.extend(value)

    assert not offenders, (
        "Engine pseudo-tool descriptions reach every developer prompt:\n  "
        + "\n  ".join(offenders)
    )


@pytest.mark.parametrize("module", RUNTIME_PROMPT_MODULES)
def test_runtime_prompt_modules_follow_the_same_rules(module: str) -> None:
    """Every runtime prompt surface follows the same language rules."""
    path = pathlib.Path(importlib.import_module(module).__file__)
    texts = _prompt_strings(path) + _runtime_attribute_strings(module)

    roles = RUNTIME_PROMPT_ROLES.get(module, {"developer"})
    allowed_tools: set[str] = set()
    for role in roles:
        allowed_tools |= _tool_names(role)
    allowed_tools |= _module_schema_tool_names(RUNTIME_SCHEMA_MODULES.get(module, set()))
    allowed_tools |= _module_api_names(RUNTIME_API_MODULES.get(module, set()))
    bad_tools = _named_tools(texts) - allowed_tools
    disabled_tools = _disabled_tool_invocations(texts)
    hedges = _hedged_lines(texts)
    arrows = [
        line.strip()
        for text in texts
        for line in text.splitlines()
        if ARROW.search(line)
    ]
    unknowns = [
        line.strip()
        for text in texts
        for line in text.splitlines()
        if UNKNOWN.search(line)
        and not any(s in line for s in UNKNOWN_ALLOWED_SUBSTRINGS)
    ]

    assert not bad_tools, f"{module} names missing tools: {sorted(bad_tools)}"
    assert not disabled_tools, f"{module} invokes disabled tools: {sorted(disabled_tools)}"
    assert not hedges, f"{module} hedges:\n  " + "\n  ".join(hedges)
    assert not arrows, f"{module} uses arrows:\n  " + "\n  ".join(arrows)
    assert not unknowns, (
        f"{module} hands the model a decision without the criterion to make "
        f"it:\n  " + "\n  ".join(unknowns)
    )
