"""Tool schema conversion + sanitization for the loop engine.

Extracted from ``loop/tools.py`` so the pure "tool → JSON schema"
pipeline can be tested without having to exercise the dispatcher or
the execute_tool_call path. Everything here is side-effect free:
inputs are ``InfinibayBaseTool`` instances or raw schema dicts,
outputs are dicts ready to be handed to LiteLLM.

Kept as module-level functions (not a class) because every call site
already imports them by name and a class would add indirection without
any state to justify it.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove Pydantic v2 artifacts that confuse LLM providers."""
    schema.pop("title", None)
    schema.pop("$defs", None)
    schema.pop("definitions", None)
    # Recurse into properties
    for prop in schema.get("properties", {}).values():
        if isinstance(prop, dict):
            prop.pop("title", None)
    return schema


def _sanitize_schema_deep(schema: dict[str, Any]) -> dict[str, Any]:
    """Aggressively simplify a JSON schema for providers that reject anyOf/oneOf.

    Qwen/DashScope and some other providers reject complex schema constructs.
    This flattens anyOf/oneOf to the first non-null type and recurses into
    nested properties and array items.
    """
    import copy
    schema = copy.deepcopy(schema)
    _simplify_node(schema)
    return schema


def _simplify_node(node: dict[str, Any]) -> None:
    """Recursively simplify a schema node in-place."""
    # Resolve anyOf/oneOf → pick first non-null type
    for key in ("anyOf", "oneOf"):
        if key in node:
            variants = node.pop(key)
            chosen = None
            for v in variants:
                if isinstance(v, dict) and v.get("type") != "null":
                    chosen = v
                    break
            if chosen:
                # Merge the chosen variant into the node
                for k, v in chosen.items():
                    if k not in node:
                        node[k] = v

    # Remove unsupported keywords
    for drop in ("$defs", "definitions", "title", "default", "examples"):
        node.pop(drop, None)

    # Recurse into properties
    for prop in node.get("properties", {}).values():
        if isinstance(prop, dict):
            _simplify_node(prop)

    # Recurse into array items
    items = node.get("items")
    if isinstance(items, dict):
        _simplify_node(items)


def tool_to_openai_schema(tool: Any) -> dict[str, Any]:
    """Convert a InfinibayBaseTool to an OpenAI function-calling tool schema."""
    parameters: dict[str, Any] = {"type": "object", "properties": {}}

    if hasattr(tool, "args_schema") and tool.args_schema is not None:
        try:
            parameters = tool.args_schema.model_json_schema()
        except Exception:
            try:
                parameters = tool.args_schema.schema()
            except Exception:
                pass
        parameters = _clean_schema(parameters)

    # Ensure required fields
    parameters.setdefault("type", "object")
    parameters.setdefault("properties", {})

    # Zero-arg tools: pin required=[] and additionalProperties=false
    # so strict-mode providers (OpenAI strict, Anthropic) reject
    # hallucinated kwargs at the provider layer instead of forcing the
    # executor to clean them up. Without the explicit required/[]
    # signal, many open-weight models invent fields (e.g. `project_id`)
    # because the empty-props schema doesn't feel "complete" to them.
    if not parameters.get("properties"):
        parameters["required"] = []
        parameters["additionalProperties"] = False

    # Strip the `description` that pydantic copies into the parameters
    # node from the model docstring — it belongs only at the function
    # level. Leaving it in makes some providers log warnings and is
    # never what the OpenAI tool schema contract expects.
    parameters.pop("description", None)

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": (tool.description or "")[:1024],
            "parameters": parameters,
        },
    }


STEP_COMPLETE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "step_complete",
        "description": (
            "After finishing the current step objective AND verifying the outcome "
            "(against the step's expected_output / success criterion), run step_complete. "
            "Do NOT call this before you have evidence the step succeeded — re-read the file, "
            "run the test, or check the command output first. "
            "WARNING: After this call, ALL tool outputs and conversation from this step will be discarded. "
            "Only the summary and your notes (add_note) survive to the next step. "
            "Before calling this, save key facts via add_note (file paths, function names, decisions). "
            "Before status='done', call add_session_note with what you learned. "
            "To modify the plan, use add_step/modify_step/remove_step BEFORE calling this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Structured summary (~150 tokens): Read: files+findings | Changed: files+edits | Remaining: next work | Decisions: key choices. Skip empty categories.",
                },
                "evidence_summary": {
                    "type": "string",
                    "description": (
                        "REQUIRED. Concrete evidence that the step's "
                        "objective was reached: which command(s) you "
                        "ran and their outcome, which file(s) you "
                        "re-read after editing, which test(s) "
                        "passed. ≥30 chars. Do NOT write 'looks "
                        "good' or 'should work' — name the actual "
                        "verification. The assistant critic uses this "
                        "field to decide whether to accept or reject "
                        "the step closure."
                    ),
                    "minLength": 30,
                },
                "status": {
                    "type": "string",
                    "enum": ["continue", "done", "blocked", "explore"],
                    "description": "continue = more work to do, done = task complete, blocked = cannot proceed, explore = delegate sub-problem to exploration tree",
                },
                "final_answer": {
                    "type": "string",
                    "description": "When status=done, the final result to return",
                },
            },
            "required": ["summary", "status", "evidence_summary"],
        },
    },
}


ADD_NOTE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_note",
        "description": (
            "IMPORTANT: Save a fact to your persistent memory. Your context is rebuilt "
            "from scratch each step — anything not saved here is PERMANENTLY LOST. "
            "Call this after every file read, discovery, or decision. "
            "Notes appear in <notes> at every step. Max 20 notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to save (1-2 sentences)",
                },
            },
            "required": ["note"],
        },
    },
}


ADD_SESSION_NOTE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_session_note",
        "description": (
            "Save a note that persists across tasks in this session. Unlike add_note "
            "(which resets each task), session notes survive until the session ends. "
            "Use for: project-wide context, user preferences discovered during work, "
            "cross-task decisions, and anything the next task will need. "
            "Max 10 session notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The session note to save (1-2 sentences)",
                },
            },
            "required": ["note"],
        },
    },
}


THINK_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "think",
        "description": (
            "Think through a problem before acting. Use this when you need to "
            "reason about what to do next, analyze an error, or plan your approach. "
            "Your reasoning is shown to the user as a progress update. "
            "This does NOT count as a tool call — use it freely."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Your reasoning, analysis, or plan",
                },
            },
            "required": ["reasoning"],
        },
    },
}


GENERATE_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_question",
        "description": (
            "Generate one investigation question about the task. "
            "Call this once per question. When you have generated enough "
            "questions, call step_complete with status='done'."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "A specific investigation question answerable by reading "
                        "code, running tests, or searching the project."
                    ),
                },
                "intent": {
                    "type": "string",
                    "description": (
                        "What you hope to learn: find_code, find_patterns, "
                        "check_tests, find_config, find_dependents, reproduce, "
                        "baseline, understand_spec"
                    ),
                },
            },
            "required": ["question", "intent"],
        },
    },
}


ADD_STEP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "add_step",
        "description": (
            "Add a new step to the plan WITHOUT completing the current step. "
            "Use this when you discover new work mid-step. "
            "Does NOT count as a tool call. "
            "If index is omitted, the step is appended at the end of the plan."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Step number (position in plan). Omit to append at end.",
                },
                "title": {
                    "type": "string",
                    "description": "Short step title naming FILE, FUNCTION, and CHANGE",
                },
                "explanation": {
                    "type": "string",
                    "description": "Detailed explanation of how to approach the step (optional)",
                },
                "expected_output": {
                    "type": "string",
                    "description": (
                        "Your own success criterion for this step — one short, verifiable "
                        "sentence stating how you will know the step is done correctly. "
                        "Examples: 'pytest tests/test_auth.py::test_expired passes', "
                        "'auth.py:52 contains payload[\"exp\"] check'."
                    ),
                },
            },
            "required": ["title"],
        },
    },
}


MODIFY_STEP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "modify_step",
        "description": (
            "Modify the title, explanation, or success criterion of an existing pending step "
            "WITHOUT completing the current step. "
            "Does NOT count as a tool call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Step number to modify",
                },
                "title": {
                    "type": "string",
                    "description": "New title (leave empty to keep current)",
                },
                "explanation": {
                    "type": "string",
                    "description": "New explanation (leave empty to keep current)",
                },
                "expected_output": {
                    "type": "string",
                    "description": "New success criterion (leave empty to keep current)",
                },
            },
            "required": ["index"],
        },
    },
}


REMOVE_STEP_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "remove_step",
        "description": (
            "Remove a pending step from the plan WITHOUT completing the current step. "
            "Does NOT count as a tool call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "index": {
                    "type": "integer",
                    "description": "Step number to remove",
                },
            },
            "required": ["index"],
        },
    },
}


def build_tool_schemas(tools: list[Any], *, small_model: bool = False) -> list[dict[str, Any]]:
    """Convert a list of tools to OpenAI function-calling schemas.

    Always appends the engine pseudo-tools (step_complete, add_note)
    so the LLM can signal step completion and take notes.
    The ``think`` pseudo-tool is excluded for small models to prevent
    reasoning bloat (small models waste tokens on think → reason → think loops).
    """
    schemas = [tool_to_openai_schema(t) for t in tools]
    schemas.append(STEP_COMPLETE_SCHEMA)
    schemas.append(ADD_NOTE_SCHEMA)
    schemas.append(ADD_SESSION_NOTE_SCHEMA)
    # think pseudo-tool disabled — models abuse it to loop without acting
    # Plan tools (add_step, modify_step, remove_step) are real tools
    # registered in META_TOOLS — they get their schemas via tool_to_openai_schema().

    # Deep-sanitize schemas for providers that reject anyOf/oneOf/complex constructs
    from infinidev.config.model_capabilities import get_model_capabilities
    if get_model_capabilities().needs_schema_sanitization:
        schemas = [_sanitize_tool_schema(s) for s in schemas]

    # For small models: shorten descriptions and remove "explore" status
    if small_model:
        schemas = [_simplify_schema_for_small(s) for s in schemas]

    return schemas


def _sanitize_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Apply deep sanitization to a single tool schema."""
    import copy
    schema = copy.deepcopy(schema)
    params = schema.get("function", {}).get("parameters")
    if params:
        sanitized = _sanitize_schema_deep(params)
        schema["function"]["parameters"] = sanitized
    return schema


_SMALL_MODEL_DESCRIPTIONS: dict[str, str] = {
    # Hand-tuned compact descriptions for small models. Used in place
    # of the rich docstring (which would otherwise get truncated to
    # 120 chars and lose its key signal). Each entry must keep the
    # *callable* name and the most important capability hint within
    # the budget; everything else moves to `help`.
    "code_interpreter": (
        "Run Python in sandbox. 13 code-intel helpers pre-imported "
        "(iter_symbols, find_references, ...). help('code_interpreter')."
    ),
}


def _simplify_schema_for_small(schema: dict[str, Any]) -> dict[str, Any]:
    """Simplify a tool schema for small models (<40B).

    - Replaces description with a hand-tuned short version when
      available in ``_SMALL_MODEL_DESCRIPTIONS``; otherwise truncates
      to ≤120 chars.
    - Removes 'explore' from step_complete status enum
    - Strips optional parameter descriptions to save tokens
    """
    import copy
    schema = copy.deepcopy(schema)
    func = schema.get("function", {})

    # Description: prefer hand-tuned short version, else truncate
    name = func.get("name", "")
    short_desc = _SMALL_MODEL_DESCRIPTIONS.get(name)
    if short_desc:
        func["description"] = short_desc
    else:
        desc = func.get("description", "")
        if len(desc) > 120:
            func["description"] = desc[:117] + "..."

    # Remove 'explore' status from step_complete (confuses small models)
    if func.get("name") == "step_complete":
        props = func.get("parameters", {}).get("properties", {})
        status_prop = props.get("status", {})
        if "enum" in status_prop:
            status_prop["enum"] = [s for s in status_prop["enum"] if s != "explore"]

    return schema


