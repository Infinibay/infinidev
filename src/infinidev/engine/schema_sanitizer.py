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


def _inline_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Inline ``$defs``/``$ref`` so nested models survive ``$defs`` stripping.

    Pydantic v2 emits nested BaseModel fields (e.g. ``steps: list[PlanStepArg]``
    in emit_plan) as a ``$ref`` into a top-level ``$defs`` table. The schema
    cleaners drop ``$defs`` for providers that choke on it — which would leave
    a DANGLING ``$ref`` and silently erase every nested field from the schema
    the LLM sees. Resolving the refs first keeps the nested fields intact.

    No-op for the common case (no ``$defs`` and a flat tool schema). A depth guard
    bounds any self-referential model rather than recursing forever.
    """
    defs = schema.get("$defs") or schema.get("definitions")
    if not defs:
        return schema

    import copy
    defs = copy.deepcopy(defs)

    def resolve(node: Any, depth: int) -> Any:
        if depth > 8:
            return node
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and "/" in ref:
                name = ref.split("/")[-1]
                target = defs.get(name)
                if isinstance(target, dict):
                    merged = copy.deepcopy(target)
                    # Preserve sibling keys (e.g. a per-field description that
                    # sits next to the $ref) over the target's own.
                    for k, v in node.items():
                        if k != "$ref":
                            merged[k] = v
                    return resolve(merged, depth + 1)
            return {k: resolve(v, depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(x, depth + 1) for x in node]
        return node

    top = {k: v for k, v in schema.items() if k not in ("$defs", "definitions")}
    return resolve(top, 0)


def _clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove Pydantic v2 artifacts that confuse LLM providers."""
    schema = _inline_defs(schema)
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
    # Resolve anyOf/oneOf by selecting the first non-null type.
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
                # Both schema extractions failed, so the tool would register with
                # an EMPTY parameter schema, which CLAUDE.md calls the security
                # boundary. Make it loud instead of silently shipping a zero-arg
                # tool the model cannot call correctly.
                logger.warning(
                    "tool_to_openai_schema: could not extract args schema for "
                    "tool %r; registering with EMPTY parameters",
                    getattr(tool, "name", type(tool).__name__),
                    exc_info=True,
                )
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

    description = (tool.description or "").strip()
    metadata: list[str] = []
    effects = getattr(tool, "effects", None)
    if effects is not None and (effect_summary := effects.summary()):
        metadata.append(f"Effects: {effect_summary}.")
    constraints = getattr(tool, "use_constraints", None)
    if constraints is not None and (constraint_summary := constraints.summary()):
        metadata.append(constraint_summary)
    suffix = (" " + " ".join(metadata)) if metadata else ""
    if len(suffix) >= 1024:
        description = suffix[-1024:]
    else:
        description = description[: 1024 - len(suffix)].rstrip() + suffix

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": description,
            "parameters": parameters,
        },
    }


STEP_COMPLETE_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "step_complete",
        "description": (
            "Call step_complete after finishing the current step and checking its "
            "expected_output or success criterion. Supply the file read, test result, "
            "or command output that proves success. The engine discards this step's "
            "tool outputs and conversation after the call; only the summary and notes "
            "survive. Save file paths, symbol names, and decisions with add_note first. "
            "Before status='done', record cross-task lessons with add_session_note. "
            "Change the remaining plan with add_step, modify_step, or remove_step before "
            "closing the step."
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
                        "Concrete evidence that the step reached its objective: commands "
                        "and outcomes, files re-read after editing, and tests that passed. "
                        "Use at least 30 characters. Replace claims such as 'looks good' "
                        "with the observed verification result. The assistant critic uses "
                        "this evidence to accept or reject the step closure."
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
            "Save a fact to working memory after each file read, discovery, or decision. "
            "The engine rebuilds context at every step and discards facts absent from "
            "working memory. Notes appear in <notes> at every step. The store accepts "
            "at most 20 notes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {
                    "type": "string",
                    "description": "The note to save (1-2 sentences)",
                    "maxLength": 800,
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
                    "maxLength": 800,
                },
            },
            "required": ["note"],
        },
    },
}


GENERATE_QUESTION_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "generate_question",
        "description": (
            "Generate one investigation question about the task. Call this once per "
            "question. Call step_complete with status='done' after the question set "
            "covers every unknown that can change the implementation."
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

    Always append the active engine pseudo-tools so the model can signal step
    completion and persist notes. The retired ``think`` pseudo-tool is omitted
    for every model because it encouraged loops without observable progress.
    """
    schemas = [tool_to_openai_schema(t) for t in tools]
    schemas.append(STEP_COMPLETE_SCHEMA)
    schemas.append(ADD_NOTE_SCHEMA)
    schemas.append(ADD_SESSION_NOTE_SCHEMA)
    # Plan tools are registered in META_TOOLS and use tool_to_openai_schema().

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
    # of the rich docstring (which would otherwise get truncated at a
    # ~160-char sentence boundary and may lose its key signal). Each entry must keep the
    # *callable* name and the most important capability hint within
    # the budget; everything else moves to `help`.
    "code_interpreter": (
        "Run Python in sandbox. 13 code-intel helpers pre-imported "
        "(iter_symbols, find_references, ...). describe_tool(context='code_interpreter')."
    ),
}


def _simplify_schema_for_small(schema: dict[str, Any]) -> dict[str, Any]:
    """Simplify a tool schema for small models (<40B).

    - Replaces description with a hand-tuned short version when
      available in ``_SMALL_MODEL_DESCRIPTIONS``; otherwise truncates
      at a ~160-char sentence/clause boundary (never mid-word).
    - Removes 'explore' from step_complete status enum
    - Strips optional parameter descriptions to save tokens
    """
    import copy
    schema = copy.deepcopy(schema)
    func = schema.get("function", {})

    # Description: prefer hand-tuned short version, else truncate at a
    # sentence/clause boundary (never mid-word) so small models still get a
    # coherent hint — a hard char cut amputated step_complete's guidance.
    name = func.get("name", "")
    short_desc = _SMALL_MODEL_DESCRIPTIONS.get(name)
    if short_desc:
        func["description"] = short_desc
    else:
        desc = func.get("description", "")
        if len(desc) > 160:
            cut = desc[:160]
            boundary = max(cut.rfind(". "), cut.rfind("; "), cut.rfind(", "))
            if boundary >= 80:
                cut = cut[: boundary + 1]
            func["description"] = cut.rstrip() + " …"

    # Remove 'explore' status from step_complete (confuses small models)
    if func.get("name") == "step_complete":
        props = func.get("parameters", {}).get("properties", {})
        status_prop = props.get("status", {})
        if "enum" in status_prop:
            status_prop["enum"] = [s for s in status_prop["enum"] if s != "explore"]

    return schema
