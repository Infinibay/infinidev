"""Validated loading for packaged phase-prompt JSON resources."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
import json
from typing import Final


PHASE_RESOURCE_PACKAGE: Final = "infinidev.prompts.phase_resources"
PHASE_TASK_TYPES: Final = ("bug", "feature", "refactor", "other")
_INVESTIGATE_TOKEN: Final = "<<INVESTIGATE_RULES>>"
_EXECUTE_TOKEN: Final = "<<EXECUTE_EDIT_CONTRACT>>"


@dataclass(frozen=True)
class SharedPhasePrompts:
    """Prompt fragments shared by every phase strategy."""

    investigate_rules: str
    execute_edit_contract: str
    planner_identity: str
    followup_prompt: str


@dataclass(frozen=True)
class PhasePromptBundle:
    """All prompt text and fallback questions for one task type."""

    task_type: str
    questions_prompt: str
    fallback_questions: tuple[str, ...]
    investigate_prompt: str
    investigate_identity: str
    plan_prompt: str
    plan_identity: str
    execute_prompt: str
    execute_identity: str


def _load_document(resource_name: str) -> dict[str, object]:
    """Load one JSON resource and require an object root."""
    resource = files(PHASE_RESOURCE_PACKAGE).joinpath(resource_name)
    try:
        document = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as err:
        raise RuntimeError(
            f"Could not load phase prompt resource {resource_name}: {err}"
        ) from err
    if not isinstance(document, dict):
        raise RuntimeError(
            f"Phase prompt resource {resource_name} must contain a JSON object"
        )
    return document


def _require_fields(
    document: dict[str, object],
    expected: set[str],
    source: str,
) -> None:
    """Reject missing and unknown fields so prompt typos fail closed."""
    actual = set(document)
    missing = expected - actual
    unknown = actual - expected
    if not missing and not unknown:
        return
    details: list[str] = []
    if missing:
        details.append(f"missing {sorted(missing)}")
    if unknown:
        details.append(f"unknown {sorted(unknown)}")
    raise RuntimeError(f"Phase prompt {source} has invalid fields: {', '.join(details)}")


def _require_schema_version(document: dict[str, object], source: str) -> None:
    """Require the single schema version understood by this loader."""
    version = document.get("schema_version")
    if type(version) is not int or version != 1:
        raise RuntimeError(f"Phase prompt {source} schema_version must be 1")


def _require_object(
    document: dict[str, object],
    field: str,
    source: str,
) -> dict[str, object]:
    """Return one required object field."""
    value = document.get(field)
    if not isinstance(value, dict):
        raise RuntimeError(f"Phase prompt {source} field {field!r} must be an object")
    return value


def _render_lines(document: dict[str, object], field: str, source: str) -> str:
    """Validate and render one JSON string-array as newline-terminated text."""
    value = document.get(field)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(line, str) for line in value)
    ):
        raise RuntimeError(
            f"Phase prompt {source} field {field!r} must be a non-empty string array"
        )
    rendered = "\n".join(value) + "\n"
    if not rendered.strip():
        raise RuntimeError(
            f"Phase prompt {source} field {field!r} must contain non-empty text"
        )
    return rendered


def _fallback_questions(document: dict[str, object], source: str) -> tuple[str, ...]:
    """Validate fallback questions for direct-LLM failure recovery."""
    value = document.get("fallback")
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        raise RuntimeError(
            f"Phase prompt {source} field 'fallback' must be a non-empty string array"
        )
    return tuple(value)


def _expand_once(
    prompt: str,
    token: str,
    replacement: str,
    source: str,
) -> str:
    """Expand one required shared-fragment marker without touching runtime braces."""
    marker = token + "\n"
    if prompt.count(token) != 1 or marker not in prompt:
        raise RuntimeError(
            f"Phase prompt {source} must contain {token!r} on its own line exactly once"
        )
    return prompt.replace(marker, replacement, 1)


@lru_cache(maxsize=1)
def load_shared_phase_prompts() -> SharedPhasePrompts:
    """Load and validate shared phase prompt fragments once per process."""
    source = "shared.json"
    document = _load_document(source)
    _require_fields(
        document,
        {
            "schema_version",
            "investigate_rules",
            "execute_edit_contract",
            "planner_identity",
            "followup_prompt",
        },
        source,
    )
    _require_schema_version(document, source)
    return SharedPhasePrompts(
        investigate_rules=_render_lines(document, "investigate_rules", source),
        execute_edit_contract=_render_lines(document, "execute_edit_contract", source),
        planner_identity=_render_lines(document, "planner_identity", source),
        followup_prompt=_render_lines(document, "followup_prompt", source),
    )


@lru_cache(maxsize=None)
def load_phase_prompt_bundle(task_type: str) -> PhasePromptBundle:
    """Load and validate all prompts for one supported phase task type."""
    if task_type not in PHASE_TASK_TYPES:
        raise ValueError(
            f"Unknown phase prompt task type {task_type!r}; "
            f"expected one of {', '.join(PHASE_TASK_TYPES)}"
        )

    source = f"{task_type}.json"
    document = _load_document(source)
    _require_fields(
        document,
        {
            "schema_version",
            "task_type",
            "questions",
            "investigate",
            "plan",
            "execute",
        },
        source,
    )
    _require_schema_version(document, source)
    if document.get("task_type") != task_type:
        raise RuntimeError(
            f"Phase prompt {source} task_type must be {task_type!r}"
        )

    questions = _require_object(document, "questions", source)
    investigate = _require_object(document, "investigate", source)
    plan = _require_object(document, "plan", source)
    execute = _require_object(document, "execute", source)
    _require_fields(questions, {"prompt", "fallback"}, f"{source}.questions")
    _require_fields(investigate, {"prompt", "identity"}, f"{source}.investigate")
    _require_fields(plan, {"prompt", "identity"}, f"{source}.plan")
    _require_fields(execute, {"prompt", "identity"}, f"{source}.execute")

    shared = load_shared_phase_prompts()
    investigate_prompt = _expand_once(
        _render_lines(investigate, "prompt", f"{source}.investigate"),
        _INVESTIGATE_TOKEN,
        shared.investigate_rules,
        f"{source}.investigate.prompt",
    )
    execute_prompt = _expand_once(
        _render_lines(execute, "prompt", f"{source}.execute"),
        _EXECUTE_TOKEN,
        shared.execute_edit_contract,
        f"{source}.execute.prompt",
    )
    return PhasePromptBundle(
        task_type=task_type,
        questions_prompt=_render_lines(questions, "prompt", f"{source}.questions"),
        fallback_questions=_fallback_questions(questions, f"{source}.questions"),
        investigate_prompt=investigate_prompt,
        investigate_identity=_render_lines(
            investigate, "identity", f"{source}.investigate"
        ),
        plan_prompt=_render_lines(plan, "prompt", f"{source}.plan"),
        plan_identity=_render_lines(plan, "identity", f"{source}.plan"),
        execute_prompt=execute_prompt,
        execute_identity=_render_lines(execute, "identity", f"{source}.execute"),
    )
