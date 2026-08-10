"""Deterministic request signals shared by orchestration gates."""

from __future__ import annotations

import os
import re


_EXECUTION_ACTION_RE = re.compile(
    r"\b(?:add|implement|fix|change|update|remove|delete|refactor|create|write|"
    r"build|run|execute|agreg(?:a|ar)|implement(?:a|ar)|arregl(?:a|ar|á)|"
    r"cambi(?:a|ar)|actualiz(?:a|ar)|elimin(?:a|ar)|cre(?:a|ar)|ejecut(?:a|ar))\b",
    re.IGNORECASE,
)
_EXECUTION_TARGET_RE = re.compile(
    r"\b(?:repo(?:sitory)?|codebase|code|bug|function|helper|class|module|"
    r"archivo|repositorio|c[oó]digo|funci[oó]n|clase|m[oó]dulo)\b|"
    r"(?:^|\s)(?:src|tests?)/\S+|\b\w+\.(?:py|js|ts|rs|go|java|c|cpp|h)\b",
    re.IGNORECASE,
)
_EXECUTION_VERIFY_RE = re.compile(
    r"\b(?:tests?|pytest|verify|validate|pruebas?|probar|verificar|validar)\b",
    re.IGNORECASE,
)
_INFORMATIONAL_RE = re.compile(
    r"^\s*(?:how\s+(?:do|can|would|should)|what\s+(?:is|does|would|should)|why\b|"
    r"explain\b|c[oó]mo\s+(?:puedo|se|har[ií]a)|qu[eé]\s+(?:es|hace)|por\s+qu[eé]\b|"
    r"explic(?:a|ame|ar)\b)",
    re.IGNORECASE,
)
_NO_EXECUTION_RE = re.compile(
    r"\b(?:do not|don't)\s+(?:change|edit|modify|implement)\s+"
    r"(?:anything|the\s+code|code|files?)\b|"
    r"\bwithout\s+(?:making\s+)?(?:changes?|edits?|modifying)\b|"
    r"\bno\s+(?:cambies|edites|modifiques|implementes)\s+"
    r"(?:nada|el\s+c[oó]digo|archivos?)\b|"
    r"\bsin\s+(?:hacer\s+)?(?:cambios|editar|modificar)\b|"
    r"\b(?:just|only|solo|solamente)\s+(?:explain|describe|explica|describe)\b",
    re.IGNORECASE,
)

# A strong target is something the runtime can look up directly. Generic words
# such as "code" or "bug" are useful for routing but not enough to skip design
# grounding for a substantial request.
_CONCRETE_TARGET_RE = re.compile(
    r"(?:^|\s)(?:src|tests?)/[^\s,;]+|"
    r"\b\w+\.(?:py|js|ts|rs|go|java|c|cpp|h)\b|"
    r"`[^`]+`|"
    r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b"
)
_BEHAVIOR_CONTRACT_RE = re.compile(
    r"\b(?:must|should|when|raise|return|accept|reject|preserve|without|"
    r"cover(?:s|ing)?|cases?|debe|cuando|lanzar|devolver|aceptar|rechazar|"
    r"preservar|sin\s+cambiar|casos?)\b",
    re.IGNORECASE,
)
_REFERENCED_CONTINUATION_RE = re.compile(
    r"\b(?:continue|resume|continua|continuar|reanuda|reanudar)\b",
    re.IGNORECASE,
)
_REFERENCED_FILE_RE = re.compile(
    r"(?<![\w/])(?P<path>[\w./-]+\.(?:md|txt|rst))(?![\w/])",
    re.IGNORECASE,
)


def explicit_execution_score(user_input: str) -> int:
    """Score high-confidence work intent without spending an LLM call."""
    text = (user_input or "").strip()
    if not text or _INFORMATIONAL_RE.search(text) or _NO_EXECUTION_RE.search(text):
        return 0
    score = 0
    if _EXECUTION_ACTION_RE.search(text):
        score += 2
    if _EXECUTION_TARGET_RE.search(text):
        score += 1
    if _EXECUTION_VERIFY_RE.search(text):
        score += 1
    return score


def is_grounded_execution_request(user_input: str) -> bool:
    """Whether the request already supplies enough contract to skip elaboration.

    This intentionally requires more evidence than conversational routing:
    implementation intent, verification, a directly addressable code target,
    and at least one behavioral constraint. Ambiguous requests such as "fix
    the bug in auth.py and run tests" still receive the grounding pass.
    """
    text = (user_input or "").strip()
    return (
        explicit_execution_score(text) >= 4
        and _CONCRETE_TARGET_RE.search(text) is not None
        and _BEHAVIOR_CONTRACT_RE.search(text) is not None
    )


def is_referenced_continuation_request(user_input: str) -> bool:
    """Whether the request asks to continue work from a named text brief."""
    text = (user_input or "").strip()
    return bool(
        _REFERENCED_CONTINUATION_RE.search(text)
        and _REFERENCED_FILE_RE.search(text)
    )


def referenced_file_paths(user_input: str) -> tuple[str, ...]:
    """Return unique text-brief paths named by the request, in order."""
    return tuple(dict.fromkeys(
        match.group("path")
        for match in _REFERENCED_FILE_RE.finditer(user_input or "")
    ))


def resolve_referenced_repository(
    user_input: str,
    workspace_path: str | None,
) -> str | None:
    """Resolve one nested Git repository containing a referenced brief.

    The file workspace remains unchanged so literal paths in the request keep
    working. This resolver supplies a separate cwd for Git and shell tools.
    It fails closed when paths escape the workspace or name multiple repos.
    """
    if not workspace_path:
        return None
    workspace = os.path.realpath(os.path.expanduser(workspace_path))
    if not os.path.isdir(workspace):
        return None

    repositories: set[str] = set()
    for raw_path in referenced_file_paths(user_input):
        candidate = os.path.realpath(os.path.join(workspace, raw_path))
        try:
            if os.path.commonpath((workspace, candidate)) != workspace:
                continue
        except ValueError:
            continue
        if not os.path.exists(candidate):
            continue
        current = candidate if os.path.isdir(candidate) else os.path.dirname(candidate)
        while True:
            if os.path.exists(os.path.join(current, ".git")):
                repositories.add(current)
                break
            if current == workspace:
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

    return next(iter(repositories)) if len(repositories) == 1 else None


__all__ = [
    "explicit_execution_score",
    "is_grounded_execution_request",
    "is_referenced_continuation_request",
    "referenced_file_paths",
    "resolve_referenced_repository",
]
