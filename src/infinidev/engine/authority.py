"""Authority and provenance labels for model-derived task artifacts."""

from __future__ import annotations

from typing import Literal, TypeAlias


AuthorityLevel: TypeAlias = Literal[
    "user_explicit",
    "user_confirmed",
    "repo_derived",
    "model_inferred",
    "default_reversible",
    "unverified_assumption",
]

USER_AUTHORITY_LEVELS: frozenset[AuthorityLevel] = frozenset(
    {"user_explicit", "user_confirmed"}
)


def is_user_authorized(authority: AuthorityLevel) -> bool:
    """Whether an artifact is backed by direct or confirmed user authority."""

    return authority in USER_AUTHORITY_LEVELS
