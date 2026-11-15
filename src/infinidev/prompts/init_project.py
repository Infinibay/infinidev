"""Capability-aware prompts for bounded, read-only project exploration."""

from __future__ import annotations

import json
from collections.abc import Collection

_DESCRIPTION = """\
## Task: Explore and Document This Project

Produce a useful first slice of project knowledge, starting with existing README,
docs, ADRs and CONTEXT files. Follow an important entry point through its callers,
implementation and tests that exercise that path. Prefer a coherent explanation over an inventory
of every file or symbol. State which areas remain unexplored.

## Scope and Evidence

Adapt exploration to this project: purpose, intended users and use cases; domain
concepts, rules and invariants; architecture and integrations; data and lifecycle;
security controls and trust boundaries; configuration, build, tests, deployment
and design decisions. A category without evidence stays unknown or unexplored;
do not invent business features to fill headings.

Distinguish documented intent, facts observed in source, hypothesis and unknowns.
Cite the paths and qualified symbols supporting each conclusion. Reading source or a test does not
establish runtime correctness or certify security. Record contradictory evidence
and assumptions rather than silently choosing one account. Do not require a
numeric confidence of 1.0 merely because you read code.

Explore without creating or modifying project files, running build/test commands,
indexing or downloading external documentation. Those effects require separate
explicit authorization. Do not read credential files or store secrets, private
reasoning, transcripts, book extracts or local-library contents.

## Finding Storage

{storage}

The final offered schema is authoritative: the initial toolbox can change during
routing or discovery. Never call a tool absent from that schema or guess its
arguments. If the selected storage tool is unavailable, denied or fails, report
unrecorded findings publicly; do not silently switch backends or claim persistence.
Keep one small conclusion per finding and reuse existing identity when supported.
A storage acknowledgement means saved, not reviewed, true or currently valid.
Do not invent hashes, validity checks or revision history the backend does not
provide. Preserve existing evidence when updating; report conflicts explicitly.

## Language and Delivery

{language}

Keep paths, identifiers and commands unchanged. Write concise, original findings;
no books or PDFs are needed. Do not depend on add_note, documentation download or
any other tool that is not actually offered. Use available read/search tools within
their schemas and keep the exploration bounded by relevance.

Deliver a self-contained public summary: what was inspected, useful conclusions
with source references, what storage actually acknowledged, unrecorded findings,
contradictions and the next useful slice. Never claim exhaustive documentation,
durable exploration checkpoints or complete coverage from this task alone.
"""

_EXPECTED = (
    "A self-contained project exploration summary in the user's language, with "
    "source references, facts distinguished from hypotheses, acknowledged storage "
    "separated from unrecorded findings, and explicit gaps and next steps."
)


def build_init_task_prompt(
    *,
    tool_names: Collection[str] | None = None,
    language: str | None = None,
) -> tuple[str, str]:
    """Compose guidance from the initial toolbox, without discovering tools.

    Names select schema-specific advice only; the model must still check the
    final offered schema. None preserves capability-neutral legacy constants.
    """
    if tool_names is None:
        storage = (
            "Check the offered finding-storage schemas first. If ken_remember is "
            "offered, use its own schema; otherwise use record_finding only if "
            "offered, with its local schema. Select one backend and do not mix "
            "their arguments. If neither is offered, deliver findings publicly "
            "and do not claim database persistence."
        )
    elif "ken_remember" in tool_names:
        storage = (
            "Use ken_remember only if it remains offered, following its own "
            "offered schema for topic, content and justification when supported. Local finding_type "
            "and confidence arguments do not belong to this backend."
        )
    elif "record_finding" in tool_names:
        storage = (
            'Use record_finding only if it remains offered. Its local schema has '
            'finding_type="project_context" for project context; use hypothesis '
            'for conjectures and do not force certainty. New findings are '
            'provisional. Include source references using the offered schema.'
        )
    else:
        storage = (
            "No finding storage tool is offered in the initial toolbox. Deliver "
            "findings in the public summary. Do not claim database persistence."
        )
    if language and language.strip():
        language_guidance = (
            "Requested output language (JSON string): "
            + json.dumps(language.strip(), ensure_ascii=False)
            + ". Use this language for findings and the public report."
        )
    else:
        language_guidance = (
            "Use the user's explicit language preference from the conversation "
            "for findings and the public report, not the language of this prompt "
            "or the source code. If no preference is available, ask before "
            "persisting language-dependent findings; report that choice as "
            "pending if interaction is unavailable."
        )
    return _DESCRIPTION.format(storage=storage, language=language_guidance), _EXPECTED


# Compatibility for hosts that import the original constant tuple components.
INIT_TASK_DESCRIPTION, INIT_EXPECTED_OUTPUT = build_init_task_prompt()
