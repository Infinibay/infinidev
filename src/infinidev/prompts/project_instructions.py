"""``AGENTS.md`` — the project's own instructions to the agent.

A convention rather than a format: a Markdown file at the root of a
repository telling whatever agent is working in it what this project
expects. Codex and OpenCode read it, ken's install docs point at it, and it
is the one place a user can say "in this repo, do it *this* way" without
touching infinidev's own prompts.

Three decisions worth stating, because each one is a place this could have
gone wrong:

**It loads on every run, not at import.** The file is the user's live
control surface — editing it and re-running has to be enough, exactly like
``settings.json``.

**It is capped.** A file this size lands in the cacheable prefix of every
single request for the whole session, so an unbounded read hands a project
the ability to spend the entire context window before the agent has done
anything. The cap truncates on a paragraph boundary and says so, which is
more useful than silently sending half a sentence.

**It never raises.** An unreadable or badly-encoded AGENTS.md degrades to
"no project instructions", never to a failed run.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# The canonical name, then the ones a project may already be carrying. The
# first hit wins: a repository with both is telling us AGENTS.md is the one
# written for any agent, and CLAUDE.md the one written for a specific host.
INSTRUCTION_FILENAMES: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")

# ~2k tokens. Enough for real project conventions, short of a design doc.
MAX_CHARS = 8000


def _resolve_workspace(workspace: str | os.PathLike[str] | None) -> str | None:
    """Fall back to the bound tool context, which is where the workspace
    actually lives — no caller has to thread it through to get this."""
    if workspace:
        return str(workspace)
    try:
        from infinidev.tools.base.context import get_current_workspace_path

        return get_current_workspace_path() or None
    except Exception:  # pragma: no cover - context not initialised
        return None


def find_instruction_file(workspace: str | os.PathLike[str] | None) -> str | None:
    """Absolute path of the project's instruction file, if it has one."""
    workspace = _resolve_workspace(workspace)
    if not workspace:
        return None
    for name in INSTRUCTION_FILENAMES:
        candidate = os.path.join(str(workspace), name)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_project_instructions(
    workspace: str | os.PathLike[str] | None,
) -> str | None:
    """The project's instructions, ready to embed in a system prompt.

    Returns ``None`` when the project has no instruction file, when it is
    empty, or when anything at all goes wrong reading it.
    """
    path = find_instruction_file(workspace)
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read(MAX_CHARS * 2 + 1)
    except OSError:
        logger.warning("could not read project instructions at %s", path, exc_info=True)
        return None

    text = text.strip()
    if not text:
        return None
    return _truncate(text, os.path.basename(path))


def _truncate(text: str, name: str) -> str:
    """Cap the text, breaking on a paragraph rather than mid-sentence."""
    if len(text) <= MAX_CHARS:
        return text
    head = text[:MAX_CHARS]
    cut = head.rfind("\n\n")
    if cut > MAX_CHARS // 2:
        head = head[:cut]
    dropped = len(text) - len(head)
    return (
        head.rstrip()
        + f"\n\n[{name} truncated here — {dropped} more characters. "
        f"Instructions past this point were not loaded.]"
    )


def render_project_instructions(
    workspace: str | os.PathLike[str] | None,
) -> str | None:
    """The instructions wrapped in the tag the prompt uses, or ``None``.

    Tagged like every other injected block so the model can tell the
    project's words from infinidev's own, and attribute the authority
    correctly: these are the user's standing instructions for this
    repository, not a suggestion from the harness.
    """
    body = load_project_instructions(workspace)
    if body is None:
        return None
    name = os.path.basename(find_instruction_file(workspace) or "AGENTS.md")
    return (
        f"<project-instructions source=\"{name}\">\n"
        "Standing instructions from this project's maintainers. They describe "
        "how work is done in this repository specifically, and they win over "
        "the general guidance above wherever the two differ.\n\n"
        f"{body}\n"
        "</project-instructions>"
    )
