"""Tool-call fingerprinting for stuck-pattern detection.

The hand-coded detectors in :mod:`detectors` each look for ONE
specific failure mode (a regression after edit, a search loop, etc.)
and require ~10-30 LOC of bespoke logic. They work well for the
patterns we've already seen, but adding a new one is friction.

This module takes a different angle: reduce a step's tool calls to a
short canonical string (the **fingerprint**), then match the string
against a library of anti-pattern signatures using regex (cheap,
precise) or fuzzy comparison (forgiving, catches near-matches).

The result is a single detector that scales by appending one line to
:data:`ANTI_PATTERNS` per new failure mode. Tomorrow when the model
develops a new pathological pattern, we record the trace from one
bad run, paste it in as a fuzzy example, and we're done.

Alphabet (one letter per tool, ~15 letters cover everything):

================  =====================================================
``R``             read_file (and the partial_read alias)
``S``             code_search, glob
``G``             get_symbol_code
``L``             find_definition, find_references, list_symbols,
                  search_symbols, find_similar_methods, search_by_docstring,
                  iter_symbols, project_stats, project_structure, analyze_code
``D``             list_directory
``E``             replace_lines, edit_symbol, edit_method, apply_patch,
                  add_content_after_line, add_content_before_line,
                  rename_symbol, move_symbol
``C``             create_file, write_file
``M``             add_symbol, add_method, remove_symbol, remove_method
``A``             multi_edit_file, edit_file (file-level edit)
``+``             add_step
``~``             modify_step, remove_step
``.``             step_complete
``X``             execute_command
``T``             think
``N``             add_note, add_session_note
``F``             record_finding, search_findings, read_findings, knowledge ops
``W``             web_search, web_fetch
``H``             help, explain_tool
``I``             code_interpreter
``V``             git_branch, git_commit, git_diff, git_status, git_push
``U``             send_message (user-facing chat)
``?``             anything not mapped (rare)
================  =====================================================

Letters are picked so the **edit-class letters** (E, C, M, A) form a
distinct group. That makes it trivial to write "trace contains NO
edit" as ``[^ECMA]*`` in regex.

Example traces from real runs:

* Productive: ``RGE.``  read file → get symbol → edit → done
* Productive: ``+RREE.``  plan → read twice → edit twice → done
* **Bad**:    ``+++++R+RR+R.``  plan/read alternation, never edits
* **Bad**:    ``RSRSRSRR.``  search/read pingpong, never edits
* **Bad**:    ``XXXXXXR.``  shell command spam (env debugging)
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any, Iterable

# ─────────────────────────────────────────────────────────────────────────
# Alphabet
# ─────────────────────────────────────────────────────────────────────────

_TOOL_LETTER_MAP: dict[str, str] = {
    # Read
    "read_file": "R",
    "partial_read": "R",
    # Search
    "code_search": "S",
    "glob": "S",
    # Get symbol body
    "get_symbol_code": "G",
    # Lookups (cheap index queries)
    "find_definition": "L",
    "find_references": "L",
    "list_symbols": "L",
    "search_symbols": "L",
    "find_similar_methods": "L",
    "search_by_docstring": "L",
    "iter_symbols": "L",
    "project_stats": "L",
    "project_structure": "L",
    "analyze_code": "L",
    # Directory
    "list_directory": "D",
    # Edit (line-level)
    "replace_lines": "E",
    "add_content_after_line": "E",
    "add_content_before_line": "E",
    "edit_symbol": "E",
    "edit_method": "E",
    "apply_patch": "E",
    "rename_symbol": "E",
    "move_symbol": "E",
    # Create new file
    "create_file": "C",
    "write_file": "C",
    # Add/remove symbol
    "add_symbol": "M",
    "add_method": "M",
    "remove_symbol": "M",
    "remove_method": "M",
    # File-level edit
    "edit_file": "A",
    "multi_edit_file": "A",
    # Plan management
    "add_step": "+",
    "modify_step": "~",
    "remove_step": "~",
    # Loop terminator
    "step_complete": ".",
    # Shell
    "execute_command": "X",
    # Notes / thinking
    "think": "T",
    "add_note": "N",
    "add_session_note": "N",
    # Knowledge
    "record_finding": "F",
    "search_findings": "F",
    "read_findings": "F",
    "validate_finding": "F",
    "reject_finding": "F",
    "update_finding": "F",
    "delete_finding": "F",
    "search_knowledge": "F",
    "summarize_findings": "F",
    "write_report": "F",
    "read_report": "F",
    "delete_report": "F",
    "find_documentation": "F",
    "update_documentation": "F",
    "delete_documentation": "F",
    # Web
    "web_search": "W",
    "web_fetch": "W",
    "code_search_web": "W",
    # Meta / help
    "help": "H",
    "explain_tool": "H",
    "tail_test_output": "X",
    "declare_test_command": "N",
    # Code interpreter
    "code_interpreter": "I",
    # Git / version control
    "git_branch": "V",
    "git_commit": "V",
    "git_diff": "V",
    "git_status": "V",
    "git_push": "V",
    # User-facing chat
    "send_message": "U",
}


def _letter_for(tool_name: str) -> str:
    """Return the canonical letter for a tool, or '?' if unknown."""
    return _TOOL_LETTER_MAP.get(tool_name, "?")


# ─────────────────────────────────────────────────────────────────────────
# Fingerprint construction
# ─────────────────────────────────────────────────────────────────────────


def build_fingerprint(messages: Iterable[dict]) -> str:
    """Reduce a sequence of conversation messages to a fingerprint string.

    Walks every assistant tool call in order and appends its canonical
    letter. Skips non-assistant messages and messages with no tool
    calls. The resulting string captures the *shape* of the model's
    work — what it did, in what order — without any of the content.
    """
    out: list[str] = []
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            name = fn.get("name") or tc.get("name") or ""
            out.append(_letter_for(str(name)))
    return "".join(out)


# ─────────────────────────────────────────────────────────────────────────
# Anti-pattern library
# ─────────────────────────────────────────────────────────────────────────
#
# Each entry: (name, kind, pattern, advice_key)
#
#   name        — human label for logs/debug
#   kind        — "regex" or "fuzzy"
#   pattern     — regex string (kind=regex) or example trace (kind=fuzzy)
#   advice_key  — key into ``library.GUIDANCE_LIBRARY``
#
# Edit-class letters are E, C, M, A. The macro ``[^ECMA]*`` means
# "any chars except an edit", which is the most common guard.
#
# Patterns are evaluated in order; first match wins.

_NO_EDIT = r"[^ECMA]"

ANTI_PATTERNS: list[tuple[str, str, str, str]] = [
    # ── Read-class loops ────────────────────────────────────────────────
    # 4+ read_file calls in a single step that ends without an edit.
    # The ``\.`` at the end forces this to fire only when step_complete
    # has been called — partial steps don't trigger.
    ("read_loop_no_edit", "regex",
     fr"^{_NO_EDIT}*R{{4,}}{_NO_EDIT}*\.$",
     "reread_loop"),

    # 3+ get_symbol_code with no edit. Model is over-fetching method
    # bodies instead of consolidating them mentally and editing.
    ("symbol_fetch_loop", "regex",
     fr"^{_NO_EDIT}*G{{3,}}{_NO_EDIT}*\.$",
     "stop_reading"),

    # Mix of read_file and get_symbol_code, no edit, ≥5 total. Catches
    # "I need to look at one more thing" forever.
    ("read_get_loop", "regex",
     fr"^{_NO_EDIT}*[RG]{{5,}}{_NO_EDIT}*\.$",
     "stop_reading"),

    # ── Search-class loops ──────────────────────────────────────────────
    # 3+ searches in a row with no edit. The model can't find what it
    # wants and keeps searching with new queries.
    ("search_storm", "regex",
     fr"^{_NO_EDIT}*S{{3,}}{_NO_EDIT}*\.$",
     "stuck_on_search"),

    # Search → read → search → read repeated. Classic exploration
    # rabbit hole — looks productive, isn't.
    ("search_read_pingpong", "regex",
     r"(SR){3,}",
     "stuck_on_search"),
    ("read_search_pingpong", "regex",
     r"(RS){3,}",
     "stuck_on_search"),

    # 3+ index lookups (find_definition, list_symbols, etc.) without
    # an edit. Model is over-querying instead of using results.
    ("lookup_storm", "regex",
     fr"^{_NO_EDIT}*L{{3,}}{_NO_EDIT}*\.$",
     "stop_reading"),

    # ── Planning-class loops ────────────────────────────────────────────
    # 4+ add_step in a row. Pure analysis paralysis. The
    # stop_planning_start_coding detector catches this in the message
    # buffer; the fingerprint catches it from the trace shape too.
    ("plan_storm", "regex",
     r"\+{4,}",
     "stop_planning_start_coding"),

    # Plan → read → plan → read repeated. Incremental procrastination
    # — the model adds a step, peeks at code, adds another step,
    # peeks again, never editing. This is the django-11179 pattern.
    ("plan_read_loop", "regex",
     fr"^{_NO_EDIT}*(\+R){{2,}}{_NO_EDIT}*\.$",
     "stop_planning_start_coding"),
    ("read_plan_loop", "regex",
     fr"^{_NO_EDIT}*(R\+){{2,}}{_NO_EDIT}*\.$",
     "stop_planning_start_coding"),

    # 3+ modify_step / remove_step in a row. Replan churn — the model
    # is rewriting its own plan instead of executing it.
    ("replan_churn", "regex",
     r"~{3,}",
     "stop_planning_start_coding"),

    # ── Shell-class loops ───────────────────────────────────────────────
    # 3+ execute_command in a row with no edit. Most commonly env
    # debugging loops (the python_env_mismatch detector handles the
    # ImportError variant; this catches the more general "running
    # commands without editing the code that needs fixing").
    ("shell_no_edit", "regex",
     fr"^{_NO_EDIT}*X{{3,}}{_NO_EDIT}*\.$",
     "stuck_on_tests"),

    # Read → shell → read → shell → ... → step_complete with no edit.
    # Looks like "I'm investigating" but produces nothing.
    ("read_shell_loop", "regex",
     fr"^{_NO_EDIT}*(RX|XR){{2,}}{_NO_EDIT}*\.$",
     "stuck_on_tests"),

    # ── Directory loops ─────────────────────────────────────────────────
    # 3+ list_directory in a row. Browsing paralysis.
    ("dir_browse_loop", "regex",
     r"D{3,}",
     "stop_reading"),

    # ── Note-taking loops (model writes notes instead of editing) ───────
    # 3+ add_note in a row with no edit. Capture-without-act pattern.
    ("note_storm", "regex",
     fr"^{_NO_EDIT}*N{{3,}}{_NO_EDIT}*\.$",
     "stop_reading"),

    # ── Help loops ──────────────────────────────────────────────────────
    # 3+ help calls in a row. Model is reading docs forever.
    ("help_storm", "regex",
     r"H{3,}",
     "stop_reading"),

    # ── Mixed long step with zero edits ─────────────────────────────────
    # Catch-all: 8+ tool calls of any non-edit kind, ending with
    # step_complete. The minimum length avoids firing on legit short
    # exploration steps.
    ("long_step_no_edit", "regex",
     fr"^{_NO_EDIT}{{8,}}\.$",
     "stop_reading"),

    # ────────────────────────────────────────────────────────────────────
    # Fuzzy patterns — match traces that LOOK like these examples with
    # ~70% similarity. Each example is at least 6 chars long because
    # SequenceMatcher gives high ratios for short strings even when
    # they're substantively different. Fuzzy match shines when the
    # model adds noise (a stray ``T`` for thinking, an ``N`` for note)
    # to an otherwise-recognised pattern.
    # ────────────────────────────────────────────────────────────────────

    # Incremental planning: add step → read → add step → read forever.
    ("fuzzy_incremental_planning", "fuzzy",
     "+R+R+R+R.",
     "stop_planning_start_coding"),

    # Plan → read → plan → run shell → plan → run shell. The
    # env-debug-loop fingerprint with planning sprinkled in.
    ("fuzzy_plan_shell_debug", "fuzzy",
     "+RX+RX+RX.",
     "stuck_on_tests"),

    # Reads then plan-loop. Model read enough but won't commit.
    ("fuzzy_read_then_plan_storm", "fuzzy",
     "RR++++++.",
     "stop_planning_start_coding"),

    # Search → read → search → read → search → read with no edit.
    # The same as the regex pingpong but the fuzzy version catches
    # "almost the same" with one or two extra letters of noise.
    ("fuzzy_search_read_loop", "fuzzy",
     "SRSRSRSR.",
     "stuck_on_search"),

    # Get-symbol-code over-fetching with light reads sprinkled in.
    ("fuzzy_symbol_overload", "fuzzy",
     "GRGRGR.",
     "stop_reading"),

    # Replan-then-replan-then-replan. modify_step / remove_step heavy.
    ("fuzzy_replan_loop", "fuzzy",
     "+~+~+~+.",
     "stop_planning_start_coding"),

    # Shell command storm with reads sprinkled in. The ``X`` represents
    # any of: pytest, python -c, pip install, ls, etc.
    ("fuzzy_shell_read_storm", "fuzzy",
     "XRXRXRXR.",
     "stuck_on_tests"),

    # Deep lookup-only step. find_references / list_symbols / etc.
    ("fuzzy_lookup_drill", "fuzzy",
     "LRLRLRLR.",
     "stop_reading"),
]

# Threshold for fuzzy match. Tuned so:
#  - "+R+R+R+R."  vs trace "+R+R+R+R."   → 1.00  ✓
#  - "+R+R+R+R."  vs trace "+R+R+RT+R."  → ~0.84 ✓ (extra T from think)
#  - "+R+R+R+R."  vs trace "RRRRRRRR."   → ~0.55 ✗ (too different)
#  - "+R+R+R+R."  vs trace "RGE."        → ~0.20 ✗ (productive)
_FUZZY_THRESHOLD = 0.70


# ─────────────────────────────────────────────────────────────────────────
# Public matcher
# ─────────────────────────────────────────────────────────────────────────


def match_antipattern(trace: str) -> str | None:
    """Return the advice_key of the first matching anti-pattern, or None.

    Patterns are evaluated in declaration order; the first match wins.
    Regex patterns are checked first via ``re.search``; fuzzy patterns
    via ``difflib.SequenceMatcher.ratio()`` with threshold
    :data:`_FUZZY_THRESHOLD`.
    """
    if not trace:
        return None
    for name, kind, pattern, advice_key in ANTI_PATTERNS:
        try:
            if kind == "regex":
                if re.search(pattern, trace):
                    return advice_key
            elif kind == "fuzzy":
                ratio = SequenceMatcher(None, trace, pattern).ratio()
                if ratio >= _FUZZY_THRESHOLD:
                    return advice_key
        except re.error:
            # Bad pattern in the library — never crash the loop, skip.
            continue
    return None


def detect_fingerprint_antipattern(messages: list[dict], state: Any = None) -> str | None:
    """Detector entry point: build the fingerprint and try to match.

    Returns the matched advice key or None. Slots into the
    ``detectors._DETECTORS`` priority list as the catch-all that runs
    after every hand-coded detector has had its chance.
    """
    trace = build_fingerprint(messages)
    return match_antipattern(trace)


__all__ = [
    "ANTI_PATTERNS",
    "build_fingerprint",
    "detect_fingerprint_antipattern",
    "match_antipattern",
]
