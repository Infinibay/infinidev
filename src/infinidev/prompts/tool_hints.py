"""Conditional prompt generation based on available tools.

Generates tool usage sections, editing rules, and examples that only
reference tools actually available to the model.  This prevents small
models from hallucinating tools they've seen in training data but
don't have access to.
"""

from __future__ import annotations


# ── Tool descriptions for prompt injection ───────────────────────────────
# Each entry: (one-line description, usage hint)

TOOL_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    # File I/O
    "read_file": (
        "Read a file with line numbers (pass offset/limit for a range)",
        "read_file(file_path='src/main.py', offset=10, limit=41)",
    ),
    "create_file": (
        "Create a NEW file (fails if file exists)",
        "create_file(file_path='src/new.py', content='...')",
    ),
    "edit_file": (
        "Change an existing file by replacing exact text (must be unique; "
        "empty new_string deletes)",
        "edit_file(file_path='src/main.py', old_string='timeout=30', new_string='timeout=120')",
    ),
    "list_directory": (
        "List directory contents",
        "list_directory(file_path='src/')",
    ),
    "code_search": (
        "Search code by pattern (regex supported)",
        "code_search(pattern='def verify_token', file_path='src/')",
    ),
    "glob": (
        "Find files by glob pattern",
        "glob(pattern='**/*.py')",
    ),
    # Git
    "git_branch": (
        "Create or checkout a branch",
        "git_branch(branch_name='fix-auth', create=True)",
    ),
    "git_commit": (
        "Stage and commit selected files, or all changes when files is omitted",
        "git_commit(message='Fix auth expiry check')",
    ),
    "git_diff": (
        "Show uncommitted changes",
        "git_diff()",
    ),
    "git_status": (
        "Show working tree status",
        "git_status()",
    ),
    # Shell
    "execute_command": (
        "Run a shell command",
        "execute_command(command='python -m pytest tests/ -x -q', "
        "rationale='Run the project tests and stop at the first failure')",
    ),
    "code_interpreter": (
        "Run Python code in a sandbox. Great for analyzing, parsing, "
        "or querying the codebase (count methods, measure spans, aggregate "
        "symbols). 13 code-intel helpers pre-imported — use `help` tool for details.",
        'code_interpreter(code=\'rows = iter_symbols(kind="method", parent="Foo")\\nprint(len(rows))\')',
    ),
    "run_in_background": (
        "Start a long-running command in the background (dev server, watcher) "
        "and keep working. Returns a task id tracked in <background-tasks>.",
        "run_in_background(command='npm run dev', description='vite dev server')",
    ),
    "background_status": (
        "Check a background task: status, runtime, and captured stdout/stderr "
        "(omit task_id to list all)",
        "background_status(task_id='bg-1')",
    ),
    "stop_background_task": (
        "Stop a background task (force=True to SIGKILL immediately)",
        "stop_background_task(task_id='bg-1')",
    ),
    "wait_for_background_task": (
        "BLOCK until a background task finishes (or prints a readiness marker "
        "via until_text), instead of polling background_status in a loop. "
        "Bounded by a timeout; returns timed_out=True if it elapses.",
        "wait_for_background_task(task_id='bg-1', until_text='Listening on')",
    ),
    "iter_symbols": (
        "Walk all indexed symbols (no search term needed)",
        "iter_symbols(kind='method', parent='UserService')",
    ),
    "project_stats": (
        "Summary of files / symbols / languages in the index",
        "project_stats()",
    ),
    # Web
    "web_search": (
        "Search the web for documentation, APIs, or solutions",
        "web_search(query='python requests timeout')",
    ),
    "web_fetch": (
        "Fetch and read a web page",
        "web_fetch(url='https://docs.python.org/3/...')",
    ),
    "code_search_web": (
        "Search the web specifically for code, API usage, and error solutions",
        "code_search_web(query='fastapi background task example')",
    ),
    # Knowledge
    "record_finding": (
        "Save a finding to the knowledge base",
        "record_finding(title='auth module', content='uses JWT with HS256')",
    ),
    "search_findings": (
        "Search saved findings",
        "search_findings(query='auth')",
    ),
    "search_knowledge": (
        "Full-text search across saved knowledge (findings + reports); "
        "omit the query to browse findings by filter",
        "search_knowledge(query='auth & token*') | "
        "search_knowledge(finding_type='project_context')",
    ),
    "update_finding": (
        "Edit the content/topic of an existing finding by id",
        "update_finding(finding_id=12, content='uses JWT RS256, not HS256')",
    ),
    "validate_finding": (
        "Mark a finding as verified/confirmed",
        "validate_finding(finding_id=12)",
    ),
    "reject_finding": (
        "Mark a finding as wrong/rejected (keeps it for audit)",
        "reject_finding(finding_id=12, reason='superseded by newer finding')",
    ),
    "delete_finding": (
        "Permanently delete a finding by id",
        "delete_finding(finding_id=12)",
    ),
    "summarize_findings": (
        "Condense the session's findings into a compact summary",
        "summarize_findings()",
    ),
    "write_report": (
        "Save a longer structured report (markdown) as an artifact",
        "write_report(title='Auth audit', content='## Findings\\n...')",
    ),
    "read_report": (
        "Read a saved report by id (omit id to list reports)",
        "read_report(report_id=3)",
    ),
    "read_command_output": (
        "Read a bounded byte range from a private execute_command output handle",
        "read_command_output(artifact_id=7, type='command_output', "
        "stream='stdout', char_count=12000, byte_count=12000, offset=0)",
    ),
    "delete_report": (
        "Delete a saved report artifact by id",
        "delete_report(artifact_id=3)",
    ),
    # Library documentation cache
    "find_documentation": (
        "Look up cached documentation for a library",
        "find_documentation(library_name='fastapi', query='background tasks')",
    ),
    "update_documentation": (
        "Fetch and cache structured documentation for a library",
        "update_documentation(library_name='fastapi', version='latest')",
    ),
    "delete_documentation": (
        "Remove cached documentation for a library",
        "delete_documentation(library_name='fastapi')",
    ),
    # Code intelligence
    "find_references": (
        "Find ALL places where a symbol is used",
        "find_references(name='verify_token')",
    ),
    "list_symbols": (
        "List all functions/classes in a file",
        "list_symbols(file_path='src/auth.py')",
    ),
    "search_symbols": (
        "Search symbols by name across the project",
        "search_symbols(query='verify')",
    ),
    "get_symbol_code": (
        "Get the full source code of a function/method by name",
        "get_symbol_code(name='AuthService.verify_token')",
    ),
    "project_structure": (
        "Show directory tree with file descriptions",
        "project_structure(file_path='src/')",
    ),
    "analyze_code": (
        "Detect broken imports, undefined symbols, unused code",
        "analyze_code(file_path='src/auth.py')",
    ),
    "rename_symbol": (
        "Rename a symbol everywhere it is referenced (definition + call sites)",
        "rename_symbol(symbol='AuthService.verify_token', new_name='check_token')",
    ),
    "move_symbol": (
        "Move a function/method to another file, updating imports",
        "move_symbol(symbol='helpers.slugify', target_file='src/text_utils.py')",
    ),
    "find_similar_methods": (
        "Find methods structurally similar to a given one (duplication hunting)",
        "find_similar_methods(qualified_name='UserService.create')",
    ),
    "search_by_docstring": (
        "Semantic search for symbols by what they DO, not their name",
        "search_by_docstring(query='validate an auth token and return claims')",
    ),
    # Plan management (developer loop pseudo-tools)
    "add_step": (
        "Add a step to the execution plan (name the FILE, FUNCTION, and CHANGE)",
        "add_step(title='auth.py verify_token: add expiry check')",
    ),
    "modify_step": (
        "Edit a pending step's title/detail by index",
        "modify_step(index=2, title='auth.py: also handle missing exp claim')",
    ),
    "remove_step": (
        "Remove a pending step from the plan by index",
        "remove_step(index=3)",
    ),
    # Project introspection
    "declare_test_command": (
        "Tell the engine which command runs this project's tests",
        "declare_test_command(command_pattern='pytest')",
    ),
    "tail_test_output": (
        "Re-read the most recent test run's output (failures, full, or tail)",
        "tail_test_output(mode='failures')",
    ),
    # Communication
    "send_message": (
        "Send a message to the user (progress update or a question)",
        "send_message(message='Found the bug in auth.py:42 — fixing now')",
    ),
    # Meta
    "help": (
        "Get detailed help and examples for any tool",
        "help(context='edit')",
    ),
    "recall_context": (
        "Retrieve tool output from earlier steps that left your context",
        "recall_context(query='the failing assertion from the auth test')",
    ),
    "view_image": (
        "Load an image so the next turn can see it (vision models only)",
        "view_image(file_path='docs/architecture.png')",
    ),
    # Engine pseudo-tools (always available)
    "step_complete": (
        "End current step (REQUIRED after each step)",
        "step_complete(summary='...', status='continue')",
    ),
    "add_note": (
        "Save a note that persists across steps",
        "add_note(note='verify_token at line 42')",
    ),
    "add_session_note": (
        "Save a note that persists across tasks in this session",
        "add_session_note(note='Auth uses JWT RS256, verify_token at src/auth/jwt.py:42')",
    ),
}


# ── MCP-provided tools ───────────────────────────────────────────────────
#
# These arrive from an MCP server at runtime and already carry the server's
# own description, so they need no entry here to be usable. The hints below
# exist for a different reason: a tool the model has to *discover* in a
# ninety-entry schema is a tool it does not reach for. Naming the handful
# worth using unprompted — and saying when — is what turns a registered
# tool into a used one.
#
# They live in their own dict because whether the matching tool exists
# depends on which servers are configured, so the staleness guard in
# ``tests/test_tool_docs_complete.py`` must not treat them as dead entries
# on a machine with no Ken installed.
# Hints for tools a *server* owns, so they are coupled to that server's
# version in a way local hints are not: ken renaming its surface silently
# retires every entry here, and the catalog's staleness guard cannot see it
# (it exempts MCP names, because a missing server is a deployment fact, not
# rot). The bridge already renders each server's own description as a
# fallback — these exist only to add the one thing a description cannot: a
# call that shows the argument shape.
MCP_TOOL_HINTS: dict[str, tuple[str, str]] = {
    "ken_find": (
        "Find things by describing them — scope picks what is searched: "
        "files, symbols, text, tests, wiring, intent",
        "ken_find(query='where provider retries are backed off', scope='files')",
    ),
    "ken_read": (
        "Read an indexed file's structure, and its source when you ask for it",
        "ken_read(path='src/infinidev/engine/loop/engine.py', include=['symbols'])",
    ),
    "ken_related": (
        "What else is connected to a file or symbol, by a named relation: "
        "blast_radius, callers, callees, cochange, clones, imports, neighbors",
        "ken_related(target='src/infinidev/engine/loop/engine.py', relation='blast_radius')",
    ),
    "ken_rank": (
        "What matters right now — ken's own ordering, not a search. Scopes: "
        "session, changes, project, architecture",
        "ken_rank(scope='session', verbose=1)",
    ),
    "ken_recall": (
        "Recall findings saved in earlier sessions for this project",
        "ken_recall(query='auth token lifetime', limit=5)",
    ),
    "ken_remember": (
        "Save a durable finding so future sessions start warm",
        "ken_remember(topic='jwt-clock-skew', content='Tokens allow 60s skew (src/auth/jwt.py:88)')",
    ),
}

TOOL_DESCRIPTIONS.update(MCP_TOOL_HINTS)


# Terminators and engine pseudo-tools. They are explained by the loop
# protocol section, not by the tool catalog, so the catalog's catch-all must
# leave them alone.
_PROTOCOL_TOOLS = {
    "step_complete",
    "add_note",
    "add_session_note",
    "respond",
    "escalate",
    "emit_plan",
    "emit_verdict",
    "channel_post",
    "conclude",
    "seed_council",
    "council_verdict",
    "synthesize_brief",
}


# ── Editing tool groups ──────────────────────────────────────────────────
# Used to generate conditional editing rules

# One tool changes file contents; the symbol pair are project-wide refactors.
_EDIT_TOOLS_SURGICAL = {"edit_file"}
_EDIT_TOOLS_SYMBOL = {"rename_symbol", "move_symbol"}


def _mcp_descriptions() -> dict[str, str]:
    """Name → description for every MCP tool currently exposed.

    Read from the bridge's cache rather than passed down the call chain:
    the identity prompt is built from a set of *names*, several layers away
    from the tool instances, and threading them through every flow just to
    caption a list would be a lot of plumbing for one paragraph.
    """
    try:
        from infinidev.tools.mcp_bridge import discover_mcp_tool_classes

        return {
            cls.model_fields["name"].default:
                (cls.model_fields["description"].default or "").strip()
            for cls in discover_mcp_tool_classes()
        }
    except Exception:  # pragma: no cover - prompts must never fail on this
        return {}


def build_tool_usage_section(
    available_tools: set[str], tools: list | None = None
) -> str:
    """Generate a '## Tool Usage' prompt section listing only available tools.

    Groups tools by category and includes usage hints.

    ``tools`` is optional and carries the instances themselves. It matters
    for MCP-provided tools, whose descriptions live on the server rather
    than in this file: without it they would be listed as bare names, which
    tells the model a tool exists but nothing about when to use it. When it
    is not supplied the descriptions are read from the MCP bridge's own
    cache, so callers that only have names still get usable output.
    """
    own_descriptions = _mcp_descriptions()
    own_descriptions.update({
        t.name: (getattr(t, "description", "") or "").strip()
        for t in (tools or [])
        if getattr(t, "name", None)
    })
    categories = [
        (
            "Reading",
            [
                "read_file",
                "list_directory",
                "glob",
                "code_search",
                "get_symbol_code",
                "list_symbols",
                "search_symbols",
                "find_references",
                "find_similar_methods",
                "search_by_docstring",
                "iter_symbols",
                "project_stats",
                "project_structure",
                "analyze_code",
                "view_image",
            ],
        ),
        (
            "Writing",
            [
                "create_file",
                "edit_file",
                "rename_symbol",
                "move_symbol",
            ],
        ),
        (
            "Execution",
            [
                "execute_command",
                "code_interpreter",
                "run_in_background",
                "background_status",
                "stop_background_task",
                "wait_for_background_task",
            ],
        ),
        ("Git", ["git_branch", "git_commit", "git_diff", "git_status"]),
        ("Web", ["web_search", "web_fetch", "code_search_web"]),
        (
            "Knowledge",
            [
                "record_finding",
                "search_findings",
                "search_knowledge",
                "update_finding",
                "validate_finding",
                "reject_finding",
                "delete_finding",
                "summarize_findings",
                "write_report",
                "read_report",
                "read_command_output",
                "delete_report",
            ],
        ),
        (
            "Library docs",
            ["find_documentation", "update_documentation", "delete_documentation"],
        ),
        (
            "Planning",
            [
                "add_step",
                "modify_step",
                "remove_step",
                "declare_test_command",
                "tail_test_output",
            ],
        ),
        ("Communication", ["send_message"]),
        ("Meta", ["help", "recall_context"]),
    ]

    lines = ["## Tool Usage", ""]
    listed: set[str] = set()
    for category, tool_names in categories:
        present = [t for t in tool_names if t in available_tools]
        if not present:
            continue
        listed.update(present)
        lines.append(f"### {category}")
        for name in present:
            desc, example = TOOL_DESCRIPTIONS.get(name, (name, ""))
            lines.append(f"- **{name}**: {desc}")
        lines.append("")

    # Tools that arrived from an MCP server are discovered at runtime, so no
    # static category can name them. Listing them last is the difference
    # between a tool the model knows it has and one that only exists in the
    # JSON schema — and a tool the model does not know about is one it never
    # calls.
    #
    # Protocol tools are excluded: `step_complete`, `respond`, `escalate` and
    # friends are terminators explained by the loop protocol, and repeating
    # them here under a heading about the project index would be actively
    # misleading.
    remaining = sorted(available_tools - listed - _PROTOCOL_TOOLS)
    if remaining:
        lines.append("### Project index (MCP)")
        lines.append(
            "Backed by a semantic index of this repository, kept up to date "
            "outside the session. Reach for these when you are looking for "
            "something by *meaning* rather than by name, or when a previous "
            "session may already have answered the question."
        )
        for name in remaining:
            desc = TOOL_DESCRIPTIONS.get(name, ("", ""))[0]
            desc = desc or own_descriptions.get(name, "")
            lines.append(f"- **{name}**{f': {desc}' if desc else ''}")
        lines.append("")

    return "\n".join(lines)


def build_editing_rules(available_tools: set[str]) -> str:
    """Editing guidance, keyed to what is actually bound.

    Written as decision rules with observable triggers rather than advice.
    "Prefer surgical edits" asks the model to rate its own behaviour; "if the
    match is not unique, add surrounding lines" is something it can check.
    """
    rules = []
    if "help" in available_tools:
        rules.append('- Unsure how the editing tools work? Call help("edit").')
    if "edit_file" in available_tools:
        rules.extend([
            "- Change an existing file with edit_file(file_path, old_string, "
            "new_string). old_string must match byte for byte, indentation "
            "included.",
            "- Read the file in this step before editing it. The text you paste "
            "has to be the text on disk right now.",
            "- If the edit is refused for appearing more than once, add the "
            "lines above and below until the match is unique — do not guess.",
            "- Deleting is new_string=\"\".",
        ])
    if "create_file" in available_tools:
        rules.append(
            "- create_file is for files that do not exist yet; it fails if one does."
        )
    if "rename_symbol" in available_tools:
        rules.append(
            "- Renaming something used elsewhere: rename_symbol, not edit_file. "
            "It rewrites every reference and import; editing files one at a "
            "time leaves the others pointing at a name that is gone."
        )
    if "move_symbol" in available_tools:
        rules.append(
            "- Moving code between files: move_symbol, for the same reason — it "
            "fixes the imports on both sides."
        )
    if not rules:
        return ""
    return "## Editing Rules\n" + "\n".join(rules)


def build_editing_examples(
    available_tools: set[str],
    *,
    task_type: str = "feature",
) -> str:
    """One worked example per distinct operation, and no more.

    There used to be four, three of which showed the same shape with a
    different tool. Extra demonstrations of a pattern the model already has
    measurably cost accuracy on structured tasks — they compete for attention
    with the instruction they are meant to support.
    """
    examples = []

    if "edit_file" in available_tools:
        examples.append(
            "Example — change an existing file:\n"
            '  1. read_file: file_path="src/auth.py"   shows the current text\n'
            "  2. edit_file:\n"
            '     file_path="src/auth.py",\n'
            "     old_string=\"    if payload.get('exp', 0) < now:\",\n"
            "     new_string=\"    if payload.get('exp', 0) < time.time():\"\n"
            '  3. execute_command: "python -m pytest tests/test_auth.py -x -q"\n'
            "     Output: PASSED\n"
            '  4. step_complete: summary="Fixed expiry check. Test passes."'
        )

    if "create_file" in available_tools:
        examples.append(
            "Example — a file that does not exist yet:\n"
            '  1. create_file: file_path="validator.py", content="class Validator:\\n'
            '    def __init__(self):\\n        self.rules = []\\n"\n'
            "  2. execute_command: \"python -c 'from validator import Validator; "
            "print(type(Validator()))'\"\n"
            "     Output: <class 'validator.Validator'>\n"
            '  3. step_complete: summary="Created Validator skeleton"'
        )

    if not examples:
        return ""
    return "## Examples of Good Execution\n\n" + "\n\n".join(examples)


def build_anti_patterns(available_tools: set[str]) -> str:
    """Generate anti-patterns (NEVER do these) based on available tools."""
    patterns = []

    # Universal anti-patterns
    patterns.append(
        "1. Rewrite a whole file to change one function:\n"
        "   INSTEAD: edit_file the one block that changes."
    )
    patterns.append(
        "2. Edit without reading first:\n"
        "   INSTEAD: read the file this step; old_string must match it exactly."
    )
    patterns.append(
        "3. Fix things not in this step:\n"
        "   INSTEAD: ONE step means ONE change. Other fixes go in their own step."
    )
    patterns.append(
        "4. Skip verification:\n"
        "   INSTEAD: ALWAYS run a test or an import check after every edit."
    )
    patterns.append(
        "5. Keep trying after 3 consecutive failures:\n"
        '   INSTEAD: STOP. Call step_complete(status="blocked"). The design needs rethinking.'
    )
    patterns.append(
        "6. Add code that wasn't asked for:\n"
        "   INSTEAD: add no logging, docstrings, type hints, or error handling unless asked."
    )
    patterns.append(
        "7. Read the same file twice in one step:\n"
        "   INSTEAD: use what you already read. It is still in your context."
    )

    return "## NEVER Do These\n\n" + "\n\n".join(patterns)


def build_execute_prompt(
    *,
    available_tools: set[str],
    step_num: int,
    total_steps: int,
    step_title: str,
    step_files: str,
) -> str:
    """Build a complete execute prompt with only available tool references.

    This replaces the static BUG_EXECUTE / FEATURE_EXECUTE / etc. prompts
    with a dynamically generated version.
    """
    parts = [
        f"STEP {step_num}/{total_steps}: {step_title}",
        f"Files you may modify: {step_files}",
        "",
        "## RULES",
        "- ONLY modify the file(s) and function(s) described in this step",
        '- Do NOT refactor, clean up, or "improve" adjacent code',
        "- Do NOT add error handling for cases that can't happen",
        "- Do NOT add abstractions for one-time operations",
        "- Verify your edit with the smallest test target that executes the changed behavior",
        "- Call step_complete when done",
        "",
        build_editing_rules(available_tools),
        "",
        build_editing_examples(available_tools),
        "",
        build_anti_patterns(available_tools),
    ]

    return "\n".join(parts)


def get_available_tool_names(tools: list) -> set[str]:
    """Extract tool names from a list of tool instances."""
    names = set()
    for t in tools:
        name = getattr(t, "name", None) or getattr(t, "_name", None)
        if name:
            names.add(name)
    # Engine pseudo-tools are always available.
    names.update({"step_complete", "add_note", "add_session_note"})
    return names
