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
        "Read a file with line numbers",
        "read_file(file_path='src/main.py')",
    ),
    "partial_read": (
        "Read a specific line range of a file",
        "partial_read(file_path='src/main.py', start_line=10, end_line=50)",
    ),
    "create_file": (
        "Create a NEW file (fails if file exists)",
        "create_file(file_path='src/new.py', content='...')",
    ),
    "replace_lines": (
        "Replace a line range in an existing file (read first to get line numbers)",
        "replace_lines(file_path='src/main.py', content='new code', start_line=10, end_line=15)",
    ),
    "add_content_after_line": (
        "Insert content after a specific line",
        "add_content_after_line(file_path='src/main.py', line_number=10, content='new line')",
    ),
    "add_content_before_line": (
        "Insert content before a specific line",
        "add_content_before_line(file_path='src/main.py', line_number=10, content='new line')",
    ),
    "apply_patch": (
        "Apply a unified diff patch to one or more files",
        "apply_patch(patch='--- a/file\\n+++ b/file\\n@@ ... @@\\n...')",
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
        "Commit staged changes",
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
        "execute_command(command='python -m pytest tests/ -x -q')",
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
    # Knowledge
    "record_finding": (
        "Save a finding to the knowledge base",
        "record_finding(topic='auth module', content='uses JWT with HS256')",
    ),
    "search_findings": (
        "Search saved findings",
        "search_findings(query='auth')",
    ),
    "read_findings": (
        "Read all findings",
        "read_findings()",
    ),
    # Code intelligence
    "find_definition": (
        "Find where a function/class is defined",
        "find_definition(name='verify_token')",
    ),
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
    "edit_symbol": (
        "Replace a method/function body by symbol name",
        "edit_symbol(symbol='AuthService.verify_token', new_code='def verify_token(self, token): ...')",
    ),
    "add_symbol": (
        "Add a new method to a class or file",
        "add_symbol(file_path='src/auth.py', code='def new_method(self): ...', class_name='AuthService')",
    ),
    "remove_symbol": (
        "Remove a method/function by name",
        "remove_symbol(symbol='AuthService._old_helper')",
    ),
    "analyze_code": (
        "Detect broken imports, undefined symbols, unused code",
        "analyze_code(file_path='src/auth.py')",
    ),
    # Meta
    "help": (
        "Get detailed help and examples for any tool",
        "help(context='edit')",
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
    "think": (
        "Reason before acting (does not count as tool call)",
        "think(reasoning='The bug is in...')",
    ),
}


# ── Editing tool groups ──────────────────────────────────────────────────
# Used to generate conditional editing rules

_EDIT_TOOLS_SURGICAL = {"replace_lines", "edit_symbol"}
_EDIT_TOOLS_INSERT = {"add_content_after_line", "add_content_before_line"}
_EDIT_TOOLS_SYMBOL = {"edit_symbol", "add_symbol", "remove_symbol"}


def build_tool_usage_section(available_tools: set[str]) -> str:
    """Generate a '## Tool Usage' prompt section listing only available tools.

    Groups tools by category and includes usage hints.
    """
    categories = [
        ("Reading", ["read_file", "partial_read", "list_directory", "glob",
                     "code_search", "get_symbol_code", "list_symbols",
                     "search_symbols", "find_definition", "find_references",
                     "project_structure", "analyze_code"]),
        ("Writing", ["create_file", "replace_lines", "edit_symbol",
                     "add_symbol", "remove_symbol", "add_content_after_line",
                     "add_content_before_line", "apply_patch"]),
        ("Execution", ["execute_command"]),
        ("Git", ["git_branch", "git_commit", "git_diff", "git_status"]),
        ("Web", ["web_search", "web_fetch"]),
        ("Knowledge", ["record_finding", "search_findings", "read_findings"]),
        ("Meta", ["help"]),
    ]

    lines = ["## Tool Usage", ""]
    for category, tool_names in categories:
        present = [t for t in tool_names if t in available_tools]
        if not present:
            continue
        lines.append(f"### {category}")
        for name in present:
            desc, example = TOOL_DESCRIPTIONS.get(name, (name, ""))
            lines.append(f"- **{name}**: {desc}")
        lines.append("")

    return "\n".join(lines)


def build_editing_rules(available_tools: set[str]) -> str:
    """Generate editing rules based on which edit tools are available."""
    rules = []

    has_replace = "replace_lines" in available_tools
    has_edit_sym = "edit_symbol" in available_tools
    has_add_sym = "add_symbol" in available_tools
    has_remove_sym = "remove_symbol" in available_tools
    has_create = "create_file" in available_tools
    has_insert = "add_content_after_line" in available_tools
    has_help = "help" in available_tools

    if has_help:
        rules.append("- Call help(\"edit\") if unsure how the editing tools work.")

    if has_edit_sym:
        rules.append("- To replace a method/function: use edit_symbol (preferred — by symbol name)")
    if has_replace:
        rules.append("- To replace specific lines: use replace_lines (by line number — read file first)")
    if has_add_sym:
        rules.append("- To add a new method to a class: use add_symbol")
    if has_remove_sym:
        rules.append("- To delete a method: use remove_symbol")
    if has_insert:
        rules.append("- To insert lines: use add_content_after_line or add_content_before_line")
    if has_create:
        rules.append("- To create a new file: use create_file (fails if file exists)")

    if not rules:
        return ""

    return "## Editing Rules\n" + "\n".join(rules)


def build_editing_examples(
    available_tools: set[str],
    *,
    task_type: str = "feature",
) -> str:
    """Generate editing examples using only available tools."""
    examples = []

    has_replace = "replace_lines" in available_tools
    has_edit_sym = "edit_symbol" in available_tools
    has_add_sym = "add_symbol" in available_tools
    has_create = "create_file" in available_tools

    if has_replace:
        examples.append(
            "Example — Replace specific lines:\n"
            "  1. read_file: path=\"src/auth.py\" → see line numbers\n"
            "  2. replace_lines: file_path=\"src/auth.py\",\n"
            "     content=\"    if payload.get('exp', 0) < time.time():\\n\",\n"
            "     start_line=15, end_line=15\n"
            "  3. execute_command: \"python -m pytest tests/test_auth.py -x -q\"\n"
            "     → PASSED\n"
            "  4. step_complete: summary=\"Fixed expiry check. Test passes.\""
        )

    if has_edit_sym:
        examples.append(
            "Example — Replace a method by name:\n"
            "  1. edit_symbol:\n"
            "     symbol=\"AuthService.verify_token\",\n"
            "     new_code=\"    def verify_token(self, token):\\n"
            "        payload = self._decode(token)\\n"
            "        if not payload or payload.get('exp', 0) < time.time():\\n"
            "            return None\\n"
            "        return payload\"\n"
            "  2. execute_command: \"python -m pytest tests/test_auth.py -v\"\n"
            "     → 3 passed\n"
            "  3. step_complete: summary=\"Rewrote verify_token() with expiry check\""
        )

    if has_add_sym:
        examples.append(
            "Example — Add a new method to a class:\n"
            "  1. add_symbol:\n"
            "     file_path=\"validator.py\",\n"
            "     code=\"def add_rule(self, rule_func):\\n    self.rules.append(rule_func)\",\n"
            "     class_name=\"Validator\"\n"
            "  2. execute_command: \"python -c 'from validator import Validator; "
            "v = Validator(); v.add_rule(lambda x: True); print(len(v.rules))'\"\n"
            "     → 1\n"
            "  3. step_complete: summary=\"Added add_rule() to Validator\""
        )

    if has_create:
        examples.append(
            "Example — Create a new file:\n"
            "  1. create_file: path=\"validator.py\", content=\"class Validator:\\n"
            "    def __init__(self):\\n        self.rules = []\\n\"\n"
            "  2. execute_command: \"python -c 'from validator import Validator; print(type(Validator()))'\"\n"
            "     → <class 'validator.Validator'>\n"
            "  3. step_complete: summary=\"Created Validator skeleton\""
        )

    if not examples:
        return ""

    return "## Examples of Good Execution\n\n" + "\n\n".join(examples)


def build_anti_patterns(available_tools: set[str]) -> str:
    """Generate anti-patterns (NEVER do these) based on available tools."""
    patterns = []

    # Universal anti-patterns
    patterns.append(
        "1. Rewrite entire file to change one function:\n"
        "   → Use surgical edits (replace_lines, edit_symbol) instead."
    )
    patterns.append(
        "2. Edit without reading first:\n"
        "   → You need exact line numbers and function names. Read the file first."
    )
    patterns.append(
        "3. Fix things not in this step:\n"
        "   → ONE step = ONE change. Other fixes go in their own step."
    )
    patterns.append(
        "4. Skip verification:\n"
        "   → ALWAYS run a test or import check after every edit."
    )
    patterns.append(
        "5. Keep trying after 3 consecutive failures:\n"
        "   → STOP. Call step_complete(status=\"blocked\"). The design needs rethinking."
    )
    patterns.append(
        "6. Add code that wasn't asked for:\n"
        "   → No extra logging, docstrings, type hints, or error handling unless requested."
    )
    patterns.append(
        "7. Read the same file twice in one step:\n"
        "   → The content is already in your context after the first read."
    )

    return "## NEVER Do These\n\n" + "\n\n".join(patterns)


def build_execute_prompt(
    *,
    available_tools: set[str],
    step_num: int,
    total_steps: int,
    step_description: str,
    step_files: str,
) -> str:
    """Build a complete execute prompt with only available tool references.

    This replaces the static BUG_EXECUTE / FEATURE_EXECUTE / etc. prompts
    with a dynamically generated version.
    """
    parts = [
        f"STEP {step_num}/{total_steps}: {step_description}",
        f"Files you may modify: {step_files}",
        "",
        "## RULES",
        "- ONLY modify the file(s) and function(s) described in this step",
        "- Do NOT refactor, clean up, or \"improve\" adjacent code",
        "- Do NOT add error handling for cases that can't happen",
        "- Do NOT add abstractions for one-time operations",
        "- Verify your edit: run the relevant test",
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
    # Engine pseudo-tools are always available
    names.update({"step_complete", "add_note", "think"})
    return names
